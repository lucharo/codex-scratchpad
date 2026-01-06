"""API routes for Messages with streaming support."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models import Tree, Branch, Message, ToolCall
from app.api.schemas import MessageCreate, MessageResponse
from app.services.claude_service import claude_service, StreamEvent

router = APIRouter(prefix="/trees/{tree_id}/branches/{branch_id}/messages", tags=["messages"])


async def build_conversation_context(
    db: AsyncSession,
    branch: Branch,
    include_branch_origin: bool = True,
) -> list[dict]:
    """Build conversation history for context.

    For branched conversations, this includes:
    1. All messages from parent branches up to the branch point
    2. The branch origin (highlighted text + prompt)
    3. All messages in the current branch
    """
    context = []

    # If this is a branched conversation, get parent context
    if branch.origin:
        # Get parent branch messages up to the source message
        parent_stmt = (
            select(Message)
            .where(Message.branch_id == branch.origin.source_branch_id)
            .order_by(Message.position)
        )
        parent_result = await db.execute(parent_stmt)
        parent_messages = parent_result.scalars().all()

        for msg in parent_messages:
            context.append({
                "role": msg.role,
                "content": msg.content,
            })
            # Stop after the source message
            if msg.id == branch.origin.source_message_id:
                break

        # Add branch origin context if requested
        if include_branch_origin:
            context.append({
                "role": "system",
                "content": f"[BRANCH POINT: User highlighted the following text]\n\n\"{branch.origin.highlighted_text}\"\n\n[User's question about this text]: {branch.origin.user_prompt}",
            })

    # Add messages from current branch
    for msg in branch.messages:
        context.append({
            "role": msg.role,
            "content": msg.content,
        })

    return context


@router.post("")
async def send_message(
    tree_id: str,
    branch_id: str,
    message_in: MessageCreate,
    db: AsyncSession = Depends(get_db),
):
    """Send a message and stream the response."""
    # Verify branch exists and belongs to tree
    stmt = (
        select(Branch)
        .where(Branch.id == branch_id)
        .where(Branch.tree_id == tree_id)
        .options(
            selectinload(Branch.messages),
            selectinload(Branch.origin),
        )
    )
    result = await db.execute(stmt)
    branch = result.scalar_one_or_none()

    if not branch:
        raise HTTPException(status_code=404, detail="Branch not found")

    # Calculate next position
    next_position = len(branch.messages)

    # Create user message
    user_message = Message(
        branch_id=branch_id,
        role="user",
        content=message_in.content,
        position=next_position,
    )
    db.add(user_message)
    await db.commit()
    await db.refresh(user_message)

    # Build context
    await db.refresh(branch, ["messages", "origin"])
    context = await build_conversation_context(db, branch)

    # Build prompt from context
    prompt_parts = []
    for msg in context:
        if msg["role"] == "user":
            prompt_parts.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            prompt_parts.append(f"Assistant: {msg['content']}")
        else:
            prompt_parts.append(msg["content"])  # System messages

    prompt = "\n\n".join(prompt_parts)

    # Stream response
    async def generate():
        collected_text = ""
        collected_thinking = ""
        collected_tool_calls = []
        session_id = None
        model = None

        async for event in claude_service.stream_response(prompt):
            # Send SSE event to client
            yield event.to_sse()

            # Collect response parts
            if event.type == "text":
                collected_text += event.content or ""
                model = event.model
            elif event.type == "thinking":
                collected_thinking += event.content or ""
            elif event.type == "tool_use":
                collected_tool_calls.append({
                    "name": event.tool_name,
                    "input": event.tool_input,
                })
            elif event.type == "tool_result":
                # Match to last tool call
                if collected_tool_calls:
                    collected_tool_calls[-1]["result"] = event.tool_result
                    collected_tool_calls[-1]["is_error"] = event.is_error
            elif event.type == "complete":
                session_id = event.session_id

        # Save assistant message to database
        async with db.begin():
            assistant_message = Message(
                branch_id=branch_id,
                role="assistant",
                content=collected_text,
                thinking=collected_thinking if collected_thinking else None,
                position=next_position + 1,
                model=model,
            )
            db.add(assistant_message)
            await db.flush()

            # Save tool calls
            for tc in collected_tool_calls:
                tool_call = ToolCall(
                    message_id=assistant_message.id,
                    tool_use_id=f"tool_{assistant_message.id}_{len(collected_tool_calls)}",
                    name=tc["name"],
                    input_data=tc.get("input", {}),
                    result=tc.get("result"),
                    is_error=tc.get("is_error", False),
                )
                db.add(tool_call)

            # Update branch session_id
            if session_id:
                branch_update = await db.get(Branch, branch_id)
                if branch_update:
                    branch_update.session_id = session_id

        # Send final message with saved message ID
        yield f"data: {{\"type\": \"saved\", \"message_id\": \"{assistant_message.id}\"}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("", response_model=list[MessageResponse])
async def list_messages(
    tree_id: str,
    branch_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List all messages in a branch."""
    # Verify branch exists
    branch_stmt = (
        select(Branch)
        .where(Branch.id == branch_id)
        .where(Branch.tree_id == tree_id)
    )
    branch_result = await db.execute(branch_stmt)
    if not branch_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Branch not found")

    # Get messages
    stmt = (
        select(Message)
        .where(Message.branch_id == branch_id)
        .options(
            selectinload(Message.tool_calls),
            selectinload(Message.attachments),
        )
        .order_by(Message.position)
    )
    result = await db.execute(stmt)
    messages = result.scalars().all()

    return messages


@router.get("/{message_id}", response_model=MessageResponse)
async def get_message(
    tree_id: str,
    branch_id: str,
    message_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific message."""
    stmt = (
        select(Message)
        .where(Message.id == message_id)
        .where(Message.branch_id == branch_id)
        .options(
            selectinload(Message.tool_calls),
            selectinload(Message.attachments),
        )
    )
    result = await db.execute(stmt)
    message = result.scalar_one_or_none()

    if not message:
        raise HTTPException(status_code=404, detail="Message not found")

    # Verify branch belongs to tree
    branch_stmt = select(Branch).where(Branch.id == branch_id).where(Branch.tree_id == tree_id)
    branch_result = await db.execute(branch_stmt)
    if not branch_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Branch not found")

    return message
