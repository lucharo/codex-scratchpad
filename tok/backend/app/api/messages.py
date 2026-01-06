"""API routes for Messages with streaming support."""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.schemas import MessageCreate, MessageResponse
from app.core.database import get_db
from app.models import Branch, Message, ToolCall
from app.services.claude_service import claude_service
from app.services.context import (
    BranchOriginContext,
    MessageContext,
    build_branch_context,
    build_linear_context,
    build_prompt_from_context,
)

router = APIRouter(prefix="/trees/{tree_id}/branches/{branch_id}/messages", tags=["messages"])


async def _get_branch_or_404(
    db: AsyncSession,
    tree_id: str,
    branch_id: str,
) -> Branch:
    """Fetch branch with messages and origin, or raise 404."""
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

    return branch


async def _fetch_parent_messages(
    db: AsyncSession,
    source_branch_id: str,
    source_message_id: str,
) -> list[MessageContext]:
    """Fetch parent branch messages up to source message."""
    stmt = (
        select(Message)
        .where(Message.branch_id == source_branch_id)
        .order_by(Message.position)
    )
    result = await db.execute(stmt)
    parent_messages = result.scalars().all()

    context = []
    for msg in parent_messages:
        context.append(MessageContext(role=msg.role, content=msg.content))
        if msg.id == source_message_id:
            break

    return context


async def _build_context_for_branch(db: AsyncSession, branch: Branch) -> list[MessageContext]:
    """Build conversation context for a branch."""
    current_messages = [
        MessageContext(role=msg.role, content=msg.content)
        for msg in branch.messages
    ]

    if branch.origin:
        parent_messages = await _fetch_parent_messages(
            db,
            branch.origin.source_branch_id,
            branch.origin.source_message_id,
        )
        return build_branch_context(
            parent_messages=parent_messages,
            origin=BranchOriginContext(
                highlighted_text=branch.origin.highlighted_text,
                user_prompt=branch.origin.user_prompt,
            ),
            current_messages=current_messages,
        )

    return build_linear_context(current_messages)


@router.post("")
async def send_message(
    tree_id: str,
    branch_id: str,
    message_in: MessageCreate,
    db: AsyncSession = Depends(get_db),
):
    """Send a message and stream the response."""
    branch = await _get_branch_or_404(db, tree_id, branch_id)
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

    # Rebuild context with new message
    await db.refresh(branch, ["messages", "origin"])
    context = await _build_context_for_branch(db, branch)
    prompt = build_prompt_from_context(context)

    async def generate():
        collected_text = ""
        collected_thinking = ""
        collected_tool_calls = []
        session_id = None
        model = None

        async for event in claude_service.stream_response(prompt):
            yield event.to_sse()

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
            elif event.type == "tool_result" and collected_tool_calls:
                collected_tool_calls[-1]["result"] = event.tool_result
                collected_tool_calls[-1]["is_error"] = event.is_error
            elif event.type == "complete":
                session_id = event.session_id

        # Save assistant message
        async with db.begin():
            assistant_message = Message(
                branch_id=branch_id,
                role="assistant",
                content=collected_text,
                thinking=collected_thinking or None,
                position=next_position + 1,
                model=model,
            )
            db.add(assistant_message)
            await db.flush()

            for tc in collected_tool_calls:
                db.add(ToolCall(
                    message_id=assistant_message.id,
                    tool_use_id=f"tool_{assistant_message.id}_{len(collected_tool_calls)}",
                    name=tc["name"],
                    input_data=tc.get("input", {}),
                    result=tc.get("result"),
                    is_error=tc.get("is_error", False),
                ))

            if session_id:
                branch_update = await db.get(Branch, branch_id)
                if branch_update:
                    branch_update.session_id = session_id

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
    await _get_branch_or_404(db, tree_id, branch_id)

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
    return result.scalars().all()


@router.get("/{message_id}", response_model=MessageResponse)
async def get_message(
    tree_id: str,
    branch_id: str,
    message_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific message."""
    await _get_branch_or_404(db, tree_id, branch_id)

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

    return message
