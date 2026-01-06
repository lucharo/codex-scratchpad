"""API routes for Trees and Branches."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models import Tree, Branch, Message
from app.api.schemas import (
    TreeCreate,
    TreeUpdate,
    TreeResponse,
    TreeDetailResponse,
    BranchResponse,
    BranchCreate,
    BranchWithMessages,
)

router = APIRouter(prefix="/trees", tags=["trees"])


@router.post("", response_model=TreeResponse)
async def create_tree(
    tree_in: TreeCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new Tree with a root branch."""
    # Create tree
    tree = Tree(name=tree_in.name)
    db.add(tree)
    await db.flush()

    # Create root branch
    root_branch = Branch(tree_id=tree.id, name="Main")
    db.add(root_branch)
    await db.commit()

    return TreeResponse(
        id=tree.id,
        name=tree.name,
        created_at=tree.created_at,
        updated_at=tree.updated_at,
        branch_count=1,
    )


@router.get("", response_model=list[TreeResponse])
async def list_trees(db: AsyncSession = Depends(get_db)):
    """List all Trees."""
    # Get trees with branch counts
    stmt = (
        select(Tree, func.count(Branch.id).label("branch_count"))
        .outerjoin(Branch)
        .group_by(Tree.id)
        .order_by(Tree.updated_at.desc())
    )
    result = await db.execute(stmt)
    rows = result.all()

    return [
        TreeResponse(
            id=tree.id,
            name=tree.name,
            created_at=tree.created_at,
            updated_at=tree.updated_at,
            branch_count=branch_count,
        )
        for tree, branch_count in rows
    ]


@router.get("/{tree_id}", response_model=TreeDetailResponse)
async def get_tree(
    tree_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a Tree with all its branches and messages."""
    stmt = (
        select(Tree)
        .where(Tree.id == tree_id)
        .options(
            selectinload(Tree.branches).selectinload(Branch.messages).selectinload(Message.tool_calls),
            selectinload(Tree.branches).selectinload(Branch.messages).selectinload(Message.attachments),
            selectinload(Tree.branches).selectinload(Branch.origin),
        )
    )
    result = await db.execute(stmt)
    tree = result.scalar_one_or_none()

    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")

    return tree


@router.patch("/{tree_id}", response_model=TreeResponse)
async def update_tree(
    tree_id: str,
    tree_in: TreeUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a Tree."""
    stmt = select(Tree).where(Tree.id == tree_id)
    result = await db.execute(stmt)
    tree = result.scalar_one_or_none()

    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")

    if tree_in.name is not None:
        tree.name = tree_in.name

    await db.commit()

    # Get branch count
    count_stmt = select(func.count(Branch.id)).where(Branch.tree_id == tree_id)
    count_result = await db.execute(count_stmt)
    branch_count = count_result.scalar() or 0

    return TreeResponse(
        id=tree.id,
        name=tree.name,
        created_at=tree.created_at,
        updated_at=tree.updated_at,
        branch_count=branch_count,
    )


@router.delete("/{tree_id}")
async def delete_tree(
    tree_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a Tree and all its branches."""
    stmt = select(Tree).where(Tree.id == tree_id)
    result = await db.execute(stmt)
    tree = result.scalar_one_or_none()

    if not tree:
        raise HTTPException(status_code=404, detail="Tree not found")

    await db.delete(tree)
    await db.commit()

    return {"status": "deleted"}


# === Branch Routes ===


@router.get("/{tree_id}/branches", response_model=list[BranchResponse])
async def list_branches(
    tree_id: str,
    db: AsyncSession = Depends(get_db),
):
    """List all branches for a Tree."""
    stmt = (
        select(Branch, func.count(Message.id).label("message_count"))
        .where(Branch.tree_id == tree_id)
        .outerjoin(Message)
        .group_by(Branch.id)
        .options(selectinload(Branch.origin))
        .order_by(Branch.created_at)
    )
    result = await db.execute(stmt)
    rows = result.all()

    return [
        BranchResponse(
            id=branch.id,
            tree_id=branch.tree_id,
            parent_branch_id=branch.parent_branch_id,
            name=branch.name,
            is_root=branch.is_root,
            origin=branch.origin,
            message_count=message_count,
            created_at=branch.created_at,
        )
        for branch, message_count in rows
    ]


@router.post("/{tree_id}/branches", response_model=BranchResponse)
async def create_branch(
    tree_id: str,
    branch_in: BranchCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new branch from highlighted text."""
    from app.models import BranchOrigin

    # Verify tree exists
    tree_stmt = select(Tree).where(Tree.id == tree_id)
    tree_result = await db.execute(tree_stmt)
    if not tree_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Tree not found")

    # Verify source message exists and belongs to this tree
    msg_stmt = (
        select(Message)
        .join(Branch)
        .where(Message.id == branch_in.source_message_id)
        .where(Branch.tree_id == tree_id)
    )
    msg_result = await db.execute(msg_stmt)
    if not msg_result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Source message not found")

    # Generate branch name from highlighted text
    branch_name = branch_in.name or f"Branch: {branch_in.highlighted_text[:30]}..."

    # Create branch
    branch = Branch(
        tree_id=tree_id,
        parent_branch_id=branch_in.source_branch_id,
        name=branch_name,
    )
    db.add(branch)
    await db.flush()

    # Create branch origin
    origin = BranchOrigin(
        branch_id=branch.id,
        source_message_id=branch_in.source_message_id,
        source_branch_id=branch_in.source_branch_id,
        highlight_start=branch_in.highlight_start,
        highlight_end=branch_in.highlight_end,
        highlighted_text=branch_in.highlighted_text,
        user_prompt=branch_in.user_prompt,
    )
    db.add(origin)
    await db.commit()

    # Refresh to get origin
    await db.refresh(branch)

    return BranchResponse(
        id=branch.id,
        tree_id=branch.tree_id,
        parent_branch_id=branch.parent_branch_id,
        name=branch.name,
        is_root=branch.is_root,
        origin=origin,
        message_count=0,
        created_at=branch.created_at,
    )


@router.get("/{tree_id}/branches/{branch_id}", response_model=BranchWithMessages)
async def get_branch(
    tree_id: str,
    branch_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a branch with all its messages."""
    stmt = (
        select(Branch)
        .where(Branch.id == branch_id)
        .where(Branch.tree_id == tree_id)
        .options(
            selectinload(Branch.messages).selectinload(Message.tool_calls),
            selectinload(Branch.messages).selectinload(Message.attachments),
            selectinload(Branch.origin),
        )
    )
    result = await db.execute(stmt)
    branch = result.scalar_one_or_none()

    if not branch:
        raise HTTPException(status_code=404, detail="Branch not found")

    return branch
