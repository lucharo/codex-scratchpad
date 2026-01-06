from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ulid import ULID

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.message import Message
    from app.models.tree import Tree


def generate_ulid() -> str:
    return str(ULID())


class BranchOrigin(Base):
    """Stores information about how a branch was created from highlighted text.

    This is the key innovation - we track exactly which text was highlighted
    and what prompt the user entered to create the branch.
    """

    __tablename__ = "branch_origins"

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=generate_ulid)
    branch_id: Mapped[str] = mapped_column(
        String(26), ForeignKey("branches.id"), unique=True
    )

    # Source information
    source_message_id: Mapped[str] = mapped_column(String(26), ForeignKey("messages.id"))
    source_branch_id: Mapped[str] = mapped_column(String(26), ForeignKey("branches.id"))

    # The exact text selection
    highlight_start: Mapped[int] = mapped_column(Integer)  # Character offset
    highlight_end: Mapped[int] = mapped_column(Integer)
    highlighted_text: Mapped[str] = mapped_column(Text)  # The actual text

    # User's prompt for the branch
    user_prompt: Mapped[str] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    branch: Mapped["Branch"] = relationship(
        "Branch", back_populates="origin", foreign_keys=[branch_id]
    )


class Branch(Base):
    """A Branch is a thread of conversation within a Tree.

    The root branch has no parent. Child branches are created when
    users highlight text and start a new conversation from that point.
    """

    __tablename__ = "branches"

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=generate_ulid)
    tree_id: Mapped[str] = mapped_column(String(26), ForeignKey("trees.id"))
    parent_branch_id: Mapped[str | None] = mapped_column(
        String(26), ForeignKey("branches.id"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), default="Main")

    # Claude session tracking (for resuming conversations)
    session_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    tree: Mapped["Tree"] = relationship("Tree", back_populates="branches")
    parent_branch: Mapped[Optional["Branch"]] = relationship(
        "Branch", remote_side=[id], backref="child_branches"
    )
    messages: Mapped[list["Message"]] = relationship(
        "Message", back_populates="branch", cascade="all, delete-orphan",
        order_by="Message.position"
    )
    origin: Mapped[Optional["BranchOrigin"]] = relationship(
        "BranchOrigin",
        back_populates="branch",
        uselist=False,
        foreign_keys="BranchOrigin.branch_id",
        cascade="all, delete-orphan",
    )

    @property
    def is_root(self) -> bool:
        """Check if this is the root branch."""
        return self.parent_branch_id is None
