from datetime import datetime
from typing import TYPE_CHECKING
from sqlalchemy import String, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ulid import ULID

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.branch import Branch


def generate_ulid() -> str:
    return str(ULID())


class Tree(Base):
    """A Tree represents a conversation with all its branches.

    Think of it like a git repository - it contains multiple branches
    that can diverge from any point in the conversation.
    """

    __tablename__ = "trees"

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=generate_ulid)
    name: Mapped[str] = mapped_column(String(255), default="New Tree")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    branches: Mapped[list["Branch"]] = relationship(
        "Branch", back_populates="tree", cascade="all, delete-orphan"
    )

    @property
    def root_branch(self) -> "Branch | None":
        """Get the root branch (main conversation thread)."""
        for branch in self.branches:
            if branch.parent_branch_id is None:
                return branch
        return None
