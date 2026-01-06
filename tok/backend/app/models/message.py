from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from ulid import ULID

from app.core.database import Base

if TYPE_CHECKING:
    from app.models.branch import Branch


def generate_ulid() -> str:
    return str(ULID())


class ToolCall(Base):
    """Represents a tool call made by Claude during a response."""

    __tablename__ = "tool_calls"

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=generate_ulid)
    message_id: Mapped[str] = mapped_column(String(26), ForeignKey("messages.id"))

    # Tool information
    tool_use_id: Mapped[str] = mapped_column(String(255))  # Claude's tool_use_id
    name: Mapped[str] = mapped_column(String(255))  # e.g., "Read", "Bash"
    input_data: Mapped[dict] = mapped_column(JSON, default=dict)  # Tool input

    # Result
    result: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_error: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    message: Mapped["Message"] = relationship("Message", back_populates="tool_calls")


class Attachment(Base):
    """File or image attachment on a message."""

    __tablename__ = "attachments"

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=generate_ulid)
    message_id: Mapped[str] = mapped_column(String(26), ForeignKey("messages.id"))

    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(512))  # Path on server
    file_type: Mapped[str] = mapped_column(String(100))  # MIME type
    file_size: Mapped[int] = mapped_column(Integer)  # Bytes

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    message: Mapped["Message"] = relationship("Message", back_populates="attachments")


class Message(Base):
    """A single message in a conversation branch.

    Messages can be from the user or assistant, and assistant messages
    may include thinking content and tool calls.
    """

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(26), primary_key=True, default=generate_ulid)
    branch_id: Mapped[str] = mapped_column(String(26), ForeignKey("branches.id"))

    # Message content
    role: Mapped[str] = mapped_column(String(20))  # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text, default="")

    # Extended thinking (for assistant messages)
    thinking: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Position in conversation (for ordering)
    position: Mapped[int] = mapped_column(Integer)

    # Metadata
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    branch: Mapped["Branch"] = relationship("Branch", back_populates="messages")
    tool_calls: Mapped[list["ToolCall"]] = relationship(
        "ToolCall", back_populates="message", cascade="all, delete-orphan"
    )
    attachments: Mapped[list["Attachment"]] = relationship(
        "Attachment", back_populates="message", cascade="all, delete-orphan"
    )

    # Track branches created from this message's text
    # (stored as IDs since branches reference back to messages)
