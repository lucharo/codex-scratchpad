"""Pydantic schemas for API request/response validation."""

from datetime import datetime

from pydantic import BaseModel, Field

# === Tree Schemas ===

class TreeCreate(BaseModel):
    name: str = Field(default="New Tree", max_length=255)


class TreeUpdate(BaseModel):
    name: str | None = Field(None, max_length=255)


class TreeResponse(BaseModel):
    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    branch_count: int = 0

    class Config:
        from_attributes = True


# === Branch Schemas ===

class BranchOriginResponse(BaseModel):
    source_message_id: str
    source_branch_id: str
    highlight_start: int
    highlight_end: int
    highlighted_text: str
    user_prompt: str

    class Config:
        from_attributes = True


class BranchResponse(BaseModel):
    id: str
    tree_id: str
    parent_branch_id: str | None
    name: str
    is_root: bool
    origin: BranchOriginResponse | None = None
    message_count: int = 0
    created_at: datetime

    class Config:
        from_attributes = True


class BranchCreate(BaseModel):
    """Create a new branch from highlighted text."""
    source_message_id: str
    source_branch_id: str
    highlight_start: int
    highlight_end: int
    highlighted_text: str
    user_prompt: str
    name: str | None = None


# === Message Schemas ===

class ToolCallResponse(BaseModel):
    id: str
    tool_use_id: str
    name: str
    input_data: dict
    result: str | None
    is_error: bool

    class Config:
        from_attributes = True


class AttachmentResponse(BaseModel):
    id: str
    filename: str
    file_type: str
    file_size: int

    class Config:
        from_attributes = True


class MessageResponse(BaseModel):
    id: str
    branch_id: str
    role: str
    content: str
    thinking: str | None = None
    position: int
    model: str | None = None
    tool_calls: list[ToolCallResponse] = []
    attachments: list[AttachmentResponse] = []
    created_at: datetime

    class Config:
        from_attributes = True


class MessageCreate(BaseModel):
    """Send a new message."""
    content: str
    # attachment_ids: list[str] = []  # For pre-uploaded files


# === Full Tree Response ===

class BranchWithMessages(BranchResponse):
    messages: list[MessageResponse] = []


class TreeDetailResponse(TreeResponse):
    branches: list[BranchWithMessages] = []
