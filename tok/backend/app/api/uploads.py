"""API routes for file uploads."""


import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ulid import ULID

from app.api.schemas import AttachmentResponse
from app.core.config import settings
from app.core.database import get_db
from app.models import Attachment, Message

router = APIRouter(prefix="/upload", tags=["uploads"])

# Allowed file types
ALLOWED_TYPES = {
    # Images
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    # Documents
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/json": ".json",
    # Code
    "text/x-python": ".py",
    "text/javascript": ".js",
    "text/typescript": ".ts",
}


@router.post("", response_model=AttachmentResponse)
async def upload_file(
    file: UploadFile = File(...),
    message_id: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Upload a file.

    If message_id is provided, attach to that message.
    Otherwise, create an orphan attachment that can be attached later.
    """
    # Validate file type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type {content_type} not allowed. Allowed: {list(ALLOWED_TYPES.keys())}",
        )

    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset

    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB",
        )

    # Verify message exists if provided
    if message_id:
        stmt = select(Message).where(Message.id == message_id)
        result = await db.execute(stmt)
        if not result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Message not found")

    # Generate unique filename
    file_id = str(ULID())
    extension = ALLOWED_TYPES.get(content_type, "")
    safe_filename = f"{file_id}{extension}"
    file_path = settings.upload_dir / safe_filename

    # Save file
    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Create attachment record
    attachment = Attachment(
        message_id=message_id or "",  # Empty if orphan
        filename=file.filename or safe_filename,
        file_path=str(file_path),
        file_type=content_type,
        file_size=file_size,
    )
    db.add(attachment)
    await db.commit()
    await db.refresh(attachment)

    return attachment


@router.get("/{attachment_id}")
async def get_file(
    attachment_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get file info (not the file content - use static serving for that)."""
    stmt = select(Attachment).where(Attachment.id == attachment_id)
    result = await db.execute(stmt)
    attachment = result.scalar_one_or_none()

    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")

    return AttachmentResponse(
        id=attachment.id,
        filename=attachment.filename,
        file_type=attachment.file_type,
        file_size=attachment.file_size,
    )
