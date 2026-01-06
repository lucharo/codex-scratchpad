"""Tree of Knowledge - Main FastAPI Application.

A chat application with text-based conversation branching powered by Claude Agent SDK.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import init_db
from app.api import trees, messages, uploads


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(
    title=settings.app_name,
    description="Chat with Claude featuring text-based conversation branching",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for uploads
app.mount("/files", StaticFiles(directory=str(settings.upload_dir)), name="files")

# API routes
app.include_router(trees.router, prefix="/api")
app.include_router(messages.router, prefix="/api")
app.include_router(uploads.router, prefix="/api")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": settings.app_name,
        "status": "healthy",
        "version": "0.1.0",
    }


@app.get("/api/health")
async def health():
    """API health check."""
    return {"status": "ok"}
