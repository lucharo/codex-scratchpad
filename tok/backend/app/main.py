"""Tree of Knowledge - Main FastAPI Application.

A chat application with text-based conversation branching powered by Claude Agent SDK.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.database import init_db
from app.api import trees, messages, uploads

# Path to frontend build (when bundled)
FRONTEND_DIR = Path(__file__).parent.parent.parent.parent / "frontend" / "dist"


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


@app.get("/api/health")
async def health():
    """API health check."""
    return {"status": "ok"}


# Serve frontend in production (when dist/ exists)
if FRONTEND_DIR.exists():
    # Serve static assets (JS, CSS, images)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    # Serve other static files (favicon, etc.)
    @app.get("/tree.svg")
    async def favicon():
        return FileResponse(FRONTEND_DIR / "tree.svg")

    # SPA catch-all: serve index.html for all non-API routes
    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        """Serve the SPA for all non-API routes."""
        # Don't intercept API routes
        if full_path.startswith("api/") or full_path.startswith("files/"):
            return {"error": "not found"}

        # Check if it's a static file
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # Otherwise serve index.html (SPA routing)
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    # Development mode - just show API info
    @app.get("/")
    async def root():
        """API info (development mode)."""
        return {
            "name": settings.app_name,
            "status": "healthy",
            "version": "0.1.0",
            "mode": "development",
            "note": "Run 'just build' to bundle frontend for production",
        }
