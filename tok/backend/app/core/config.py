from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    app_name: str = "Tree of Knowledge"
    debug: bool = False

    # Database
    database_url: str = "sqlite+aiosqlite:///./tok.db"

    # File uploads
    upload_dir: Path = Path("./uploads")
    max_upload_size_mb: int = 10

    # Claude Agent SDK
    anthropic_api_key: str = ""

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure upload directory exists
settings.upload_dir.mkdir(parents=True, exist_ok=True)
