from pathlib import Path

from platformdirs import user_data_dir
from pydantic_settings import BaseSettings

# XDG-compliant data directory
# Linux: ~/.local/share/tok
# macOS: ~/Library/Application Support/tok
# Windows: C:\Users\<user>\AppData\Local\tok
APP_NAME = "tok"
DATA_DIR = Path(user_data_dir(APP_NAME, appauthor=False))


def get_default_database_url() -> str:
    """Get default database URL in user data directory."""
    db_path = DATA_DIR / "tok.db"
    return f"sqlite+aiosqlite:///{db_path}"


def get_default_upload_dir() -> Path:
    """Get default upload directory in user data directory."""
    return DATA_DIR / "uploads"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Settings
    app_name: str = "Tree of Knowledge"
    debug: bool = False

    # Database (default: ~/.local/share/tok/tok.db)
    database_url: str = ""

    # File uploads (default: ~/.local/share/tok/uploads)
    upload_dir: Path = Path("")
    max_upload_size_mb: int = 10

    # Claude Agent SDK
    anthropic_api_key: str = ""

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.database_url:
            self.database_url = get_default_database_url()
        if self.upload_dir == Path(""):
            self.upload_dir = get_default_upload_dir()

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)

# Log the data location on import
print(f"ToK data directory: {DATA_DIR}")
