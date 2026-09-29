from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# apps/api/.env, regardless of the working directory the app is started from.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
LOCAL_DATABASE_URL = "postgresql+asyncpg://voicelog:voicelog_local@127.0.0.1:5432/voicelog"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    environment: Literal["local", "test", "production"] = "local"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    database_url: str = LOCAL_DATABASE_URL

    # Without a key, notes are saved without AI metadata or embeddings, and audio
    # upload and search return 503.
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-5.4-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_transcription_model: str = "gpt-4o-mini-transcribe"

    @model_validator(mode="after")
    def _require_production_settings(self) -> Self:
        """Fail at startup rather than run production on local defaults."""
        if self.environment != "production":
            return self
        missing = []
        if self.database_url == LOCAL_DATABASE_URL:
            missing.append("DATABASE_URL")
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value():
            missing.append("OPENAI_API_KEY")
        if missing:
            raise ValueError(f"Set {', '.join(missing)} when ENVIRONMENT=production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
