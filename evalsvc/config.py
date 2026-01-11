"""Service configuration, read from environment variables (prefix EVALSVC_) or a .env file."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="EVALSVC_", extra="ignore")

    # Storage. SQLite works for local dev; docker-compose points this at PostgreSQL.
    database_url: str = "sqlite:///./evalsvc.db"

    # Queue. When unset, batch items run on an in-process thread pool instead of Celery.
    redis_url: str | None = None
    # Run batch items inline inside the request (used by tests).
    sync_tasks: bool = False

    # Judges
    default_judge: str = "heuristic"
    anthropic_model: str = "claude-opus-5-5"
    openai_model: str = "gpt-4.1"
    local_base_url: str = "http://localhost:11434/v1"  # Ollama's OpenAI-compatible endpoint
    local_model: str = "llama3.1"
    judge_max_retries: int = 2
    judge_concurrency: int = 4
    judge_max_tokens: int = 8000

    # Scoring policy
    low_confidence_threshold: float = 0.6
    borderline_margin: float = 0.25
    disagreement_threshold: int = 2

    # Cost tracking and caching
    pricing_json: str | None = None      # extra/override prices: {"model": [input_per_mtok, output_per_mtok]}
    judge_cache: bool = True             # reuse verdicts for identical (judge, rubric dimension, item) requests

    # Optional shared-secret auth. When set, every request needs header X-API-Key.
    api_key: str | None = None


settings = Settings()
