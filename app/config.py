from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    database_url: str = "sqlite+aiosqlite:///./data/movies.db"

    embedding_model: str = "all-MiniLM-L6-v2"
    faiss_index_path: str = "./data/faiss.index"
    movie_ids_path: str = "./data/movie_ids.npy"

    log_level: str = "INFO"
    max_retrieval_results: int = 10


@lru_cache
def get_settings() -> Settings:
    return Settings()
