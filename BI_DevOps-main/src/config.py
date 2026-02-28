"""
config.py
---------
Centralised, environment-driven configuration using Pydantic Settings.
All secrets are read from environment variables or a .env file at startup.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "sentinel_db"
    POSTGRES_USER: str = "sentinel_user"
    POSTGRES_PASSWORD: str

    # Vector storage
    VECTOR_DB_PATH: str = "data/vector_index"

    # Embedding backend: "openai" | "huggingface"
    EMBEDDING_BACKEND: str = "openai"
    OPENAI_API_KEY: str = ""
    HF_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Ingestion
    DEFAULT_BATCH_SIZE: int = 512
    MAX_TOKENS_PER_CHUNK: int = 512


settings = Settings()
