"""Application configuration and environment settings.

All settings are read from environment variables or a local .env file
using pydantic-settings. Defaults are provided for local development.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings for Document Q&A Assistant."""

    # LLM Settings (OpenAI-compatible: OpenAI, Groq, Ollama, etc.)
    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # Storage Paths
    pdf_data_dir: str = "./data"
    chroma_dir: str = "./data/chroma"
    chroma_collection: str = "document_qa"
    bm25_index_path: str = "./data/bm25_index.pkl"

    # Ingestion / Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval Settings
    top_k: int = 4
    rerank_candidates: int = 15
    retrieval_strategy: Literal["dense", "hybrid", "hybrid_rerank"] = "dense"

    # Local Model Names (FastEmbed lightweight ONNX models)
    fastembed_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # LangGraph Workflow Settings
    max_rewrites: int = 1

    # MLflow Tracking
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "document-qa-rag"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()
