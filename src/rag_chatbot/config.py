from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    pdf_directory: Path
    collection_name: str
    weaviate_url: str | None
    weaviate_api_key: str | None
    weaviate_host: str
    weaviate_port: int
    weaviate_grpc_port: int
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_mode: str = "hybrid"
    hybrid_alpha: float = 0.5
    retrieval_candidates: int = 10
    rerank_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_device: str = "cpu"
    reranker_batch_size: int = 16

    def __post_init__(self) -> None:
        if self.retrieval_mode not in {"hybrid", "vector"}:
            raise ValueError("RAG_RETRIEVAL_MODE must be 'hybrid' or 'vector'.")
        if not 0 <= self.hybrid_alpha <= 1:
            raise ValueError("RAG_HYBRID_ALPHA must be between 0 and 1.")
        if self.retrieval_candidates < 1:
            raise ValueError("RAG_RETRIEVAL_CANDIDATES must be positive.")
        if self.reranker_batch_size < 1:
            raise ValueError("RAG_RERANKER_BATCH_SIZE must be positive.")
        if not self.reranker_model.strip() or not self.reranker_device.strip():
            raise ValueError("Reranker model and device must not be empty.")


def _env_bool(name: str, default: str) -> bool:
    value = os.getenv(name, default).strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true, false, 1, or 0.")
    return value in {"true", "1"}


def load_settings() -> Settings:
    return Settings(
        pdf_directory=Path(
            os.getenv("RAG_PDF_DIRECTORY", PROJECT_ROOT / "data" / "pdfs")
        ),
        collection_name=os.getenv("WEAVIATE_COLLECTION", "EnterprisePolicyChunks"),
        # WEAVIATE_URI is retained for compatibility with the existing .env.
        weaviate_url=os.getenv("WEAVIATE_URL") or os.getenv("WEAVIATE_URI"),
        weaviate_api_key=os.getenv("WEAVIATE_API_KEY"),
        weaviate_host=os.getenv("WEAVIATE_HOST", "localhost"),
        weaviate_port=int(os.getenv("WEAVIATE_HTTP_PORT", os.getenv("WEAVIATE_PORT", "8080"))),
        weaviate_grpc_port=int(os.getenv("WEAVIATE_GRPC_PORT", "50051")),
        chat_model=os.getenv("CHAT_MODEL", "gpt-5.6-sol"),
        embedding_model=(
            os.getenv("EMBED_MODEL")
            or os.getenv("OPENAI_EMBEDDING_MODEL")
            or "text-embedding-3-small"
        ),
        chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "150")),
        retrieval_mode=os.getenv("RAG_RETRIEVAL_MODE", "hybrid").strip().lower(),
        hybrid_alpha=float(os.getenv("RAG_HYBRID_ALPHA", "0.5")),
        retrieval_candidates=int(os.getenv("RAG_RETRIEVAL_CANDIDATES", "10")),
        rerank_enabled=_env_bool("RAG_RERANK_ENABLED", "true"),
        reranker_model=os.getenv(
            "RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
        ),
        reranker_device=os.getenv("RAG_RERANKER_DEVICE", "cpu"),
        reranker_batch_size=int(os.getenv("RAG_RERANKER_BATCH_SIZE", "16")),
    )
