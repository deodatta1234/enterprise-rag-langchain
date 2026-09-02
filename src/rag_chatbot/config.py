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
    )
