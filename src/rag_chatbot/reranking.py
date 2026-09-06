"""Local cross-encoder reranking of already-authorized candidates."""

from __future__ import annotations

import math
from pathlib import Path
from functools import lru_cache
from threading import Lock
from typing import Any

from langchain_core.documents import Document
from langsmith import traceable

from .config import Settings


class LocalCrossEncoder:
    def __init__(self, model_name: str, device: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=512,
            local_files_only=Path(model_name).is_dir(),
        )
        # FastAPI can invoke the shared model from concurrent worker threads.
        self.lock = Lock()

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> Any:
        with self.lock:
            return self.model.predict(pairs, **kwargs)


@lru_cache(maxsize=1)
def _cached_reranker(model_name: str, device: str) -> LocalCrossEncoder:
    return LocalCrossEncoder(model_name, device)


_load_lock = Lock()


def get_reranker(settings: Settings) -> LocalCrossEncoder:
    """Load lazily and reuse one model per process for the active configuration."""
    # lru_cache alone can initialize twice during concurrent first requests.
    with _load_lock:
        return _cached_reranker(settings.reranker_model, settings.reranker_device)


@traceable(
    name="local_cross_encoder_rerank",
    process_inputs=lambda inputs: {
        key: inputs[key] for key in ("question", "documents", "k") if key in inputs
    },
)
def rerank_documents(
    question: str,
    documents: list[Document],
    settings: Settings,
    *,
    k: int,
    reranker: Any = None,
) -> list[Document]:
    """Score question/passage pairs, preserve source metadata, return the top k.

    Scores are ranking signals, not calibrated probabilities of answerability.
    Failure is propagated rather than silently changing retrieval behavior.
    """
    if k < 1:
        raise ValueError("k must be positive.")
    if not documents:
        return []

    model = reranker if reranker is not None else get_reranker(settings)
    pairs = [
        (
            question,
            f"Document: {doc.metadata.get('document_id', '')}\n{doc.page_content}",
        )
        for doc in documents
    ]
    scores = model.predict(
        pairs,
        batch_size=settings.reranker_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    if len(scores) != len(documents):
        raise RuntimeError("Reranker returned an unexpected number of scores.")

    scored: list[Document] = []
    for document, raw_score in zip(documents, scores, strict=True):
        score = float(raw_score)
        if not math.isfinite(score):
            raise RuntimeError("Reranker returned a non-finite score.")
        scored.append(Document(
            page_content=document.page_content,
            metadata={**document.metadata, "rerank_score": score},
        ))

    # Stable sorting preserves first-stage order when scores tie.
    return sorted(scored, key=lambda doc: doc.metadata["rerank_score"], reverse=True)[:k]
