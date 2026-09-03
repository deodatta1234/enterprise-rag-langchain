from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievalMetrics:
    hit_at_1: int
    hit_at_3: int
    hit_at_5: int
    reciprocal_rank: float
    page_hit: int


def document_id(document: Document) -> str:
    return str(
        document.metadata.get("document_id", "")
    )


def page_number(document: Document) -> str:
    value = document.metadata.get("page_number", "")

    if value is None:
        return ""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    return str(value)


def calculate_retrieval_metrics(
    retrieved_documents: list[Document],
    expected_document_ids: set[str],
    expected_pages: set[str],
) -> RetrievalMetrics:
    """
    Calculate retrieval quality for a single evaluation example.

    Metrics:
    - Hit@1
    - Hit@3
    - Hit@5
    - MRR / reciprocal rank
    - expected page hit
    """

    retrieved_ids = [
        document_id(document)
        for document in retrieved_documents
    ]

    def hit_at(k: int) -> int:
        top_k = retrieved_ids[:k]

        return int(
            any(
                retrieved_id in expected_document_ids
                for retrieved_id in top_k
            )
        )

    reciprocal_rank = 0.0

    for rank, retrieved_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if retrieved_id in expected_document_ids:
            reciprocal_rank = 1.0 / rank
            break

    page_hit = 0

    for document in retrieved_documents:
        doc_id = document_id(document)
        page = page_number(document)

        if (
            doc_id in expected_document_ids
            and (
                not expected_pages
                or page in expected_pages
            )
        ):
            page_hit = 1
            break

    return RetrievalMetrics(
        hit_at_1=hit_at(1),
        hit_at_3=hit_at(3),
        hit_at_5=hit_at(5),
        reciprocal_rank=reciprocal_rank,
        page_hit=page_hit,
    )