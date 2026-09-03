from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag_chatbot.chat_pipeline import (
    build_rag_chain,
    retrieve_documents_with_resources,
)
from rag_chatbot.config import load_settings
from rag_chatbot.indexing import (
    connect_weaviate,
    create_embeddings,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation_dataset.json"

RETRIEVAL_K = 5

# Don't make these CI gates yet.
PERFORMANCE_SAMPLE_SIZE = 20


@dataclass
class PerformanceResult:
    example_id: str
    retrieval_ms: float
    total_rag_ms: float


def parse_semicolon_field(
    value: Any,
) -> list[str]:

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]


def load_dataset() -> list[dict]:

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return [
        example
        for example in data["examples"]
        if example.get(
            "answerable",
            False,
        )
    ]


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    sorted_values = sorted(
        values
    )

    index = int(
        round(
            percentile_value
            * (len(sorted_values) - 1)
        )
    )

    return sorted_values[
        index
    ]


def main() -> None:
    load_dotenv()

    settings = load_settings()

    examples = load_dataset()[
        :PERFORMANCE_SAMPLE_SIZE
    ]

    client = connect_weaviate(
        settings
    )

    embeddings = create_embeddings(
        settings
    )

    results: list[
        PerformanceResult
    ] = []

    try:
        for example in examples:

            question = str(
                example["question"]
            )

            user_groups = (
                parse_semicolon_field(
                    example.get(
                        "allowed_user_roles_groups"
                    )
                )
            )

            # -----------------------------------------------
            # Retrieval timing
            # -----------------------------------------------

            retrieval_start = (
                time.perf_counter()
            )

            retrieve_documents_with_resources(
                question=question,
                user_groups=user_groups,
                client=client,
                embeddings=embeddings,
                collection_name=(
                    settings.collection_name
                ),
                k=RETRIEVAL_K,
            )

            retrieval_ms = (
                time.perf_counter()
                - retrieval_start
            ) * 1000

            # -----------------------------------------------
            # Complete RAG request timing
            # -----------------------------------------------

            chain = build_rag_chain(
                client=client,
                settings=settings,
                user_groups=user_groups,
                k=RETRIEVAL_K,
            )

            total_start = (
                time.perf_counter()
            )

            chain.invoke(
                question
            )

            total_rag_ms = (
                time.perf_counter()
                - total_start
            ) * 1000

            result = PerformanceResult(
                example_id=str(
                    example.get(
                        "id",
                        "unknown",
                    )
                ),
                retrieval_ms=(
                    retrieval_ms
                ),
                total_rag_ms=(
                    total_rag_ms
                ),
            )

            results.append(
                result
            )

            print(
                f"{result.example_id}: "
                f"retrieval="
                f"{retrieval_ms:.0f} ms | "
                f"total="
                f"{total_rag_ms:.0f} ms"
            )

    finally:
        client.close()

    retrieval_times = [
        result.retrieval_ms
        for result in results
    ]

    total_times = [
        result.total_rag_ms
        for result in results
    ]

    print()
    print("#" * 80)
    print("PERFORMANCE SUMMARY")
    print("#" * 80)

    print(
        f"Examples: "
        f"{len(results)}"
    )

    print()
    print("Retrieval")

    print(
        f"Average : "
        f"{statistics.mean(retrieval_times):.0f} ms"
    )

    print(
        f"P50     : "
        f"{statistics.median(retrieval_times):.0f} ms"
    )

    print(
        f"P95     : "
        f"{percentile(retrieval_times, 0.95):.0f} ms"
    )

    print()
    print("Total RAG")

    print(
        f"Average : "
        f"{statistics.mean(total_times):.0f} ms"
    )

    print(
        f"P50     : "
        f"{statistics.median(total_times):.0f} ms"
    )

    print(
        f"P95     : "
        f"{percentile(total_times, 0.95):.0f} ms"
    )


if __name__ == "__main__":
    main()