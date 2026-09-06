from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag_chatbot.chat_pipeline import (
    retrieve_documents_with_resources,
)
from rag_chatbot.config import load_settings
from rag_chatbot.indexing import (
    connect_weaviate,
    create_embeddings,
)

from .retrieval_metrics import (
    RetrievalMetrics,
    calculate_retrieval_metrics,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "evaluation_dataset.json"
)

RETRIEVAL_K = 5


# ---------------------------------------------------------------------------
# CI/CD quality gates
# ---------------------------------------------------------------------------

# Current baseline:
#
# Hit@1         = 93.88%
# Hit@3         = 100.00%
# Hit@5         = 100.00%
# MRR           = 0.9660
# Page Hit Rate = 100.00%
#
# Thresholds are intentionally slightly below baseline.

MIN_HIT_AT_3 = 0.97
MIN_HIT_AT_5 = 0.97
MIN_MRR = 0.93
MIN_PAGE_HIT_RATE = 0.98


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    example_id: str
    question: str
    test_type: str

    expected_documents: list[str]
    retrieved_documents: list[str]

    hit_at_1: int
    hit_at_3: int
    hit_at_5: int

    reciprocal_rank: float
    page_hit: int

    latency_ms: float


@dataclass
class AggregateMetrics:
    example_count: int

    hit_at_1: float
    hit_at_3: float
    hit_at_5: float

    mrr: float
    page_hit_rate: float

    average_latency_ms: float


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def parse_semicolon_field(
    value: Any,
) -> list[str]:
    """
    Convert dataset values into a list.

    Example:

        "All-Employees;Managers;HR"

    becomes:

        [
            "All-Employees",
            "Managers",
            "HR",
        ]
    """

    if value is None:
        return []

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    value = str(value).strip()

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(";")
        if item.strip()
    ]


def load_dataset() -> list[dict]:
    """
    Load examples from evaluation_dataset.json.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    examples = data.get(
        "examples"
    )

    if not isinstance(
        examples,
        list,
    ):
        raise ValueError(
            "evaluation_dataset.json must "
            "contain an 'examples' list."
        )

    return examples


def should_run_retrieval_eval(
    example: dict,
) -> bool:
    """
    Run positive retrieval tests only.

    We currently skip:
    - no-answer examples
    - RBAC unauthorized examples

    Those will get separate evaluators.
    """

    if not example.get(
        "answerable",
        False,
    ):
        return False

    test_type = str(
        example.get(
            "test_type",
            "",
        )
    ).strip()

    if test_type == "rbac_unauthorized":
        return False

    expected_documents = (
        parse_semicolon_field(
            example.get(
                "expected_source_documents"
            )
        )
    )

    return bool(
        expected_documents
    )


# ---------------------------------------------------------------------------
# Individual retrieval evaluation
# ---------------------------------------------------------------------------

def evaluate_example(
    example: dict,
    settings,
    client,
    embeddings,
    *,
    k: int = RETRIEVAL_K,
) -> EvaluationResult:
    """
    Run retrieval for one example.

    The Weaviate client and embedding client are
    passed in rather than recreated for every query.
    """

    question = str(
        example["question"]
    ).strip()

    user_groups = (
        parse_semicolon_field(
            example.get(
                "allowed_user_roles_groups"
            )
        )
    )

    expected_documents = set(
        parse_semicolon_field(
            example.get(
                "expected_source_documents"
            )
        )
    )

    expected_pages = set(
        parse_semicolon_field(
            example.get(
                "expected_page"
            )
        )
    )

    start = time.perf_counter()

    documents = (
        retrieve_documents_with_resources(
            question=question,
            user_groups=user_groups,
            client=client,
            embeddings=embeddings,
            collection_name=(
                settings.collection_name
            ),
            k=k,
            settings=settings,
        )
    )

    latency_ms = (
        time.perf_counter()
        - start
    ) * 1000

    metrics: RetrievalMetrics = (
        calculate_retrieval_metrics(
            retrieved_documents=documents,
            expected_document_ids=(
                expected_documents
            ),
            expected_pages=(
                expected_pages
            ),
        )
    )

    retrieved_document_ids = [
        str(
            document.metadata.get(
                "document_id",
                "",
            )
        )
        for document in documents
    ]

    return EvaluationResult(
        example_id=str(
            example.get(
                "id",
                "unknown",
            )
        ),
        question=question,
        test_type=str(
            example.get(
                "test_type",
                "",
            )
        ),
        expected_documents=sorted(
            expected_documents
        ),
        retrieved_documents=(
            retrieved_document_ids
        ),
        hit_at_1=(
            metrics.hit_at_1
        ),
        hit_at_3=(
            metrics.hit_at_3
        ),
        hit_at_5=(
            metrics.hit_at_5
        ),
        reciprocal_rank=(
            metrics.reciprocal_rank
        ),
        page_hit=(
            metrics.page_hit
        ),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_example_result(
    result: EvaluationResult,
) -> None:
    print("=" * 80)

    print(
        f"{result.example_id}: "
        f"{result.question}"
    )

    print(
        "Expected:",
        result.expected_documents,
    )

    print(
        "Retrieved:",
        result.retrieved_documents,
    )

    print(
        f"Hit@1={result.hit_at_1} | "
        f"Hit@3={result.hit_at_3} | "
        f"Hit@5={result.hit_at_5} | "
        f"MRR="
        f"{result.reciprocal_rank:.3f} | "
        f"PageHit={result.page_hit} | "
        f"Latency="
        f"{result.latency_ms:.0f} ms"
    )


def calculate_aggregate_results(
    results: list[EvaluationResult],
) -> AggregateMetrics:
    """
    Calculate aggregate retrieval metrics.
    """

    if not results:
        raise ValueError(
            "No retrieval evaluation "
            "results were produced."
        )

    count = len(
        results
    )

    hit_at_1 = (
        sum(
            result.hit_at_1
            for result in results
        )
        / count
    )

    hit_at_3 = (
        sum(
            result.hit_at_3
            for result in results
        )
        / count
    )

    hit_at_5 = (
        sum(
            result.hit_at_5
            for result in results
        )
        / count
    )

    mrr = (
        sum(
            result.reciprocal_rank
            for result in results
        )
        / count
    )

    page_hit_rate = (
        sum(
            result.page_hit
            for result in results
        )
        / count
    )

    average_latency_ms = (
        sum(
            result.latency_ms
            for result in results
        )
        / count
    )

    return AggregateMetrics(
        example_count=count,
        hit_at_1=hit_at_1,
        hit_at_3=hit_at_3,
        hit_at_5=hit_at_5,
        mrr=mrr,
        page_hit_rate=(
            page_hit_rate
        ),
        average_latency_ms=(
            average_latency_ms
        ),
    )


def print_aggregate_results(
    metrics: AggregateMetrics,
) -> None:
    print()
    print("#" * 80)
    print(
        "RETRIEVAL EVALUATION SUMMARY"
    )
    print("#" * 80)

    print(
        f"Examples evaluated : "
        f"{metrics.example_count}"
    )

    print(
        f"Hit@1              : "
        f"{metrics.hit_at_1:.2%}"
    )

    print(
        f"Hit@3              : "
        f"{metrics.hit_at_3:.2%}"
    )

    print(
        f"Hit@5              : "
        f"{metrics.hit_at_5:.2%}"
    )

    print(
        f"MRR                : "
        f"{metrics.mrr:.4f}"
    )

    print(
        f"Page Hit Rate      : "
        f"{metrics.page_hit_rate:.2%}"
    )

    print(
        f"Avg retrieval time : "
        f"{metrics.average_latency_ms:.0f} ms"
    )


# ---------------------------------------------------------------------------
# CI/CD quality gate
# ---------------------------------------------------------------------------

def check_quality_gate(
    metrics: AggregateMetrics,
) -> list[str]:
    """
    Return quality-gate failures.

    Empty list = PASS.
    """

    failures: list[str] = []

    if (
        metrics.hit_at_3
        < MIN_HIT_AT_3
    ):
        failures.append(
            f"Hit@3 = "
            f"{metrics.hit_at_3:.2%}, "
            f"required >= "
            f"{MIN_HIT_AT_3:.2%}"
        )

    if (
        metrics.hit_at_5
        < MIN_HIT_AT_5
    ):
        failures.append(
            f"Hit@5 = "
            f"{metrics.hit_at_5:.2%}, "
            f"required >= "
            f"{MIN_HIT_AT_5:.2%}"
        )

    if metrics.mrr < MIN_MRR:
        failures.append(
            f"MRR = "
            f"{metrics.mrr:.4f}, "
            f"required >= "
            f"{MIN_MRR:.4f}"
        )

    if (
        metrics.page_hit_rate
        < MIN_PAGE_HIT_RATE
    ):
        failures.append(
            f"Page Hit Rate = "
            f"{metrics.page_hit_rate:.2%}, "
            f"required >= "
            f"{MIN_PAGE_HIT_RATE:.2%}"
        )

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()

    settings = load_settings()

    examples = load_dataset()

    results: list[
        EvaluationResult
    ] = []

    evaluation_errors: list[
        str
    ] = []

    print(
        f"Loaded {len(examples)} "
        f"evaluation examples."
    )

    print(
        f"Retrieval k = "
        f"{RETRIEVAL_K}"
    )

    # -----------------------------------------------------------------------
    # Create shared resources ONCE
    # -----------------------------------------------------------------------

    print(
        "Connecting to Weaviate..."
    )

    client = connect_weaviate(
        settings
    )

    try:
        if not client.is_ready():
            raise ConnectionError(
                "Weaviate is not ready."
            )

        if not client.collections.exists(
            settings.collection_name
        ):
            raise RuntimeError(
                f"Weaviate collection "
                f"'{settings.collection_name}' "
                f"does not exist."
            )

        print(
            "Weaviate connection ready."
        )

        print(
            "Creating embeddings client..."
        )

        embeddings = create_embeddings(
            settings
        )

        print(
            "Starting retrieval evaluation..."
        )

        # -------------------------------------------------------------------
        # Run all evaluation examples using the same resources
        # -------------------------------------------------------------------

        for example in examples:

            if not should_run_retrieval_eval(
                example
            ):
                continue

            try:
                result = (
                    evaluate_example(
                        example=example,
                        settings=settings,
                        client=client,
                        embeddings=embeddings,
                        k=RETRIEVAL_K,
                    )
                )

                results.append(
                    result
                )

                print_example_result(
                    result
                )

            except Exception as error:
                example_id = str(
                    example.get(
                        "id",
                        "unknown",
                    )
                )

                question = str(
                    example.get(
                        "question",
                        "",
                    )
                )

                error_message = (
                    f"{example_id}: "
                    f"{error}"
                )

                evaluation_errors.append(
                    error_message
                )

                print("=" * 80)

                print(
                    f"{example_id} FAILED"
                )

                print(
                    f"Question: "
                    f"{question}"
                )

                print(
                    f"Error: "
                    f"{error}"
                )

    finally:
        print(
            "Closing Weaviate connection."
        )

        client.close()

    # -----------------------------------------------------------------------
    # No successful evaluations
    # -----------------------------------------------------------------------

    if not results:
        print()

        print(
            "ERROR: No retrieval "
            "examples were evaluated."
        )

        raise SystemExit(
            1
        )

    # -----------------------------------------------------------------------
    # Aggregate results
    # -----------------------------------------------------------------------

    metrics = (
        calculate_aggregate_results(
            results
        )
    )

    print_aggregate_results(
        metrics
    )

    # -----------------------------------------------------------------------
    # Evaluation execution errors
    # -----------------------------------------------------------------------

    if evaluation_errors:
        print()
        print("#" * 80)
        print(
            "EVALUATION ERRORS"
        )
        print("#" * 80)

        for error in (
            evaluation_errors
        ):
            print(
                f"- {error}"
            )

    # -----------------------------------------------------------------------
    # Quality gate
    # -----------------------------------------------------------------------

    quality_failures = (
        check_quality_gate(
            metrics
        )
    )

    if evaluation_errors:
        quality_failures.append(
            f"{len(evaluation_errors)} "
            f"evaluation example(s) "
            f"failed to execute."
        )

    print()
    print("#" * 80)
    print(
        "QUALITY GATE"
    )
    print("#" * 80)

    if quality_failures:
        print(
            "FAILED"
        )

        for failure in (
            quality_failures
        ):
            print(
                f"- {failure}"
            )

        # Exit code 1 makes the
        # GitHub Actions step fail.
        raise SystemExit(
            1
        )

    print(
        "PASSED"
    )

    print(
        f"Hit@3 >= "
        f"{MIN_HIT_AT_3:.0%} ✓"
    )

    print(
        f"Hit@5 >= "
        f"{MIN_HIT_AT_5:.0%} ✓"
    )

    print(
        f"MRR >= "
        f"{MIN_MRR:.2f} ✓"
    )

    print(
        f"Page Hit Rate >= "
        f"{MIN_PAGE_HIT_RATE:.0%} ✓"
    )


if __name__ == "__main__":
    main()
