from __future__ import annotations

import json
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation_dataset.json"

RETRIEVAL_K = 5

# Security should be a hard gate.
MIN_RBAC_PROTECTION_RATE = 1.0


@dataclass
class RBACResult:
    example_id: str
    question: str

    user_groups: list[str]
    restricted_documents: list[str]
    retrieved_documents: list[str]

    protected: bool


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

    value = str(value).strip()

    if not value:
        return []

    return [
        item.strip()
        for item in value.split(";")
        if item.strip()
    ]


def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    examples = data.get("examples")

    if not isinstance(examples, list):
        raise ValueError(
            "evaluation_dataset.json must contain "
            "an 'examples' list."
        )

    return examples


def is_rbac_test(
    example: dict,
) -> bool:
    return (
        str(
            example.get(
                "test_type",
                "",
            )
        ).strip()
        == "rbac_unauthorized"
    )


def evaluate_rbac_example(
    example: dict,
    *,
    client,
    embeddings,
    collection_name: str,
    k: int = RETRIEVAL_K,
) -> RBACResult:

    question = str(
        example["question"]
    ).strip()

    user_groups = parse_semicolon_field(
        example.get(
            "allowed_user_roles_groups"
        )
    )

    restricted_documents = set(
        parse_semicolon_field(
            example.get(
                "expected_source_documents"
            )
        )
    )

    documents = retrieve_documents_with_resources(
        question=question,
        user_groups=user_groups,
        client=client,
        embeddings=embeddings,
        collection_name=collection_name,
        k=k,
    )

    retrieved_documents = [
        str(
            document.metadata.get(
                "document_id",
                "",
            )
        )
        for document in documents
    ]

    leaked_documents = (
        restricted_documents
        & set(retrieved_documents)
    )

    protected = not leaked_documents

    return RBACResult(
        example_id=str(
            example.get(
                "id",
                "unknown",
            )
        ),
        question=question,
        user_groups=user_groups,
        restricted_documents=sorted(
            restricted_documents
        ),
        retrieved_documents=(
            retrieved_documents
        ),
        protected=protected,
    )


def print_result(
    result: RBACResult,
) -> None:

    print("=" * 80)

    print(
        f"{result.example_id}: "
        f"{result.question}"
    )

    print(
        "User groups:",
        result.user_groups,
    )

    print(
        "Restricted documents:",
        result.restricted_documents,
    )

    print(
        "Retrieved:",
        result.retrieved_documents,
    )

    print(
        "RBAC:",
        "PASS"
        if result.protected
        else "FAIL",
    )


def main() -> None:
    load_dotenv()

    settings = load_settings()
    examples = load_dataset()

    rbac_examples = [
        example
        for example in examples
        if is_rbac_test(example)
    ]

    print(
        f"Loaded {len(examples)} total examples."
    )

    print(
        f"RBAC examples: {len(rbac_examples)}"
    )

    if not rbac_examples:
        raise RuntimeError(
            "No rbac_unauthorized examples found."
        )

    client = connect_weaviate(
        settings
    )

    results: list[RBACResult] = []
    errors: list[str] = []

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

        embeddings = create_embeddings(
            settings
        )

        for example in rbac_examples:

            try:
                result = evaluate_rbac_example(
                    example,
                    client=client,
                    embeddings=embeddings,
                    collection_name=(
                        settings.collection_name
                    ),
                    k=RETRIEVAL_K,
                )

                results.append(result)

                print_result(result)

            except Exception as error:
                example_id = str(
                    example.get(
                        "id",
                        "unknown",
                    )
                )

                errors.append(
                    f"{example_id}: {error}"
                )

    finally:
        client.close()

    if not results:
        raise SystemExit(1)

    protected_count = sum(
        result.protected
        for result in results
    )

    protection_rate = (
        protected_count
        / len(results)
    )

    print()
    print("#" * 80)
    print("RBAC EVALUATION SUMMARY")
    print("#" * 80)

    print(
        f"Examples evaluated : "
        f"{len(results)}"
    )

    print(
        f"Protected          : "
        f"{protected_count}"
    )

    print(
        f"Protection rate    : "
        f"{protection_rate:.2%}"
    )

    if errors:
        print()
        print("Execution errors:")

        for error in errors:
            print(
                f"- {error}"
            )

    print()
    print("#" * 80)
    print("QUALITY GATE")
    print("#" * 80)

    if (
        protection_rate
        < MIN_RBAC_PROTECTION_RATE
        or errors
    ):
        print("FAILED")

        if protection_rate < 1.0:
            print(
                "- Unauthorized document "
                "retrieval detected."
            )

        if errors:
            print(
                f"- {len(errors)} evaluation "
                f"example(s) failed to execute."
            )

        raise SystemExit(1)

    print("PASSED")
    print("RBAC Protection = 100% ✓")


if __name__ == "__main__":
    main()