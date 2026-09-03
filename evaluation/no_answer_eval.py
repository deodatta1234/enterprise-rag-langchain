from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag_chatbot.chat_pipeline import (
    NO_ANSWER,
    answer_question,
)
from rag_chatbot.config import load_settings


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation_dataset.json"

MIN_NO_ANSWER_ACCURACY = 0.95


@dataclass
class NoAnswerResult:
    example_id: str
    question: str

    answer: str
    citation_count: int

    correct_answer: bool
    citations_empty: bool

    passed: bool


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


def is_no_answer_test(
    example: dict,
) -> bool:
    """
    Select corpus no-answer tests only.

    RBAC unauthorized tests are deliberately
    evaluated separately.
    """

    test_type = str(
        example.get(
            "test_type",
            "",
        )
    ).strip()

    return (
        test_type == "no_answer"
    )


def evaluate_example(
    example: dict,
    settings,
) -> NoAnswerResult:

    question = str(
        example["question"]
    ).strip()

    user_groups = parse_semicolon_field(
        example.get(
            "allowed_user_roles_groups"
        )
    )

    response = answer_question(
        question=question,
        user_groups=user_groups,
        settings=settings,
        k=5,
    )

    answer = response.answer.strip()

    correct_answer = (
        answer == NO_ANSWER
    )

    citations_empty = (
        len(response.citations) == 0
    )

    passed = (
        correct_answer
        and citations_empty
    )

    return NoAnswerResult(
        example_id=str(
            example.get(
                "id",
                "unknown",
            )
        ),
        question=question,
        answer=answer,
        citation_count=len(
            response.citations
        ),
        correct_answer=correct_answer,
        citations_empty=citations_empty,
        passed=passed,
    )


def print_result(
    result: NoAnswerResult,
) -> None:

    print("=" * 80)

    print(
        f"{result.example_id}: "
        f"{result.question}"
    )

    print(
        f"Answer: {result.answer}"
    )

    print(
        f"Citations: "
        f"{result.citation_count}"
    )

    print(
        "Expected fallback:",
        NO_ANSWER,
    )

    print(
        "Result:",
        "PASS"
        if result.passed
        else "FAIL",
    )


def main() -> None:
    load_dotenv()

    settings = load_settings()
    examples = load_dataset()

    no_answer_examples = [
        example
        for example in examples
        if is_no_answer_test(
            example
        )
    ]

    print(
        f"Loaded {len(examples)} "
        f"total examples."
    )

    print(
        f"No-answer examples: "
        f"{len(no_answer_examples)}"
    )

    if not no_answer_examples:
        raise RuntimeError(
            "No no_answer examples found."
        )

    results: list[
        NoAnswerResult
    ] = []

    errors: list[str] = []

    for example in no_answer_examples:

        try:
            result = evaluate_example(
                example,
                settings,
            )

            results.append(
                result
            )

            print_result(
                result
            )

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

    if not results:
        raise SystemExit(1)

    passed_count = sum(
        result.passed
        for result in results
    )

    accuracy = (
        passed_count
        / len(results)
    )

    print()
    print("#" * 80)
    print("NO-ANSWER EVALUATION SUMMARY")
    print("#" * 80)

    print(
        f"Examples evaluated : "
        f"{len(results)}"
    )

    print(
        f"Passed             : "
        f"{passed_count}"
    )

    print(
        f"No-answer accuracy : "
        f"{accuracy:.2%}"
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
        accuracy
        < MIN_NO_ANSWER_ACCURACY
        or errors
    ):
        print("FAILED")

        if (
            accuracy
            < MIN_NO_ANSWER_ACCURACY
        ):
            print(
                f"- No-answer accuracy "
                f"{accuracy:.2%} is below "
                f"{MIN_NO_ANSWER_ACCURACY:.2%}."
            )

        if errors:
            print(
                f"- {len(errors)} example(s) "
                f"failed to execute."
            )

        raise SystemExit(1)

    print("PASSED")

    print(
        f"No-answer accuracy >= "
        f"{MIN_NO_ANSWER_ACCURACY:.0%} ✓"
    )


if __name__ == "__main__":
    main()