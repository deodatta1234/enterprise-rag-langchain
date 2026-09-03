from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from rag_chatbot.chat_pipeline import (
    NO_ANSWER,
    build_rag_chain,
)
from rag_chatbot.config import load_settings
from rag_chatbot.indexing import connect_weaviate


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "evaluation_dataset.json"

RETRIEVAL_K = 5

# Quality gates
MIN_CORRECTNESS = 0.90
MIN_FAITHFULNESS = 0.95
MIN_CITATION_DOCUMENT_ACCURACY = 0.95
MIN_CITATION_PAGE_ACCURACY = 0.95


@dataclass
class GenerationResult:
    example_id: str
    question: str

    expected_answer: str
    generated_answer: str

    correctness: float
    faithfulness: float

    citation_document_correct: int
    citation_page_correct: int


@dataclass
class GenerationSummary:
    example_count: int

    correctness: float
    faithfulness: float

    citation_document_accuracy: float
    citation_page_accuracy: float


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
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["examples"]


def should_run_generation_eval(
    example: dict,
) -> bool:
    """
    Only evaluate answerable generation examples.

    RBAC and no-answer are handled separately.
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
    )

    return test_type in {
        "retrieval_and_generation",
        "generation",
        "multi_hop_generation",
        "ambiguity",
    }


def format_context(
    documents,
) -> str:
    parts = []

    for document in documents:
        doc_id = str(
            document.metadata.get(
                "document_id",
                "",
            )
        )

        page = int(
            document.metadata.get(
                "page_number",
                0,
            )
        )

        parts.append(
            f"[{doc_id}, page {page}]\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(
        parts
    )


def judge_answer(
    *,
    judge: ChatOpenAI,
    question: str,
    expected_answer: str,
    generated_answer: str,
    context: str,
) -> tuple[float, float]:
    """
    Return:
        correctness
        faithfulness

    Both scores are normalized to 0.0 - 1.0.
    """

    prompt = f"""
You are evaluating an enterprise RAG system.

Evaluate the generated answer using ONLY the information
provided below.

Question:
{question}

Reference answer:
{expected_answer}

Retrieved context:
{context}

Generated answer:
{generated_answer}

Score the answer on two criteria.

CORRECTNESS:
Does the generated answer correctly answer the question
and agree with the reference answer?

FAITHFULNESS:
Are the factual claims in the generated answer supported
by the retrieved context?

Return ONLY valid JSON:

{{
  "correctness": <number from 0 to 1>,
  "faithfulness": <number from 0 to 1>
}}
"""

    response = judge.invoke(
        prompt
    )

    content = str(
        response.content
    ).strip()

    # Handle accidental Markdown JSON fences.
    if content.startswith("```"):
        content = (
            content
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )

    result = json.loads(
        content
    )

    correctness = float(
        result["correctness"]
    )

    faithfulness = float(
        result["faithfulness"]
    )

    return (
        correctness,
        faithfulness,
    )


def evaluate_example(
    *,
    example: dict,
    client,
    settings,
    judge: ChatOpenAI,
) -> GenerationResult:

    question = str(
        example["question"]
    ).strip()

    expected_answer = str(
        example.get(
            "expected_answer",
            "",
        )
    ).strip()

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

    user_groups = (
        parse_semicolon_field(
            example.get(
                "allowed_user_roles_groups"
            )
        )
    )

    chain = build_rag_chain(
        client=client,
        settings=settings,
        user_groups=user_groups,
        k=RETRIEVAL_K,
    )

    result = chain.invoke(
        question
    )

    generated_answer = str(
        result["answer"]
    ).strip()

    documents = result[
        "documents"
    ]

    context = format_context(
        documents
    )

    correctness, faithfulness = (
        judge_answer(
            judge=judge,
            question=question,
            expected_answer=(
                expected_answer
            ),
            generated_answer=(
                generated_answer
            ),
            context=context,
        )
    )

    retrieved_document_ids = {
        str(
            document.metadata.get(
                "document_id",
                "",
            )
        )
        for document in documents
    }

    retrieved_pages = {
        str(
            int(
                document.metadata.get(
                    "page_number",
                    0,
                )
            )
        )
        for document in documents
        if str(
            document.metadata.get(
                "document_id",
                "",
            )
        )
        in expected_documents
    }

    citation_document_correct = int(
        bool(
            expected_documents
            & retrieved_document_ids
        )
    )

    citation_page_correct = int(
        bool(
            expected_pages
            & retrieved_pages
        )
    )

    return GenerationResult(
        example_id=str(
            example.get(
                "id",
                "unknown",
            )
        ),
        question=question,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
        correctness=correctness,
        faithfulness=faithfulness,
        citation_document_correct=(
            citation_document_correct
        ),
        citation_page_correct=(
            citation_page_correct
        ),
    )


def main() -> None:
    load_dotenv()

    settings = load_settings()

    examples = [
        example
        for example in load_dataset()
        if should_run_generation_eval(
            example
        )
    ]

    print(
        f"Generation examples: "
        f"{len(examples)}"
    )

    client = connect_weaviate(
        settings
    )

    judge = ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
    )

    results: list[
        GenerationResult
    ] = []

    errors: list[str] = []

    try:
        for example in examples:
            try:
                result = evaluate_example(
                    example=example,
                    client=client,
                    settings=settings,
                    judge=judge,
                )

                results.append(
                    result
                )

                print("=" * 80)

                print(
                    f"{result.example_id}: "
                    f"{result.question}"
                )

                print(
                    f"Correctness: "
                    f"{result.correctness:.2f}"
                )

                print(
                    f"Faithfulness: "
                    f"{result.faithfulness:.2f}"
                )

                print(
                    f"Citation document: "
                    f"{result.citation_document_correct}"
                )

                print(
                    f"Citation page: "
                    f"{result.citation_page_correct}"
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

    finally:
        client.close()

    if not results:
        raise SystemExit(1)

    count = len(
        results
    )

    correctness = (
        sum(
            result.correctness
            for result in results
        )
        / count
    )

    faithfulness = (
        sum(
            result.faithfulness
            for result in results
        )
        / count
    )

    citation_document_accuracy = (
        sum(
            result.citation_document_correct
            for result in results
        )
        / count
    )

    citation_page_accuracy = (
        sum(
            result.citation_page_correct
            for result in results
        )
        / count
    )

    print()
    print("#" * 80)
    print("GENERATION EVALUATION SUMMARY")
    print("#" * 80)

    print(
        f"Examples evaluated          : "
        f"{count}"
    )

    print(
        f"Answer correctness          : "
        f"{correctness:.2%}"
    )

    print(
        f"Faithfulness                : "
        f"{faithfulness:.2%}"
    )

    print(
        f"Citation document accuracy  : "
        f"{citation_document_accuracy:.2%}"
    )

    print(
        f"Citation page accuracy      : "
        f"{citation_page_accuracy:.2%}"
    )

    failures = []

    if correctness < MIN_CORRECTNESS:
        failures.append(
            f"Correctness "
            f"{correctness:.2%} < "
            f"{MIN_CORRECTNESS:.2%}"
        )

    if faithfulness < MIN_FAITHFULNESS:
        failures.append(
            f"Faithfulness "
            f"{faithfulness:.2%} < "
            f"{MIN_FAITHFULNESS:.2%}"
        )

    if (
        citation_document_accuracy
        < MIN_CITATION_DOCUMENT_ACCURACY
    ):
        failures.append(
            "Citation document accuracy "
            f"{citation_document_accuracy:.2%} < "
            f"{MIN_CITATION_DOCUMENT_ACCURACY:.2%}"
        )

    if (
        citation_page_accuracy
        < MIN_CITATION_PAGE_ACCURACY
    ):
        failures.append(
            "Citation page accuracy "
            f"{citation_page_accuracy:.2%} < "
            f"{MIN_CITATION_PAGE_ACCURACY:.2%}"
        )

    if errors:
        failures.append(
            f"{len(errors)} evaluation "
            f"example(s) failed."
        )

    print()
    print("#" * 80)
    print("QUALITY GATE")
    print("#" * 80)

    if failures:
        print("FAILED")

        for failure in failures:
            print(
                f"- {failure}"
            )

        raise SystemExit(1)

    print("PASSED")


if __name__ == "__main__":
    main()