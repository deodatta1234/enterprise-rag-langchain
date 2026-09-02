from __future__ import annotations

import argparse

from dotenv import load_dotenv

from rag_chatbot.chat_pipeline import answer_question
from rag_chatbot.config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the local Chain RAG chatbot a question.")
    parser.add_argument("question", help="Question to ask.")
    parser.add_argument(
        "--groups",
        required=True,
        help="Comma-separated user groups. Development only; Entra roles replace this in production.",
    )
    args = parser.parse_args()

    load_dotenv()
    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    result = answer_question(args.question, groups, load_settings())

    print("\nAnswer:\n")
    print(result.answer)
    if result.citations:
        print("\nRetrieved context candidates:")
        for citation in result.citations:
            print(f"- {citation.document_id}, page {citation.page_number} ({citation.source_file})")


if __name__ == "__main__":
    main()
