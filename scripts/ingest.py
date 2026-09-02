from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow this source-layout project to run as ``python scripts/ingest.py``
# without first installing it as an editable package.
SRC_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SRC_DIRECTORY))

from rag_chatbot.config import load_settings
from rag_chatbot.ingest_pipeline import ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF policies into local Weaviate.")
    parser.add_argument("path", nargs="?", type=Path, help="PDF file or directory (defaults to RAG_PDF_DIRECTORY).")
    parser.add_argument("--rebuild", action="store_true", help="Delete and recreate the collection before indexing.")
    args = parser.parse_args()

    load_dotenv()
    settings = load_settings()
    pdf_count, chunk_count = ingest(args.path or settings.pdf_directory, settings, rebuild=args.rebuild)
    print(f"Indexed {chunk_count} chunks from {pdf_count} PDF(s) into '{settings.collection_name}'.")


if __name__ == "__main__":
    main()
