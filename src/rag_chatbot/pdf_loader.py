from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from rag_chatbot.checksums import sha256_file
from rag_chatbot.registry import access_groups_for, document_id_from_path


def find_pdfs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path}")
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*.pdf") if item.is_file())
    raise ValueError(f"Path does not exist: {path}")


def load_pdf_pages(
    pdf_paths: Iterable[Path], checksums: Mapping[Path, str] | None = None
) -> list[Document]:
    """Extract one LangChain document per non-empty PDF page."""
    pages: list[Document] = []
    for pdf_path in pdf_paths:
        document_id = document_id_from_path(pdf_path)
        access_groups = access_groups_for(document_id)
        document_checksum = checksums.get(pdf_path) if checksums else sha256_file(pdf_path)
        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text", sort=True).strip()
                if text:
                    pages.append(Document(
                        page_content=text,
                        metadata={
                            "document_id": document_id,
                            "document_checksum": checksums[pdf_path],
                            "source_file": pdf_path.name,
                            "source_path": str(pdf_path.resolve()),
                            "page_number": page_number,
                            "access_groups": access_groups,
                        },
                    ))
    return pages
