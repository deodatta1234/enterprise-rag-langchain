from __future__ import annotations

import os
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .checksums import (
    build_checksum_manifest,
    load_checksum_manifest,
    save_checksum_manifest,
)
from .config import Settings
from .indexing import (
    connect_weaviate,
    delete_document_chunks,
    rebuild_collection,
    store_chunks,
)
from .pdf_loader import find_pdfs, load_pdf_pages
from .registry import document_id_from_path


def ingest(path: Path, settings: Settings, *, rebuild: bool) -> tuple[int, int]:
    """Synchronize PDF files with Weaviate. Returns (PDF count, chunk count)."""

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is missing. Add it to .env.")

    if settings.chunk_overlap >= settings.chunk_size:
        raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE.")

    pdf_directory = path if path.is_dir() else path.parent
    pdfs = find_pdfs(path)

    if not pdfs:
        raise ValueError(f"No PDF files found in: {path}")

    manifest_path = pdf_directory.parent / "checksums.json"
    previous_manifest = load_checksum_manifest(manifest_path)
    current_manifest = build_checksum_manifest(pdfs, pdf_directory)

    def relative_key(pdf_path: Path) -> str:
        return str(pdf_path.relative_to(pdf_directory))

    if rebuild:
        pdfs_to_index = pdfs
        removed_files: set[str] = set()
    else:
        pdfs_to_index = [
            pdf_path
            for pdf_path in pdfs
            if previous_manifest.get(relative_key(pdf_path))
            != current_manifest[relative_key(pdf_path)]
        ]
        removed_files = set(previous_manifest) - set(current_manifest)

    client = connect_weaviate(settings)

    try:
        if not client.is_ready():
            raise ConnectionError("Local Weaviate is not ready.")

        collection_exists = client.collections.exists(settings.collection_name)

        if rebuild:
            rebuild_collection(client, settings.collection_name)

        elif collection_exists and not previous_manifest:
            raise ValueError(
                "The Weaviate collection exists but data/checksums.json is missing. "
                "Run once with --rebuild to establish the checksum manifest."
            )

        # Remove old chunks for PDFs that changed.
        for pdf_path in pdfs_to_index:
            if relative_key(pdf_path) in previous_manifest:
                document_id = document_id_from_path(pdf_path)
                delete_document_chunks(
                    client,
                    settings.collection_name,
                    document_id,
                )

        # Remove chunks for PDFs deleted from the local folder.
        for removed_relative_path in removed_files:
            document_id = document_id_from_path(Path(removed_relative_path))
            delete_document_chunks(
                client,
                settings.collection_name,
                document_id,
            )

        # Create and store chunks only for new or changed PDFs.
        checksums_for_indexing = {
            pdf_path: current_manifest[relative_key(pdf_path)]
            for pdf_path in pdfs_to_index
        }

        pages = load_pdf_pages(pdfs_to_index, checksums_for_indexing)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            add_start_index=True,
        )
        chunks = splitter.split_documents(pages)

        for position, chunk in enumerate(chunks, start=1):
            chunk.metadata["chunk_number"] = position

        if chunks:
            store_chunks(client, settings, chunks)

        # Save only after deletion/indexing has completed successfully.
        save_checksum_manifest(current_manifest, manifest_path)

    finally:
        client.close()

    skipped = len(pdfs) - len(pdfs_to_index)
    if skipped:
        print(f"Skipped {skipped} unchanged PDF(s).")

    return len(pdfs), len(chunks)