import json
from hashlib import sha256
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum without loading the whole PDF into memory."""
    digest = sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def load_checksum_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def build_checksum_manifest(pdf_files: list[Path], pdf_directory: Path) -> dict[str, str]:
    """Create a stable {relative_pdf_path: checksum} mapping."""
    return {
        str(pdf_path.relative_to(pdf_directory)): sha256_file(pdf_path)
        for pdf_path in sorted(pdf_files)
    }


def save_checksum_manifest(manifest: dict[str, str], output_path: Path) -> None:
    """Save one checksum per PDF as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )