"""Trusted document identifiers and their access-control groups.

Document IDs are derived only from a PDF filename prefix, for example
``SEC-001_security_policy.pdf``. Keep this mapping application-controlled: it
is used as retrieval metadata and must not be supplied by an LLM or end user.
"""

from __future__ import annotations

import re
from pathlib import Path


DOCUMENT_ACCESS_GROUPS: dict[str, list[str]] = {
    "HR-001": ["All-Employees"],
    "HR-002": ["All-Employees", "Managers"],
    "HR-003": ["All-Employees", "Managers", "HR"],
    "IT-001": ["IT-Admins", "HR", "Managers"],
    "IT-002": ["IT-Admins", "Security"],
    "IT-003": ["Data-Analysts", "Finance", "Engineering", "Security"],
    "SEC-001": ["All-Employees", "Contractors"],
    "SEC-002": ["IT-Admins", "Security", "Managers"],
    "SEC-003": ["Security", "IT-Admins", "Executives"],
    "SEC-004": ["All-Employees", "Data-Stewards", "Legal"],
    "ENG-001": ["Engineering", "Product", "Security"],
    "ENG-002": ["Engineering", "IT-Admins", "Operations"],
    "ENG-003": ["Engineering", "Product"],
    "ENG-004": ["Engineering", "Product", "IT-Admins"],
    "FIN-001": ["All-Employees", "Finance", "Managers"],
    "FIN-002": ["Finance", "Executives", "Procurement"],
    "FIN-003": ["Finance", "Executives", "Audit"],
    "PROC-001": ["Procurement", "Security", "Legal"],
    "PROC-002": ["All-Employees", "Procurement", "Finance"],
    "PROC-003": ["Procurement", "Legal", "Suppliers"],
    "LEGAL-001": ["All-Employees", "Legal", "Records-Managers"],
    "LEGAL-002": ["Legal", "Procurement", "Executives"],
    "LEGAL-003": ["All-Employees", "Sales", "Procurement"],
    "LEGAL-004": ["Legal", "Support", "Privacy-Team"],
    "LEGAL-005": ["Legal", "Sales", "Security"],
    "OPS-001": ["Operations", "IT-Admins", "Executives"],
    "OPS-002": ["All-Employees", "Facilities", "Safety-Officers"],
    "OPS-003": ["Support", "Engineering", "Operations"],
    "GEN-001": ["All-Employees", "Contractors"],
    "GEN-002": ["All-Employees"],
}

_DOCUMENT_ID_PATTERN = re.compile(r"^(?P<document_id>[A-Z]+-\d+)(?:_|$)")


def document_id_from_path(pdf_path: Path) -> str:
    """Return the document-ID prefix from a PDF filename.

    Raises:
        ValueError: If the file is not named like ``SEC-001_policy.pdf``.
    """
    match = _DOCUMENT_ID_PATTERN.match(pdf_path.stem)
    if not match:
        raise ValueError(
            "PDF filename must start with a document ID, for example "
            f"'SEC-001_security_policy.pdf': {pdf_path.name}"
        )
    return match.group("document_id")


def access_groups_for(document_id: str) -> list[str]:
    """Return a copy of the permitted groups for a registered document."""
    try:
        return list(DOCUMENT_ACCESS_GROUPS[document_id])
    except KeyError as error:
        raise ValueError(
            f"No access-group mapping is configured for document ID '{document_id}'."
        ) from error
