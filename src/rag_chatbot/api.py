from __future__ import annotations

import base64
import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .chat_pipeline import answer_question
from .config import load_settings


load_dotenv()

app = FastAPI(
    title="Enterprise RAG API",
    version="0.1.0",
)

settings = load_settings()


ROLE_TO_RAG_GROUPS = {
    "Rag.Employee": "All-Employees",
    "Rag.Manager": "Managers",
    "Rag.HR": "HR",
    "Rag.ITAdmin": "IT-Admins",
    "Rag.Security": "Security",
    "Rag.Engineering": "Engineering",
    "Rag.Finance": "Finance",
    "Rag.Legal": "Legal",
}


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2_000,
        examples=["How long are security audit logs retained?"],
    )


class CitationResponse(BaseModel):
    document_id: str
    page_number: int
    source_file: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]


def extract_entra_roles(request: Request) -> list[str]:
    """
    Extract Entra app roles from the X-MS-CLIENT-PRINCIPAL header
    injected by Azure Container Apps Easy Auth.
    """

    encoded_principal = request.headers.get(
        "x-ms-client-principal"
    )

    if not encoded_principal:
        raise HTTPException(
            status_code=401,
            detail="Authenticated Entra identity is missing.",
        )

    try:
        decoded_principal = base64.b64decode(
            encoded_principal
        )

        principal = json.loads(
            decoded_principal.decode("utf-8")
        )

    except (
        ValueError,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as error:
        raise HTTPException(
            status_code=401,
            detail="Invalid Entra identity information.",
        ) from error

    roles = [
        claim["val"]
        for claim in principal.get("claims", [])
        if claim.get("typ") == "roles"
        and claim.get("val")
    ]

    return roles


def get_rag_groups(request: Request) -> list[str]:
    """
    Convert Entra application roles into internal
    RAG access groups used by Weaviate.
    """

    roles = extract_entra_roles(request)

    rag_groups = {
        ROLE_TO_RAG_GROUPS[role]
        for role in roles
        if role in ROLE_TO_RAG_GROUPS
    }

    if not rag_groups:
        raise HTTPException(
            status_code=403,
            detail="User has no authorized RAG access role.",
        )

    return sorted(rag_groups)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Enterprise RAG API is running.",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:

    user_groups = get_rag_groups(request)

    try:
        result = answer_question(
            question=payload.question,
            user_groups=user_groups,
            settings=settings,
        )

        return ChatResponse(
            answer=result.answer,
            citations=[
                CitationResponse(
                    document_id=citation.document_id,
                    page_number=citation.page_number,
                    source_file=citation.source_file,
                )
                for citation in result.citations
            ],
        )

    except ConnectionError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error