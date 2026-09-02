from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chat_pipeline import answer_question
from .config import load_settings
from fastapi import Request


load_dotenv()

app = FastAPI(
    title="Enterprise RAG API",
    version="0.1.0",
)

settings = load_settings()


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


def development_user_groups() -> list[str]:
    """
    Temporary local-development authorization.
    Replace this function with validated Entra roles in the next step.
    """
    environment = os.getenv("APP_ENV", "development")

    if environment != "development":
        raise HTTPException(
            status_code=503,
            detail="Microsoft Entra authentication is not configured yet.",
        )

    return [
        group.strip()
        for group in os.getenv(
            "RAG_DEV_GROUPS",
            "All-Employees",
        ).split(",")
        if group.strip()
    ]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Enterprise RAG API is running.",
        "docs": "/docs",
    }

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = answer_question(
            question=payload.question,
            user_groups=development_user_groups(),
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
        raise HTTPException(status_code=503, detail=str(error)) from error

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

@app.get("/debug-auth")
def debug_auth(request: Request):
    return {
        "principal": request.headers.get("x-ms-client-principal"),
        "principal_id": request.headers.get("x-ms-client-principal-id"),
        "principal_name": request.headers.get("x-ms-client-principal-name"),
    }