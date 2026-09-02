from __future__ import annotations

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore

from .config import Settings


def connect_weaviate(settings: Settings) -> weaviate.WeaviateClient:
    """Connect to Weaviate Cloud when URL/API key are configured, else local."""

    if settings.weaviate_url:
        if not settings.weaviate_api_key:
            raise EnvironmentError(
                "WEAVIATE_API_KEY is required when WEAVIATE_URL is set."
            )

        return weaviate.connect_to_weaviate_cloud(
            cluster_url=settings.weaviate_url,
            auth_credentials=Auth.api_key(settings.weaviate_api_key),
        )

    return weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )


def rebuild_collection(
    client: weaviate.WeaviateClient,
    collection_name: str,
) -> None:
    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)


def delete_document_chunks(
    client: weaviate.WeaviateClient,
    collection_name: str,
    document_id: str,
) -> None:
    """Delete all old chunks for one changed or deleted PDF."""

    if not client.collections.exists(collection_name):
        return

    collection = client.collections.use(collection_name)

    collection.data.delete_many(
        where=Filter.by_property("document_id").equal(document_id)
    )


def create_embeddings(settings: Settings) -> OpenAIEmbeddings:
    """Create the shared embedding client used by ingestion and retrieval."""

    return OpenAIEmbeddings(
        model=settings.embedding_model
    )


def get_vectorstore(
    client: weaviate.WeaviateClient,
    settings: Settings,
) -> WeaviateVectorStore:
    """
    Connect LangChain to the already existing Weaviate collection.
    Does not create a new collection.
    """

    embeddings = create_embeddings(settings)

    if not client.collections.exists(settings.collection_name):
        raise RuntimeError(
            f"Weaviate collection '{settings.collection_name}' does not exist. "
            "Run the ingestion pipeline first."
        )

    return WeaviateVectorStore(
        client=client,
        index_name=settings.collection_name,
        text_key="text",
        embedding=embeddings,
    )


def store_chunks(
    client: weaviate.WeaviateClient,
    settings: Settings,
    chunks: list[Document],
) -> None:
    """Add chunks to the already existing Weaviate collection."""

    vectorstore = get_vectorstore(
        client=client,
        settings=settings,
    )

    vectorstore.add_documents(chunks)