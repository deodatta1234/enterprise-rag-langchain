from dotenv import load_dotenv

from rag_chatbot.config import load_settings
from rag_chatbot.indexing import connect_weaviate, create_embeddings
from langchain_weaviate import WeaviateVectorStore


load_dotenv()
settings = load_settings()
client = connect_weaviate(settings)

try:
    collection = client.collections.use(settings.collection_name)

    # 1. Confirm stored objects contain vectors.
    response = collection.query.fetch_objects(
        limit=3,
        include_vector=True,
    )

    print(f"Stored chunks returned: {len(response.objects)}")

    for item in response.objects:
        vector = item.vector

        if isinstance(vector, dict):
            dimensions = {
                name: len(values)
                for name, values in vector.items()
            }
        else:
            dimensions = len(vector) if vector else 0

        print(
            f"Document: {item.properties.get('document_id')} | "
            f"Page: {item.properties.get('page_number')} | "
            f"Vector dimensions: {dimensions}"
        )

    # 2. Confirm semantic search works.
    vector_store = WeaviateVectorStore(
        client=client,
        index_name=settings.collection_name,
        text_key="text",
        embedding=create_embeddings(settings),
    )

    results = vector_store.similarity_search(
        "How long are security audit logs retained?",
        k=3,
    )

    print("\nSemantic search results:")
    for result in results:
        print(
            f"- {result.metadata['document_id']}, "
            f"page {result.metadata['page_number']}: "
            f"{result.page_content[:150]}"
        )

finally:
    client.close()