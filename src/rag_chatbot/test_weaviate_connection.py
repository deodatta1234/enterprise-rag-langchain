from dotenv import load_dotenv

from rag_chatbot.config import load_settings
from rag_chatbot.indexing import connect_weaviate


load_dotenv()
settings = load_settings()
client = connect_weaviate(settings)

try:
    print("Weaviate ready:", client.is_ready())
finally:
    client.close()