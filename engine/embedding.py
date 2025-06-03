from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

collection_name = "arxiv-cvpr-main"

embeddings = OllamaEmbeddings(
   model="nomic-embed-text:latest"
)

client = QdrantClient(
    url="http://localhost:6333",
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

def get_vector_store():
    """
    Returns the Qdrant vector store instance.
    """
    return vector_store

def get_embeddings():
    """
    Returns the Ollama embeddings instance.
    """
    return embeddings

def get_qdrant_client():
    """
    Returns the Qdrant client instance.
    """
    return client