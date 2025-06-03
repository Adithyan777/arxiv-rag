import pandas as pd
from langchain_core.documents import Document

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

collection_name = "arxiv-abstracts"

embeddings = OllamaEmbeddings(
   model="nomic-embed-text:latest"
)

client = QdrantClient(
    url="http://localhost:6333",
)

client.create_collection(
    collection_name=collection_name,
    vectors_config={
        "size": 768, # Size of the embedding vector
        "distance": "Cosine"
    }
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

df = pd.read_csv('final_paper_list.csv')

abstr = df['abstract']
ids = df['id']
df = pd.DataFrame({
    "page_content": abstr,
    "metadata": [id for id in ids]
})
df = df.rename(columns={"page_content": "content", "metadata": "metadata"})


abstracts = [Document(page_content=row['content'], metadata={"id": row['metadata']}) for _, row in df.iterrows()]

vector_store.add_documents(abstracts)

print(f"Added {len(abstracts)} abstracts to the vector store.")
