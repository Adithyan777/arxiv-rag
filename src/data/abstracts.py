import pandas as pd
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

class AbstractVectorStore:
    def __init__(self, collection_name="arxiv-abstracts", qdrant_url="http://localhost:6333"):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
        self.client = None
        self.vector_store = None
        
    def initialize_store(self):
        """Initialize the vector store and create collection if needed"""
        self.client = QdrantClient(url=self.qdrant_url)
        
        # Create collection if it doesn't exist
        # try:
        #     self.client.create_collection(
        #         collection_name=self.collection_name,
        #         vectors_config={
        #             "size": 768,
        #             "distance": "Cosine"
        #         }
        #     )
        # except Exception as e:
        #     print(f"Collection might already exist: {e}")
            
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )
        
    def add_single_abstract(self, abstract: str, id: str):
        """Add a single abstract document to the vector store"""
        document = Document(page_content=abstract, metadata={"id": str(id)})
        self.vector_store.add_documents([document])
        return True
        
    def load_from_csv(self, csv_path: str):
        """Load abstracts from CSV file and add to vector store"""
        df = pd.read_csv(csv_path)
        abstracts = [
            Document(
                page_content=row['abstract'],
                metadata={"id": str(row['id'])}
            ) for _, row in df.iterrows()
        ]
        self.vector_store.add_documents(abstracts)
        return len(abstracts)
        
    def get_vector_store(self):
        """Return the vector store instance"""
        return self.vector_store

def main():
    # Example usage when run as script
    store = AbstractVectorStore()
    metadata = {
        "title": "Dense Match Summarization for Faster Two-view Estimation",
        "authors": "['jonathan astermark', 'anders heyden', 'viktor larsson']",
        "abstract": "In this paper, we speed up robust two-view relative pose from dense correspondences. Previous work has shown that dense matchers can significantly improve both accuracy and robustness in the resulting pose. However, the large number of matches comes with a significantly increased runtime during robust estimation in RANSAC. To avoid this, we propose an efficient match summarization scheme which provides comparable accuracy to using the full set of dense matches, while having 10-100x faster runtime. We validate our approach on standard benchmark datasets together with multiple state-of-the-art dense matchers.",
        "created": "2025-06-03",
        "updated": "2025-06-03",
        "id": "2506.02893",
        "categories": "cs.CV"
    }
    store.initialize_store()
    store.add_single_abstract(metadata['abstract'], str(metadata['id']))
    print(f"Added abstract with ID: {metadata['id']} to the vector store.")

if __name__ == "__main__":
    main()