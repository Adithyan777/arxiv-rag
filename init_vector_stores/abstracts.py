import json
import time
import threading
import sys
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
        
    def show_loading_animation(self, stop_event, message="Processing"):
        """Display a loading animation"""
        animation = "|/-\\"
        idx = 0
        while not stop_event.is_set():
            print(f"\r{message} {animation[idx % len(animation)]}", end="", flush=True)
            idx += 1
            time.sleep(0.1)
        
    def initialize_store(self):
        """Initialize the vector store and create collection if needed"""
        print("🚀 Initializing vector store...")
        
        # Start loading animation
        stop_event = threading.Event()
        loading_thread = threading.Thread(target=self.show_loading_animation, args=(stop_event, "Connecting to Qdrant"))
        loading_thread.start()
        
        try:
            self.client = QdrantClient(url=self.qdrant_url)

            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        "size": 768,
                        "distance": "Cosine"
                    }
                )
            except Exception as e:
                pass  # Collection might already exist
                
            self.vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            
            # Stop loading animation
            stop_event.set()
            loading_thread.join()
            print(f"\r✅ Vector store initialized successfully!{' ' * 20}")
            
        except Exception as e:
            stop_event.set()
            loading_thread.join()
            print(f"\r❌ Error initializing vector store: {e}{' ' * 20}")
            raise
    
    def load_from_json(self, json_path: str):
        """Load abstracts from JSON file and add to vector store"""
        print("📖 Reading JSON file...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get('papers', [])
        total_papers = len(papers)
        print(f"📊 Found {total_papers} papers in JSON file")
        
        abstracts = []
        valid_papers = 0
        
        print("📝 Processing papers...")
        for i, paper in enumerate(papers):
            if 'abstract' in paper and paper['abstract']:
                abstracts.append(Document(
                    page_content=paper['abstract'],
                    metadata={
                        "id": str(paper['id']),
                        "title": paper['title'],
                        "categories": paper.get('categories', ''),
                        "authors": paper.get('authors', ''),
                        "created": paper.get('created', '')
                    }
                ))
                valid_papers += 1
            
            # Show progress
            if (i + 1) % 5 == 0 or i == total_papers - 1:
                progress = (i + 1) / total_papers * 100
                print(f"\r📊 Processing: {i + 1}/{total_papers} ({progress:.1f}%) - Valid papers: {valid_papers}", end="", flush=True)
        
        print(f"\n🚀 Embedding and storing {len(abstracts)} abstracts...")
        
        # Start loading animation for embedding process
        stop_event = threading.Event()
        loading_thread = threading.Thread(target=self.show_loading_animation, args=(stop_event, "Generating embeddings"))
        loading_thread.start()
        
        try:
            self.vector_store.add_documents(abstracts)
            stop_event.set()
            loading_thread.join()
            print(f"\r✅ Successfully stored {len(abstracts)} abstracts!{' ' * 30}")
        except Exception as e:
            stop_event.set()
            loading_thread.join()
            print(f"\r❌ Error storing abstracts: {e}{' ' * 30}")
            raise
            
        return len(abstracts)
        
    def get_vector_store(self):
        """Return the vector store instance"""
        return self.vector_store

def main():
    """Load papers from JSON file to the vector store"""
    import os
    
    print("=" * 60)
    print("🔬 ArXiv Abstract Vector Store Loader")
    print("=" * 60)
    
    # Initialize the vector store
    store = AbstractVectorStore()
    store.initialize_store()
    
    # Get the parent directory path (arxiv-rag folder)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Define JSON file path
    json_path = os.path.join(parent_dir, "initial_papers.json")
    
    # Load from JSON file if it exists
    if os.path.exists(json_path):
        try:
            print(f"📁 Found JSON file: {os.path.basename(json_path)}")
            count = store.load_from_json(json_path)
            print("\n" + "=" * 60)
            print(f"🎉 SUCCESS: Loaded {count} abstracts to vector store!")
            print("=" * 60)
        except Exception as e:
            print(f"\n❌ Error loading from JSON: {e}")
    else:
        print(f"❌ JSON file not found: {json_path}")

if __name__ == "__main__":
    main()