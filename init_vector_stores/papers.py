from pathlib import Path
import time
import threading
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_docling import DoclingLoader
from src.data import merge_same_heading_docs, extract_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

class DocumentSplitter:
    def __init__(self, chunk_size: int = 1750, chunk_overlap: int = 175):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " "],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def show_loading_animation(self, stop_event, message="Processing"):
        """Display a loading animation"""
        animation = "|/-\\"
        idx = 0
        while not stop_event.is_set():
            print(f"\r{message} {animation[idx % len(animation)]}", end="", flush=True)
            idx += 1
            time.sleep(0.1)

    def split_section_doc(self, doc: Document) -> List[Document]:
        """
        Given a Document whose .page_content may be very long, 
        return a list of Documents:
        - If below threshold: [doc] unchanged.
        - Otherwise: each chunk is a new Document with the same metadata.
        """
        text = doc.page_content
        if len(text) <= self.chunk_size:
            return [doc]

        sub_texts = self.splitter.split_text(text)
        chunked_docs = []
        for i, chunk_text in enumerate(sub_texts):
            meta = doc.metadata.copy()
            meta["chunk_index"] = i
            chunked_docs.append(Document(page_content=chunk_text, metadata=meta))
        return chunked_docs

def init_vector_store(collection_name: str, vector_size: int = 768) -> QdrantVectorStore:
    print("🚀 Initializing vector store...")
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    client = QdrantClient(url="http://localhost:6333")
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_size, "distance": "Cosine"}
        )
        print(f"✅ Created new collection: {collection_name}")
    except Exception as e:
        print(f"ℹ️  Collection '{collection_name}' already exists")
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    
    print("✅ Vector store initialized successfully!")
    return vector_store

def process_markdown_file(file_path: Path, directory_name: str) -> List[Document]:
    print(f"📄 Loading document: {file_path.name}")
    
    # Start loading animation
    stop_event = threading.Event()
    loading_thread = threading.Thread(target=DocumentSplitter().show_loading_animation, args=(stop_event, f"Loading {file_path.name}"))
    loading_thread.start()
    
    try:
        loader = DoclingLoader(file_path=f"{directory_name}/{file_path.name}")
        docs = loader.load()
        
        stop_event.set()
        loading_thread.join()
        print(f"\r✅ Loaded {len(docs)} sections from {file_path.name}{' ' * 20}")
        
        print(f"🔄 Merging sections...")
        merged_docs = merge_same_heading_docs(docs)
        print(f"✅ Merged into {len(merged_docs)} documents")
        
        print(f"📝 Extracting metadata...")
        for doc in merged_docs:
            try:
                metadata = extract_metadata(file_path.name)
                dl_meta = doc.metadata.get("dl_meta", {})
                headings = dl_meta.get("headings", [])
                metadata['heading'] = headings[0] if headings else "No heading"
                doc.metadata = metadata
            except (ValueError, FileNotFoundError) as e:
                print(f"⚠️  Error processing document metadata: {e}")
        
        return merged_docs
        
    except Exception as e:
        stop_event.set()
        loading_thread.join()
        print(f"\r❌ Error loading {file_path.name}: {e}{' ' * 20}")
        return []

def process_directory(
    directory_path: str,
    collection_name: str,
    splitter: Optional[DocumentSplitter] = None
) -> int:
    print("=" * 60)
    print("📚 ArXiv Papers Vector Store Loader")
    print("=" * 60)
    
    if splitter is None:
        splitter = DocumentSplitter()
    
    vector_store = init_vector_store(collection_name)
    md_files = list(Path(directory_path).glob("*.md"))
    total_docs = 0
    
    print(f"📁 Found {len(md_files)} markdown files in {directory_path}")
    
    if len(md_files) == 0:
        print("❌ No markdown files found!")
        return 0
    
    print("\n🚀 Starting document processing...")
    
    for i, file_path in enumerate(md_files, 1):
        print(f"\n📊 Progress: {i}/{len(md_files)} files")
        print("-" * 40)
        
        merged_docs = process_markdown_file(file_path, directory_path)
        
        if not merged_docs:
            print(f"⚠️  Skipping {file_path.name} - no documents loaded")
            continue
            
        print(f"✂️  Splitting documents into chunks...")
        all_chunked_docs = []
        for j, doc in enumerate(merged_docs):
            chunked = splitter.split_section_doc(doc)
            all_chunked_docs.extend(chunked)
            if (j + 1) % 5 == 0 or j == len(merged_docs) - 1:
                print(f"\r   Processed: {j + 1}/{len(merged_docs)} sections", end="", flush=True)
        
        print(f"\n🚀 Embedding and storing {len(all_chunked_docs)} chunks...")
        
        # Start loading animation for embedding process
        stop_event = threading.Event()
        loading_thread = threading.Thread(target=splitter.show_loading_animation, args=(stop_event, "Generating embeddings"))
        loading_thread.start()
        
        try:
            vector_store.add_documents(all_chunked_docs)
            stop_event.set()
            loading_thread.join()
            print(f"\r✅ Successfully stored {len(all_chunked_docs)} chunks from {file_path.name}!{' ' * 30}")
        except Exception as e:
            stop_event.set()
            loading_thread.join()
            print(f"\r❌ Error storing chunks: {e}{' ' * 30}")
            continue
        
        total_docs += len(all_chunked_docs)
    
    return total_docs

if __name__ == "__main__":
    DIRECTORY_PATH = "markdown"
    COLLECTION_NAME = "arxiv-cvpr-main"
    
    total_processed = process_directory(DIRECTORY_PATH, COLLECTION_NAME)
    
    print("\n" + "=" * 60)
    if total_processed > 0:
        print(f"🎉 SUCCESS: Processed {total_processed} document chunks!")
        print("✅ All documents have been processed and added to the vector store.")
    else:
        print("❌ No documents were processed.")
    print("=" * 60)