from pathlib import Path
from langchain_docling import DoclingLoader
from utils import merge_same_heading_docs, extract_metadata
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
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    client = QdrantClient(url="http://localhost:6333")
    
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": vector_size, "distance": "Cosine"}
        )
    except Exception as e:
        print(f"Collection might already exist: {e}")
    
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )

def process_markdown_file(file_path: Path, directory_name: str) -> List[Document]:
    loader = DoclingLoader(file_path=f"{directory_name}/{file_path.name}")
    docs = loader.load()
    
    merged_docs = merge_same_heading_docs(docs)
    
    for doc in merged_docs:
        try:
            metadata = extract_metadata(file_path.name)
            dl_meta = doc.metadata.get("dl_meta", {})
            headings = dl_meta.get("headings", [])
            metadata['heading'] = headings[0] if headings else "No heading"
            doc.metadata = metadata
        except (ValueError, FileNotFoundError) as e:
            print(f"Error processing document: {e}")
    
    return merged_docs

def process_directory(
    directory_path: str,
    collection_name: str,
    splitter: Optional[DocumentSplitter] = None
) -> int:
    if splitter is None:
        splitter = DocumentSplitter()
    
    vector_store = init_vector_store(collection_name)
    md_files = list(Path(directory_path).glob("*.md"))
    total_docs = 0
    
    print(f"Found {len(md_files)} markdown files in {directory_path}")
    
    for file_path in md_files:
        print(f"Processing file: {file_path.name}")
        merged_docs = process_markdown_file(file_path, directory_path)
        print(f"Found {len(merged_docs)} merged documents")
        
        all_chunked_docs = []
        for doc in merged_docs:
            chunked = splitter.split_section_doc(doc)
            all_chunked_docs.extend(chunked)
        
        total_docs += len(all_chunked_docs)
        vector_store.add_documents(all_chunked_docs)
        print(f"Processed {len(all_chunked_docs)} chunks from {file_path.name}")
    
    return total_docs

if __name__ == "__main__":
    DIRECTORY_PATH = "markdown"
    COLLECTION_NAME = "arxiv-cvpr-main"
    
    total_processed = process_directory(DIRECTORY_PATH, COLLECTION_NAME)
    print(f"Total documents processed: {total_processed}")
    print("All documents have been processed and added to the vector store.")