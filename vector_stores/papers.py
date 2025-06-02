from pathlib import Path
from langchain_docling import DoclingLoader
from utils import merge_docs
from metadata import extract_metadata
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing import List

from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

directory_path = "markdown"
directory_name = "markdown"

collection_name = "arxiv-cvpr-main"

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

count = 0

md_files = list(Path(directory_path).glob("*.md"))
print(f"Found {len(md_files)} JSON files in {directory_path}")

for file_path in md_files:
    paper_id = file_path.stem
    filename = file_path.name
    print(f"Processing file: {filename}")

    # ---------- Load the document using DoclingLoader ----------
    
    loader = DoclingLoader(
        file_path=f"{directory_name}/{filename}"
    )
    docs = loader.load()

    # ---------- Merge documents based on headings ----------

    merged_docs = merge_docs(docs)
    print(f"Found {len(merged_docs)} merged documents:")

    # ---------------- Modify metadata ----------------------

    for i, doc in enumerate(merged_docs):
        try:
            metadata = extract_metadata(filename)
            # Add the heading to the metadata
            dl_meta = doc.metadata.get("dl_meta", {})
            headings = dl_meta.get("headings", [])
            if headings:
                metadata['heading'] = headings[0]
            else:
                metadata['heading'] = "No heading"
            
            # Update the document's metadata
            doc.metadata = metadata
            
        except ValueError as e:
            print(f"Error processing document {i+1}: {e}")
        except FileNotFoundError as e:
            print(e)

# ---------- Split long sections into smaller chunks ----------

    LONG_SECTION_THRESHOLD = 1750

    # 1) Configure your splitter
    chunk_size     = LONG_SECTION_THRESHOLD
    chunk_overlap  = 175
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "], 
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    def split_section_doc(doc: Document) -> List[Document]:
        """
        Given a Document whose .page_content may be very long, 
        return a list of Documents:
        - If below threshold: [doc] unchanged.
        - Otherwise: each chunk is a new Document with the same metadata.
        """
        text = doc.page_content
        if len(text) <= LONG_SECTION_THRESHOLD:
            # nothing to split; keep as‐is
            return [doc]

        # Otherwise, run the splitter
        sub_texts = splitter.split_text(text)  # → List[str]
        chunked_docs = []
        for i, chunk_text in enumerate(sub_texts):
            # Clone metadata, but you might want to note which sub‐chunk this is
            meta = doc.metadata.copy()
            # Optionally add an index so you know the chunk number inside that section:
            meta["chunk_index"] = i
            # (You already have: meta["dl_meta"]["headings"], meta["paper_id"], meta["paper_name"], etc.)
            chunked_docs.append(Document(page_content=chunk_text, metadata=meta))
        return chunked_docs

    # 3) Example usage over all merged_docs:
    all_chunked_docs = []
    for sec_doc in merged_docs:
        chunked = split_section_doc(sec_doc)
        all_chunked_docs.extend(chunked)

    print(f"Total documents after splitting: {len(all_chunked_docs)}")
    count += len(all_chunked_docs)

# ---------- Create a vector store and add documents ----------

    vector_store.add_documents(all_chunked_docs)

print(f"Total documents processed: {count}")
print("All documents have been processed and added to the vector store.")