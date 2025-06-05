import streamlit as st
import json
import os
from typing import Dict, Tuple, Optional, List
import re
from engine.llm import get_llm
from engine.embedding import get_vector_store, get_embeddings, get_qdrant_client
from utils import (
    get_paper_id_from_search_query,
    get_context_for_qa,
    get_context_for_qa_without_id,
    get_rewritten_queries,
    get_llm_generation_using_context,
    clean_arxiv_md_text,
    merge_same_heading_docs,
    get_arxiv_metadata_from_paper_id,
    add_paper_to_json
)
import streamlit as st
import pandas as pd
import os
from typing import Dict, Tuple, Optional, List
from docling.document_converter import DocumentConverter
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from init_vector_stores.papers import DocumentSplitter
from init_vector_stores.abstracts import AbstractVectorStore
import json

def load_papers(filepath: str = "final_papers.json") -> Dict[str, str]:
    """Load and parse the papers JSON file."""
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_path = os.path.join(project_root, filepath)
        
        if not os.path.exists(abs_path):
            abs_path = os.path.join(os.getcwd(), filepath)
            st.write(f"Trying current directory: {abs_path}")
        
        with open(abs_path, 'r') as f:
            data = json.load(f)
            
        # Create dictionary of title+id -> id mappings
        return {f"{paper['title']} ({paper['id']})": paper['id'] 
                for paper in data['papers']}
                
    except Exception as e:
        st.error(f"Error loading papers: {str(e)}")
        return {}

def get_paper_metadata(paper_id: str, filepath: str = "final_papers.json") -> Dict:
    """Retrieve metadata for a specific paper ID."""
    if not paper_id:
        raise ValueError("Paper ID cannot be empty")
    
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_path = os.path.join(project_root, filepath)
        
        if not os.path.exists(abs_path):
            abs_path = os.path.join(os.getcwd(), filepath)
            
        with open(abs_path, 'r') as f:
            data = json.load(f)
        
        # Find paper with matching ID
        paper = next((p for p in data['papers'] if p['id'] == paper_id), None)
        
        if not paper:
            raise ValueError(f"No metadata found for ID: {paper_id}")
            
        return {k: v for k, v in paper.items() if v}  # Filter out empty values
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Paper list not found at: {abs_path}")

@st.cache_resource
def initialize_rag_components(provider: str, model: str):
    """Initialize RAG components with caching."""
    llm, smol = get_llm(provider, model)
    vector_store = get_vector_store()
    embeddings = get_embeddings()
    client = get_qdrant_client()
    return llm, smol, vector_store, embeddings, client

def display_paper_metadata(paper_id: str):
    """Display paper metadata in the UI."""
    try:
        metadata = get_paper_metadata(paper_id)
        st.markdown("### Paper Details")
        st.markdown(f"**Title:** {metadata.get('title', 'N/A')}")
        st.markdown(f"**Authors:** {metadata.get('authors', 'N/A')}")
    except Exception as e:
        st.warning(f"Could not load paper metadata: {str(e)}")

def perform_search(
    search_query: str,
    use_paper_id: bool,
    use_global_context: bool,
    paper_id: Optional[str],
    llm,
    smol,
    vector_store,
    embeddings,
    client
) -> None:
    """Execute the search and display results."""
    try:
        if use_global_context:
            # Global context search across all papers
            rewritten_queries = get_rewritten_queries(search_query, smol)
            results = get_context_for_qa_without_id(rewritten_queries, vector_store)
            st.info("Searching across all papers")
        elif use_paper_id and paper_id:
            # Direct search with paper ID
            paper_title = get_paper_metadata(paper_id).get('title', 'N/A')
            st.markdown(f"Searching within paper ID: [{paper_id} - {paper_title}](https://arxiv.org/pdf/{paper_id}.pdf)")
            rewritten_queries = get_rewritten_queries(search_query, smol)
            results = get_context_for_qa(paper_id, rewritten_queries, vector_store)
        else:
            # Auto paper detection
            with st.spinner("Detecting relevant paper..."):
                paper_id, rewritten_queries = get_paper_id_from_search_query(
                    search_query,
                    "arxiv-abstracts",
                    embeddings,
                    client,
                    smol
                )
                st.info(f"Found relevant paper: {paper_id} - {get_paper_metadata(paper_id).get('title', 'N/A')}(https://arxiv.org/pdf/{paper_id}.pdf)")
            results = get_context_for_qa(paper_id, rewritten_queries, vector_store)
            
        context = [res for res, score in results if res.page_content and len(res.page_content) > 0]
        
        if context:
            display_search_results(context, search_query, rewritten_queries, llm, vector_store)
        else:
            st.warning("No relevant context found for your query.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def display_search_results(
    context: List,
    search_query: str,
    rewritten_queries: List[str],
    llm,
    vector_store
) -> None:
    """Display search results in the UI."""
    st.write("### Answer")
    response = get_llm_generation_using_context(
        context=context,
        question=search_query,
        llm=llm
    )
    
    if "I don't know" in response:
        st.warning("Selected paper doesnt have enough context to answer the question, trying with overall context...")
        global_results = get_context_for_qa_without_id(rewritten_queries, vector_store)
        global_context = [res for res, score in global_results if res.page_content and len(res.page_content) > 0]
        
        if global_context:
            response = get_llm_generation_using_context(
                context=global_context,
                question=search_query,
                llm=llm
            )
        else:
            response = "No relevant context found for your query."
            
    st.write(response)
    
    # Show context
    with st.expander("View Source Context", expanded=False):
        st.markdown("### Reference Context")
        for idx, ctx in enumerate(context):
            st.markdown(f"**Excerpt {idx + 1}:**")
            st.markdown(ctx.page_content)
            st.markdown("---")

def process_arxiv_paper(arxiv_link: str) -> str:
    """
    Process a new ArXiv paper from its PDF link.
    This is a placeholder function that should be implemented to:
    1. Download the PDF
    2. Extract text and process it
    3. Add to vector store
    4. Update paper list
    
    Args:
        arxiv_link (str): URL to the ArXiv PDF
        
    Returns:
        str: The paper ID of the processed paper
    """
    if not arxiv_link or not arxiv_link.startswith("https://arxiv.org/pdf/"):
        raise ValueError("Invalid ArXiv PDF link")
        
    # Extract paper ID from the link
    paper_id = arxiv_link.split('/')[-1].replace('.pdf', '')
    if not paper_id:
        raise ValueError("Could not extract paper ID from the link")
    if not re.match(r'^\d{4}\.\d{5}$', paper_id):
        raise ValueError("Invalid paper ID format. Expected format: XXXX.XXXXX")
    
    st.info(f"Processing paper with ID: {paper_id} from {arxiv_link}")

    converter = DocumentConverter()
    try:
        # ------------- Download PDF and convert to markdown -------------

        text = clean_arxiv_md_text(converter.convert(arxiv_link).document.export_to_markdown())
        st.info(f"PDF converted to text successfully for paper ID: {paper_id}")
        
        # ------------- Convert markdown to documents -------------

        temp_file_path = f"temp_{paper_id}.md"
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        loader = DoclingLoader(
            file_path=temp_file_path,
        )
        docs = loader.load()
        merged_docs = merge_same_heading_docs(docs)
        # remove the temporary file after loading
        os.remove(temp_file_path)

        # ------------- Change metadata for each document -------------
        global_metadata = get_arxiv_metadata_from_paper_id(paper_id)
        for doc in merged_docs:
            metadata = global_metadata.copy()
            dl_meta = doc.metadata.get("dl_meta", {})
            headings = dl_meta.get("headings", [])
            metadata['heading'] = headings[0] if headings else "No heading"
            doc.metadata = metadata

        # -------------- Chunking the documents --------------

        splitter = DocumentSplitter()
        if not splitter:
            raise Exception("Document splitter is not initialized. Please check your configuration.")
        all_chunked_docs = []
        for doc in merged_docs:
            chunked = splitter.split_section_doc(doc)
            all_chunked_docs.extend(chunked)
        st.info(f"Documents processed successfully for paper ID: {paper_id}. Total chunks: {len(all_chunked_docs)}")
        
        # ------------- Add to vector store -------------

        vector_store = get_vector_store()
        if not vector_store:
            raise Exception("Vector store is not initialized. Please check your configuration.")
        vector_store.add_documents(all_chunked_docs)

        st.info(f"Added {len(all_chunked_docs)} chunks to the vector store for paper ID: {paper_id}")

        # ------------- Update paper list CSV file and abstract vector store -------------

        abs_vector_store = AbstractVectorStore()
        abs_vector_store.initialize_store()
        abs_vector_store.add_single_abstract(
            abstract=global_metadata['abstract'],
            id=global_metadata['id']
        )
        st.info(f"Added abstract for paper ID: {paper_id} to the vector store")
        add_paper_to_json(paper_metadata = global_metadata)
        st.info(f"Updated paper list with metadata for paper ID: {paper_id}")      

    except Exception as e:
        raise Exception(f"Failed to process paper: {str(e)}")

    return paper_id
