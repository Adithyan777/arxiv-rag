import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.llm import lmstudio_model_list, open_router_model_list, io_net_model_list
from components import (
    load_papers,
    get_paper_metadata,
    initialize_rag_components,
    perform_search,
    process_arxiv_paper
)

def main():
    # Set up page configuration
    st.set_page_config(
        page_title="ArXiv RAG System",
        page_icon="📚",
        layout="wide"
    )

    # Configure sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["RAG Q&A", "Add Research Paper"])

    # Initialize RAG components
    st.sidebar.markdown("---")
    st.sidebar.title("RAG Configuration")

    # LLM Provider selection
    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        ["ionet", "lmstudio", "openrouter"],
        help="Select the LLM provider for responses",
        key="llm_provider"
    )

    # Model selection
    model_options = {
        "lmstudio": lmstudio_model_list,
        "openrouter": open_router_model_list,
        "ionet": io_net_model_list
    }
    model_name = st.sidebar.selectbox(
        "Model",
        model_options[llm_provider],
        help="Select the model to use for responses",
        key="model_name"
    )

    # Initialize RAG components
    llm, smol, vector_store, embeddings, client = initialize_rag_components(llm_provider, model_name)

    if page == "RAG Q&A":
        display_qa_section(llm, smol, vector_store, embeddings, client)
    else:
        display_add_paper_section()

def display_qa_section(llm, smol, vector_store, embeddings, client):
    st.title("ArXiv Research Assistant - Q&A 📚")
    st.markdown("""
    This RAG (Retrieval Augmented Generation) system helps you explore and understand ArXiv papers more effectively.
    Ask questions about papers, and get accurate responses based on the paper's content.
    """)

    # Search interface
    search_query = st.text_input(
        "Enter your research question:", 
        placeholder="How is visual hallucination still an issue in LVLMs?"
    )

    # Advanced options
    paper_id = None
    use_paper_id = False
    
    with st.expander("Advanced Options", expanded=False):
        use_paper_id = st.checkbox("Restrict search to specific paper", value=False)
        if use_paper_id:
            paper_options = load_papers()
            selected_paper = st.selectbox(
                "Select paper:",
                options=list(paper_options.keys()),
                format_func=lambda x: x.split(" (")[0]
            )
            if selected_paper:
                paper_id = paper_options[selected_paper]
                # Display paper metadata
                try:
                    metadata = get_paper_metadata(paper_id)
                    st.markdown("### Paper Details")
                    st.markdown(f"**Title:** {metadata.get('title', 'N/A')}")
                    st.markdown(f"**Authors:** {metadata.get('authors', 'N/A')}")
                except Exception as e:
                    st.warning(f"Could not load paper metadata: {str(e)}")

    # Handle search
    if st.button("Search"):
        if search_query:
            with st.spinner("Searching for relevant information..."):
                perform_search(
                    search_query=search_query,
                    use_paper_id=use_paper_id,
                    paper_id=paper_id,
                    llm=llm,
                    smol=smol,
                    vector_store=vector_store,
                    embeddings=embeddings,
                    client=client
                )
        else:
            st.warning("Please enter a search query.")

def display_add_paper_section():
    st.title("ArXiv Research Assistant - Add Paper 📑")
    st.markdown("""
    Add new research papers to the system by providing their ArXiv PDF links.
    The papers will be processed and made available for querying.
    """)

    arxiv_link = st.text_input(
        "ArXiv PDF Link",
        placeholder="https://arxiv.org/pdf/paper-id"
    )
    
    if st.button("Add Paper"):
        if arxiv_link and arxiv_link.strip():
            with st.spinner("Processing paper..."):
                try:
                    process_arxiv_paper(arxiv_link)
                    st.success(f"Successfully added paper.")
                except Exception as e:
                    st.error(f"{str(e)}")
        else:
            st.warning("Please enter an ArXiv PDF link")

    # # Footer
    # st.markdown("---")
    # st.markdown("Built with Streamlit, LangChain, and Qdrant")

if __name__ == "__main__":
    main()
