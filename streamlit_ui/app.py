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
from agents import compare_papers, summarize_paper

def main():
    # Set up page configuration
    st.set_page_config(
        page_title="ArXiv RAG System",
        page_icon="📚",
        layout="wide"
    )

    # Configure sidebar
    st.sidebar.title("Navigation")
    # Update navigation options
    page = st.sidebar.radio("Go to", ["RAG Q&A", "Add Research Paper", "Agents"])

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
    elif page == "Add Research Paper":
        display_add_paper_section()
    else:
        display_agents_section()

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
        search_mode = st.radio(
            "Search Mode",
            ["Restrict to Specific Paper","Use Global Context"],
            key="search_mode"
        )
        
        if search_mode == "Restrict to Specific Paper":
            paper_selection_mode = st.radio(
                "Paper Selection Mode",
                ["Route Automatically", "Select Yourself"],
                index=0  # Default to "Route Automatically"
            )
            
            if paper_selection_mode == "Select Yourself":
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
            with st.spinner("Generating response..."):
                # Determine if we should use paper_id based on search mode
                use_global_context = search_mode == "Use Global Context"
                use_paper_id = search_mode == "Restrict to Specific Paper"
                
                if use_paper_id and paper_selection_mode == "Route Automatically":
                    # Let the system automatically find the most relevant paper
                    paper_id = None
                # For "Select Yourself" mode, paper_id is already set from the selectbox
                
                perform_search(
                    search_query=search_query,
                    use_paper_id=use_paper_id,
                    use_global_context=use_global_context,
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

def display_agents_section():
    st.title("ArXiv Research Assistant - Agents 🤖")
    st.markdown("""
    Use specialized agents to analyze and compare research papers.
    Choose an agent type below to get started.
    """)

    agent_type = st.radio("Select Agent Type", ["Compare Agent", "Summarize Agent"])

    if agent_type == "Compare Agent":
        st.subheader("Compare Papers")
        st.markdown("Select multiple papers to compare their methodologies, results, and approaches.")
        
        paper_options = load_papers()
        selected_papers = st.multiselect(
            "Select papers to compare",
            options=list(paper_options.keys()),
            format_func=lambda x: x.split(" (")[0],
            max_selections=2
        )
        
        aspects = st.multiselect(
            "Select aspects to compare",
            ["Methodology", "Results", "Architecture", "Performance", "Dataset"],
            default=["Methodology"],
            max_selections=1,
        )
        
        if st.button("Compare Papers") and len(selected_papers) > 1:
            paper_ids = [paper_options[paper] for paper in selected_papers]
            with st.spinner("Generating comparison..."):
                result = compare_papers(paper_ids, aspects)
                st.write(result)
        elif len(selected_papers) <= 1:
            st.info("Please select at least 2 papers to compare")

    else:  # Summarize Agent
        st.subheader("Summarize Paper")
        st.markdown("Get a detailed summary of a research paper with optional focus areas.")
        
        paper_options = load_papers()
        selected_paper = st.selectbox(
            "Select paper to summarize",
            options=list(paper_options.keys()),
            format_func=lambda x: x.split(" (")[0]
        )
        
        focus_area = st.selectbox(
            "Focus area (optional)",
            ["Full Paper", "Methodology", "Results", "Conclusions"]
        )
        
        if st.button("Generate Summary") and selected_paper:
            paper_id = paper_options[selected_paper]
            with st.spinner("Generating summary..."):
                result = summarize_paper(paper_id, focus_area)
                st.write(result)

    # # Footer
    # st.markdown("---")
    # st.markdown("Built with Streamlit, LangChain, and Qdrant")

if __name__ == "__main__":
    main()
