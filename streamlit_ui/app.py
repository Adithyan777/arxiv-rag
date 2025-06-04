import streamlit as st
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.llm import get_llm, lmstudio_model_list, open_router_model_list, io_net_model_list
from engine.embedding import get_vector_store, get_embeddings, get_qdrant_client
from engine.utils import get_paper_id_from_search_query, get_context_for_qa, get_context_for_qa_without_id, get_rewritten_queries, get_llm_generation_using_context

# Page config
st.set_page_config(
    page_title="ArXiv RAG System",
    page_icon="📚",
    layout="wide"
)

# Sidebar
st.sidebar.title("RAG Configuration")

llm_provider = st.sidebar.selectbox(
    "LLM Provider",
    ["ionet", "lmstudio", "openrouter"],
    help="Select the LLM provider for responses",
    key="llm_provider"
)

# Dynamic model selection based on provider
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
@st.cache_resource
def initialize_rag_components(provider, model):
    llm, smol = get_llm(provider, model)
    vector_store = get_vector_store()
    embeddings = get_embeddings()
    client = get_qdrant_client()
    return llm, smol, vector_store, embeddings, client

# Helper function to load papers
@st.cache_data
def load_papers(filepath="final_paper_list.csv"):
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_path = os.path.join(project_root, filepath)
        # st.write(f"Looking for paper list at: {abs_path}")  # Debug print
        
        if not os.path.exists(abs_path):
            st.error(f"File not found at: {abs_path}")
            # Try current working directory
            abs_path = os.path.join(os.getcwd(), filepath)
            st.write(f"Trying current directory: {abs_path}")
            
        df = pd.read_csv(abs_path)
        df['id'] = df['id'].astype(str).str.strip()
        # Create a dict of "title (id)" : "id" pairs
        paper_options = {f"{row['title']} ({row['id']})": row['id'] 
                        for _, row in df.iterrows()}
        return paper_options
    except FileNotFoundError as e:
        st.error(f"Paper list not found. Tried paths:\n- {abs_path}")
        return {}
    except Exception as e:
        st.error(f"Error loading papers: {str(e)}")
        return {}

def get_paper_metadata(paper_id: str, filepath: str = "final_paper_list.csv"):
    """Retrieve metadata for a specific paper ID"""
    if not paper_id:
        raise ValueError("Paper ID cannot be empty")
    
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_path = os.path.join(project_root, filepath)
        
        if not os.path.exists(abs_path):
            # Try current working directory
            abs_path = os.path.join(os.getcwd(), filepath)
        
        df = pd.read_csv(abs_path)
        df['id'] = df['id'].astype(str).str.strip()
        paper_data = df[df['id'].str.contains(paper_id, regex=False)]
        
        if paper_data.empty:
            raise ValueError(f"No metadata found for ID: {paper_id}")
        
        metadata = {}
        columns = ['id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors']
        
        for col in columns:
            value = paper_data[col].iloc[0]
            if pd.notna(value) and str(value).strip():
                metadata[col] = value
        
        return metadata
    except FileNotFoundError:
        raise FileNotFoundError(f"Paper list not found at: {abs_path}")

# Main content
st.title("ArXiv Research Assistant 📚")
st.markdown("""
This RAG (Retrieval Augmented Generation) system helps you explore and understand ArXiv papers more effectively.
Ask questions about papers, and get accurate responses based on the paper's content.
""")

# Initialize components
llm, smol, vector_store, embeddings, client = initialize_rag_components(llm_provider, model_name)

# Search interface
search_query = st.text_input("Enter your research question:", placeholder="How is visual hallucination still an issue in LVLMs?")

with st.expander("Advanced Options", expanded=False):
    use_paper_id = st.checkbox("Restrict search to specific paper", value=False)
    if use_paper_id:
        paper_options = load_papers()
        selected_paper = st.selectbox(
            "Select paper:",
            options=list(paper_options.keys()),
            format_func=lambda x: x.split(" (")[0]  # Show only title in dropdown
        )
        paper_id = paper_options[selected_paper]
        
        # Display paper metadata
        try:
            metadata = get_paper_metadata(paper_id)
            st.markdown("### Paper Details")
            st.markdown(f"**Title:** {metadata.get('title', 'N/A')}")
            st.markdown(f"**Authors:** {metadata.get('authors', 'N/A')}")
        except Exception as e:
            st.warning(f"Could not load paper metadata: {str(e)}")

if st.button("Search"):
    if search_query:
        with st.spinner("Searching for relevant information..."):
            try:
                if use_paper_id and paper_id:
                    # Direct search with paper ID
                    st.info(f"Searching within paper ID: {paper_id} - {get_paper_metadata(paper_id).get('title', 'N/A')}")
                    rewritten_queries = get_rewritten_queries(search_query, smol)
                    print(f"Rewritten queries: {rewritten_queries}")
                    results = get_context_for_qa(paper_id, rewritten_queries, vector_store)
                    context = [res for res, score in results if res.page_content and len(res.page_content) > 0]
                else:
                    # Auto paper detection
                    with st.spinner("Detecting relevant paper..."):
                        paper_id, rewritten_queries = get_paper_id_from_search_query(
                            search_query,
                            "arxiv-cvpr-main",
                            embeddings,
                            client,
                            smol
                        )
                        st.info(f"Found relevant paper: {paper_id} - {get_paper_metadata(paper_id).get('title', 'N/A')}")
                    results = get_context_for_qa(paper_id, rewritten_queries, vector_store)
                    context = [res for res, score in results if res.page_content and len(res.page_content) > 0]
                    print(f"Context found for paper ID {paper_id}: {len(context)} excerpts")
                # Get LLM response
                if context:
                    st.write("### Answer")
                    response = get_llm_generation_using_context(
                        context=context,
                        question=search_query,
                        llm=llm
                    )
                    if "I don't know" in response: # TODO: Can make this an LLM call or anything to check negative response
                        st.warning("Routing failed, trying with overall context...")
                        global_results = get_context_for_qa_without_id(rewritten_queries, vector_store)
                        global_context = [res for res, score in global_results if res.page_content and len(res.page_content) > 0]
                        print(f"Global context found: {len(global_context)} excerpts")
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
                else:
                    st.warning("No relevant context found for your query.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a search query.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, LangChain, and Qdrant")
