"""Configuration management for the ArXiv RAG application."""

import streamlit as st
from typing import Dict, List

from engine.llm import lmstudio_model_list, open_router_model_list, io_net_model_list

# Model configuration
MODEL_OPTIONS: Dict[str, List[str]] = {
    "lmstudio": lmstudio_model_list,
    "openrouter": open_router_model_list,
    "ionet": io_net_model_list
}

# Page configuration
PAGE_CONFIG = {
    "page_title": "ArXiv RAG System",
    "page_icon": "📚",
    "layout": "wide"
}

PAPERS_CSV_FILENAME = "final_paper_list.csv"

def setup_page_config():
    """Initialize the Streamlit page configuration."""
    st.set_page_config(**PAGE_CONFIG)

def setup_sidebar() -> tuple:
    """Setup and return the sidebar configuration."""
    st.sidebar.title("RAG Configuration")
    
    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        ["ionet", "lmstudio", "openrouter"],
        help="Select the LLM provider for responses",
        key="llm_provider"
    )
    
    model_name = st.sidebar.selectbox(
        "Model",
        MODEL_OPTIONS[llm_provider],
        help="Select the model to use for responses",
        key="model_name"
    )
    
    return llm_provider, model_name
