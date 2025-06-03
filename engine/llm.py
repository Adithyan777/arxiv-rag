# LLM config

from langchain_openai import ChatOpenAI
from langchain import hub
from os import getenv
from dotenv import load_dotenv
load_dotenv()

# llm = ChatOpenAI(
def get_llm(provider="lmstudio"):
    """Get the LLM instance based on provider.
    
    Args:
        provider (str): The LLM provider to use ("lmstudio", "openrouter", or "ionet")
    """
    if provider == "lmstudio":
        llm = ChatOpenAI(
            model = "qwen/qwen3-4b",
            base_url = "http://127.0.0.1:1234/v1"
        )
        smol = ChatOpenAI(
            model = "qwen/qwen3-4b", 
            base_url = "http://127.0.0.1:1234/v1"
        )
    elif provider == "openrouter":
        llm = ChatOpenAI(
            model = "qwen/qwen3-8b:free",
            base_url = "https://openrouter.ai/api/v1",
            api_key = getenv("OPENROUTER_API_KEY")
        )
        smol = ChatOpenAI(
            model = "qwen/qwen3-8b:free",
            base_url = "https://openrouter.ai/api/v1", 
            api_key = getenv("OPENROUTER_API_KEY")
        )
    elif provider == "ionet":
        llm = ChatOpenAI(
            model = "Qwen/QwQ-32B",
            base_url = "https://api.intelligence.io.solutions/api/v1",
            api_key = getenv("OPENAI_API_KEY")
        )
        smol = ChatOpenAI(
            model = "Qwen/QwQ-32B",
            base_url = "https://api.intelligence.io.solutions/api/v1",
            api_key = getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return llm, smol