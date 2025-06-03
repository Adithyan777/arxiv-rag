# LLM config

from langchain_openai import ChatOpenAI
from langchain import hub
from os import getenv
from dotenv import load_dotenv
load_dotenv()

io_net_model_list = [
  "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
  "meta-llama/Llama-3.3-70B-Instruct",
  "mistralai/Ministral-8B-Instruct-2410",
  "google/gemma-3-27b-it",
  "mistralai/Mistral-Large-Instruct-2411",
  "microsoft/phi-4",
]

open_router_model_list = [
  "qwen/qwen3-8b:free",
  "google/gemma-3-27b-it:free",
  "qwen/qwen3-14b:free",
  "meta-llama/llama-3.3-70b-instruct:free",
  "meta-llama/llama-3.3-8b-instruct:free",
  "meta-llama/llama-3.1-8b-instruct:free"
]

lmstudio_model_list = [
    "qwen/qwen3-8b",
    "qwen3-14b-cvpr-chat-full",
    "qwen/qwen3-4b"
]

def get_llm(provider="lmstudio", model_name=None):
    """Get the LLM instance based on provider and model.
    
    Args:
        provider (str): The LLM provider to use ("lmstudio", "openrouter", or "ionet")
        model_name (str): Name of the model to use
    """
    if provider == "lmstudio":
        if not model_name or model_name not in lmstudio_model_list:
            raise ValueError(f"Invalid model name for lmstudio. Choose from: {lmstudio_model_list}")
        llm = ChatOpenAI(
            model = model_name,
            base_url = "http://127.0.0.1:1234/v1"
        )
        smol = ChatOpenAI(
            model = model_name,
            base_url = "http://127.0.0.1:1234/v1"
        )
    elif provider == "openrouter":
        if not model_name or model_name not in open_router_model_list:
            raise ValueError(f"Invalid model name for openrouter. Choose from: {open_router_model_list}")
        llm = ChatOpenAI(
            model = model_name,
            base_url = "https://openrouter.ai/api/v1",
            api_key = getenv("OPENROUTER_API_KEY")
        )
        smol = ChatOpenAI(
            model = model_name,
            base_url = "https://openrouter.ai/api/v1",
            api_key = getenv("OPENROUTER_API_KEY")
        )
    elif provider == "ionet":
        if not model_name or model_name not in io_net_model_list:
            raise ValueError(f"Invalid model name for ionet. Choose from: {io_net_model_list}")
        llm = ChatOpenAI(
            model = model_name,
            base_url = "https://api.intelligence.io.solutions/api/v1",
            api_key = getenv("OPENAI_API_KEY")
        )
        smol = ChatOpenAI(
            model = model_name,
            base_url = "https://api.intelligence.io.solutions/api/v1",
            api_key = getenv("OPENAI_API_KEY")
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return llm, smol