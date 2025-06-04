import dotenv
from os import getenv

dotenv.load_dotenv()


# from llama_index.core import SimpleDirectoryReader
# from llama_index.core import VectorStoreIndex

# # SPLITTING PDF INTO DOCUMENTS USING LLAMA_INDEX

# documents = SimpleDirectoryReader("data").load_data()
# for doc in documents[:5]:
#     print(doc.get_content())
#     print("-" * 80)

# -----------------------------------------USING AGNO LIBRARY------------------------------------------

import asyncio

from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.qdrant import Qdrant
from agno.models.openai.like import OpenAILike
from agno.models.ollama import Ollama
from agno.models.lmstudio import LMStudio
from agno.embedder.ollama import OllamaEmbedder
from agno.utils.pprint import pprint_run_response

from qdrant_client import QdrantClient, models


COLLECTION_NAME = "arxiv-reader"

# client = QdrantClient(url="http://localhost:6333")

# client.create_collection(
#     collection_name=COLLECTION_NAME,
#     vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
# )

vector_db = Qdrant(
    collection=COLLECTION_NAME, 
    url="http://localhost:6333", 
    embedder=OllamaEmbedder(id="nomic-embed-text:latest",host="http://localhost:11434")
)

# # Create a knowledge base with the PDFs from the data/pdfs directory
# knowledge_base = PDFKnowledgeBase(
#     path="data/2505.0038.pdf",
#     vector_db=vector_db,
#     reader=PDFReader(chunk=True),
# )

# # iterate and print first 3 documents from document lists
# for docs in knowledge_base.document_lists:
#     for doc in docs[:3]:
#         print(doc)
#         print("-" * 80)

# Create an agent with the knowledge base

agent = Agent(
    debug_mode=True,
    knowledge=knowledge_base,
    search_knowledge=True,
    model=OpenAILike(
        id = "mistralai/Ministral-8B-Instruct-2410",
        base_url= "https://api.intelligence.io.solutions/api/v1",
        api_key=getenv("OPENAI_API_KEY")
    )
#     model=LMStudio(
#         id="llama-3.2-3b-instruct"
#     )
)

response = agent.run("What is the main contribution of this paper?")
pprint_run_response(response, markdown=True)

# ---------------------------------------------USING LANGCHAIN INSTEAD----------------------------------------------

