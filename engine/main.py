from llm import get_llm
from embedding import get_vector_store, get_embeddings, get_qdrant_client
from utils import get_paper_id_from_search_query, get_context_for_qa, get_context_for_qa_without_id, get_llm_generation_using_context

llm, smol = get_llm("openrouter")
vector_store = get_vector_store()
embeddings = get_embeddings()
client = get_qdrant_client()

search_query = "How is visual hallucination still an issue in LVLMs?"

paper_id, rewritten_queries = get_paper_id_from_search_query(
    search_query,
    "arxiv-cvpr-main",
    embeddings,
    client,
    smol
)

results = get_context_for_qa(search_query,paper_id, rewritten_queries, vector_store)
context_using_paper_id = [res for res, score in results if res.page_content and len(res.page_content) > 0]

print(f"Response using context from paper ID {paper_id}:")

response = get_llm_generation_using_context(
    context=context_using_paper_id,
    question=search_query,
    llm=llm
)
print(response)
print("\n\n")
if ( "i don't" in response.lower() or "i do not" in response.lower() ):
    print("No context found for the paper ID, trying without paper ID...")
    results_without_id = get_context_for_qa_without_id(search_query, rewritten_queries, vector_store)
    context_without_paper_id = [res for res, score in results_without_id if res.page_content and len(res.page_content) > 0]
    
    print(f"Response using context without paper ID:")
    response_without_id = get_llm_generation_using_context(
        context=context_without_paper_id,
        search_query=search_query,
        llm=llm
    )
    print(response_without_id)