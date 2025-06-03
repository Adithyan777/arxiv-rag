from pathlib import Path
from collections import Counter
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from langchain_core.documents import Document


def get_rewritten_queries(question: str, llm) -> List[str]:
    """Generate multiple versions of the input question using an LLM."""
    multi_query_template = PromptTemplate.from_template("""You are an AI language model assistant. Your task is to generate three 
    different versions of the given user question to retrieve relevant documents from a vector 
    database of research papers in the field Computer Vision and Pattern Recognition. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide ONLY the alternative questions separated by newlines. Original question: {question}""")
    
    multi_query_chain = multi_query_template | llm
    # TODO: make a custom parser to handle <think> tags in the response
    queries = multi_query_chain.invoke({"question": question}).content.split('\n')
    
    # Clean up queries and add original question
    queries = [q.strip() for q in queries if q.strip()]
    queries.append(question)
    return queries

def get_top_paper_id(queries: List[str], vector_store) -> str:
    """Get the most frequent paper ID from multiple queries."""
    all_results = []
    
    for query in queries:
        results = vector_store.similarity_search(
            query,
            k=1,
            score_threshold=0.4
        )
        if results:
            all_results.append(results[0].metadata['id'])

    print(f"All results: {all_results}")
    # Return most common paper ID if we have results, else None
    if all_results:
        return Counter(all_results).most_common(1)[0][0]
    return None

def get_paper_id_from_search_query(search_query: str, abstracts_vector_store_collection_name, embeddings, vectordb_client, smol) -> str:
    """Get the paper ID from a search query using query reconstruction and similarity search."""

    abstract_vector_store = QdrantVectorStore(
        client=vectordb_client,
        collection_name=abstracts_vector_store_collection_name,
        embedding=embeddings,
    )
    
    # print(f"\nOriginal Question: {search_query}")
    queries = get_rewritten_queries(search_query, smol)
    # print("Rewritten queries:")
    for i, q in enumerate(queries, 1):
        print(f"{i}. {q}")
    
    # Get most frequent paper ID from all queries
    predicted_paper_id = get_top_paper_id(queries, abstract_vector_store)
    # print(f"Predicted Paper ID for re-written queries: {predicted_paper_id}")

    return predicted_paper_id, queries

def get_context_for_qa(search_query: str ,paper_id: str, rewritten_queries: List[str], vector_store, k : int = 3) -> List[models.Record]:
    """Get context for QA from the vector store based on paper ID and search query."""
    results = []
    for query in rewritten_queries:
        # Perform similarity search with filter for the specific paper ID
        individual_results = vector_store.similarity_search_with_score(
            search_query, 
            k=k, 
            score_threshold=0.4,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.id",
                        match=models.MatchValue(value=str(paper_id))
                    )
                ]
            )
        )
        if individual_results:
            for doc in individual_results:
                if doc not in results:
                    results.append(doc)
    
    return results

def get_context_for_qa_without_id(search_query: str ,rewritten_queries: List[str], vector_store, k : int = 3) -> List[models.Record]:
    """Get context for QA from the vector store based on paper ID and search query."""
    results = []
    for query in rewritten_queries:
        # Perform similarity search with filter for the specific paper ID
        individual_results = vector_store.similarity_search_with_score(
            search_query, 
            k=k, 
            score_threshold=0.4,
        )
        if individual_results:
            for doc in individual_results:
                if doc not in results:
                    results.append(doc)
    
    return results

def get_llm_generation_using_context(
    context: List[Document], 
    question: str, 
    llm,
) -> str:
    """Generate an answer using the LLM based on the provided context and question."""
    
    prompt = PromptTemplate.from_template("""You are an expert in CVPR topics and help students to learn by answering questions solely based on the provided context which are taken from research papers in arxiv.

    Focus on explaining concepts in detail and substantiate answers with relevant context from the given information.

    # Steps

    1. **Identify Key Concepts**: Upon receiving a question, pinpoint the core topics within CVPR relevant to the inquiry.
    2. **Contextual Analysis**: Thoroughly review the provided context to gather accurate and pertinent information specific to the question.
    3. **Detailed Explanation**: Craft a comprehensive explanation, incorporating key details and any relevant examples that illuminate the concept.
    4. **Clarification and Depth**: Ensure the response is clear, well-substantiated, and sufficiently detailed to aid student understanding.

    # Output Format

    - Provide a paragraph elaborating the concept or answering the inquiry.
    - Ensure clarity and depth, utilizing examples if applicable.

    # Notes

    - Always derive the response solely from the given context.
    - Ensure terminologies and technical details are accurately explained within the framework of the provided context.
                                        
    Context: {context}
    Question: {question}
    Answer the question based on the context provided above. If the context is not sufficient, say "I don't know" or "I don't have enough information to answer this question." Do not make up answers or provide information not present in the context.                                      
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        prompt | llm 
    )

    response = rag_chain.invoke({"question": question, "context": format_docs(context)})

    #TODO: make a custom parser to strip everything bw the <think> tags in the response
    # if "<think>" in response.content:
    #     start = response.content.index("<think>") + len("<think>")
    #     end = response.content.index("</think>")
    #     response.content = response.content[start:end].strip()

    return response.content