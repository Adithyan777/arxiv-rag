from pathlib import Path
from collections import Counter
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from langchain_core.documents import Document
import pandas as pd
import re
import requests
import xml.etree.ElementTree as ET

papers_csv_filename = "final_paper_list.csv"

def get_arxiv_metadata_from_paper_id(paper_id):
    # ArXiv API endpoint
    url = f'https://export.arxiv.org/api/query?id_list={paper_id}'
    
    try:
        # Make the request
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        
        # Find the entry element (contains paper details)
        entry = root.find('{http://www.w3.org/2005/Atom}entry')
        
        if entry is None:
            return {"error": "No paper found with given ID"}
            
        # Extract namespace for arxiv-specific elements
        ns = {'atom': 'http://www.w3.org/2005/Atom',
              'arxiv': 'http://arxiv.org/schemas/atom'}
        
        # Extract authors
        authors = [author.find('{http://www.w3.org/2005/Atom}name').text 
                  for author in entry.findall('{http://www.w3.org/2005/Atom}author')]
        
        # Extract other metadata
        metadata = {
            "title": ' '.join(entry.find('{http://www.w3.org/2005/Atom}title').text.strip().split()),
            "authors": authors,
            "abstract": ' '.join(entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().split()),
            "created": entry.find('{http://www.w3.org/2005/Atom}published').text.split('T')[0],  # Extract date part only
            "updated": entry.find('{http://www.w3.org/2005/Atom}updated').text.split('T')[0],  # Extract date part only
            "id": str(paper_id),
            "pdf_url": next(link.get('href') for link in entry.findall('{http://www.w3.org/2005/Atom}link') 
                          if link.get('title') == 'pdf'),
            "categories": entry.find('.//{http://arxiv.org/schemas/atom}primary_category').get('term'),
        }
        
        # Optional fields
        doi = entry.find('.//{http://arxiv.org/schemas/atom}doi')
        if doi is not None:
            metadata["doi"] = ' '.join(doi.text.strip().split())
            
        journal_ref = entry.find('.//{http://arxiv.org/schemas/atom}journal_ref')
        if journal_ref is not None:
            metadata["journal_ref"] = ' '.join(journal_ref.text.strip().split())
            
        return metadata
        
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except ET.ParseError as e:
        return {"error": f"XML parsing failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def extract_metadata(filename):
    # Extract ID from filename (assumes format: "id title")
    id_match = re.match(r'(\d{4}\.\d+)', filename)
    if not id_match:
        raise ValueError("Invalid filename format. Expected format: 'XXXX.XXXXX Title'")
    
    arxiv_id = id_match.group(1)
    
    try:
        df = pd.read_csv(papers_csv_filename)

        # print(f"Extracting metadata for ID: {arxiv_id}")
        # print(f"Total papers in dataset: {len(df)}")

        # Convert ID column to string and remove any whitespace
        df['id'] = df['id'].astype(str).str.strip()
        paper_data = df[df['id'].str.contains(arxiv_id, regex=False)]
        
        if paper_data.empty:
            raise ValueError(f"No metadata found for ID: {arxiv_id}")
        
        # Initialize empty metadata dictionary
        metadata = {}
        
        # Define columns to check
        columns = ['id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors']
        
        # Only add non-empty values to metadata
        for col in columns:
            value = paper_data[col].iloc[0]
            # Check if value is not empty/null/NaN
            if pd.notna(value) and str(value).strip():
                metadata[col] = value
        
        return metadata
        
    except FileNotFoundError:
        raise FileNotFoundError(f"{papers_csv_filename} not found")

def merge_same_heading_docs(docs):
    if not docs:
        return []

    merged_docs = []
    current_heading = None
    current_doc = None

    for doc in docs:
        # Check if doc is None
        if doc is None:
            print("Warning: Found None document, skipping...")
            continue
            
        # Check if doc has required attributes
        if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
            print(f"Warning: Document missing required attributes, skipping...")
            continue
            
        # Get heading from the correct nested location
        dl_meta = doc.metadata.get("dl_meta")
        if dl_meta and dl_meta.get('headings'):
            heading = dl_meta.get('headings')[0]
        else:
            heading = None
        
        # If this is a new heading, start a new merged document
        if heading != current_heading:
            if current_doc is not None:
                merged_docs.append(current_doc)
            
            # Create a copy of the document instead of using reference
            current_doc = type(doc)(
                page_content=doc.page_content,
                metadata=doc.metadata.copy()
            )
            current_heading = heading
        else:
            # For same heading, merge content while handling the heading prefix
            content = doc.page_content
            if content and "\n" in content:
                # Remove heading prefix if present
                content = content.split("\n", 1)[1]
            
            # Only append if content exists
            if content:
                current_doc.page_content += "\n" + content

    # Add the last document
    if current_doc is not None:
        merged_docs.append(current_doc)

    return merged_docs

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
    print(f"Predicted Paper ID for re-written queries: {predicted_paper_id}")

    return str(predicted_paper_id), queries

def get_context_for_qa(paper_id: str, rewritten_queries: List[str], vector_store, k : int = 3) -> List[models.Record]:
    """Get context for QA from the vector store based on paper ID and search query."""
    results = []
    for query in rewritten_queries:
        # Perform similarity search with filter for the specific paper ID
        individual_results = vector_store.similarity_search_with_score(
            query, 
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
            if results:
                for doc, score in individual_results:
                    if not any(doc.page_content == existing_doc.page_content for existing_doc, score in results):
                        results.append((doc, score))
            else:
                results = individual_results
    
    return results

def get_context_for_qa_without_id(rewritten_queries: List[str], vector_store, k : int = 3) -> List[models.Record]:
    """Get context for QA from the vector store based on paper ID and search query."""
    results = []
    for query in rewritten_queries:
        # Perform similarity search with filter for the specific paper ID
        individual_results = vector_store.similarity_search_with_score(
            query, 
            k=k, 
            score_threshold=0.4,
        )
        if individual_results:
            if results:
                for doc, score in individual_results:
                    if not any(doc.page_content == existing_doc.page_content for existing_doc, score in results):
                        results.append((doc, score))
            else:
                results = individual_results
    
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