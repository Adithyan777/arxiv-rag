"""Data processing utilities for ArXiv papers."""

import os
import json
import requests
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging
from langchain_core.documents import Document

# from ..core.config import app_config

logger = logging.getLogger(__name__)

class ArxivMetadataExtractor:
    """Handles ArXiv metadata extraction from API."""
    
    @staticmethod
    def get_arxiv_metadata_from_paper_id(paper_id: str) -> Dict:
        """
        Extract metadata for a paper from ArXiv API.
        
        Args:
            paper_id (str): ArXiv paper ID (e.g., "2401.12345")
            
        Returns:
            Dict: Paper metadata including title, authors, abstract, etc.
        """
        url = f'https://export.arxiv.org/api/query?id_list={paper_id}'
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            entry = root.find('{http://www.w3.org/2005/Atom}entry')
            
            if entry is None:
                raise ValueError(f"No paper found with ID: {paper_id}")
            
            # Extract authors
            authors = [
                author.find('{http://www.w3.org/2005/Atom}name').text 
                for author in entry.findall('{http://www.w3.org/2005/Atom}author')
            ]
            
            # Extract metadata
            metadata = {
                "title": ' '.join(entry.find('{http://www.w3.org/2005/Atom}title').text.strip().split()),
                "authors": str([author.lower() for author in authors]),
                "abstract": ' '.join(entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().split()),
                "created": entry.find('{http://www.w3.org/2005/Atom}published').text.split('T')[0],
                "updated": entry.find('{http://www.w3.org/2005/Atom}updated').text.split('T')[0],
                "id": str(paper_id),
                "categories": entry.find('.//{http://arxiv.org/schemas/atom}primary_category').get('term'),
            }
            
            # Optional fields
            doi = entry.find('.//{http://arxiv.org/schemas/atom}doi')
            if doi is not None:
                metadata["doi"] = ' '.join(doi.text.strip().split())
                
            journal_ref = entry.find('.//{http://arxiv.org/schemas/atom}journal_ref')
            if journal_ref is not None:
                metadata["journal_ref"] = ' '.join(journal_ref.text.strip().split())
                
            logger.info(f"Successfully extracted metadata for paper: {paper_id}")
            return metadata
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for paper {paper_id}: {str(e)}")
            raise
        except ET.ParseError as e:
            logger.error(f"XML parsing failed for paper {paper_id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error for paper {paper_id}: {str(e)}")
            raise

class PaperDatabase:
    """Manages the paper metadata database."""
    
    def __init__(self, filepath: str = None):
        self.filepath = filepath or "final_papers.json"  # Default filename
        self._ensure_absolute_path()
    
    def _ensure_absolute_path(self):
        """Ensure we have an absolute path to the papers file."""
        if not os.path.isabs(self.filepath):
            # Try project root first
            project_root = Path(__file__).parent.parent.parent
            abs_path = project_root / self.filepath
            if abs_path.exists():
                self.filepath = str(abs_path)
            else:
                # Fallback to current working directory
                self.filepath = os.path.join(os.getcwd(), self.filepath)
    
    def load_papers(self) -> Dict[str, str]:
        """Load papers and return title+id -> id mapping."""
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            papers_list = data if isinstance(data, list) else data.get('papers', [])
            return {f"{paper['title']} ({paper['id']})": paper['id'] 
                   for paper in papers_list}
        except FileNotFoundError:
            logger.warning(f"Papers file not found: {self.filepath}")
            return {}
        except Exception as e:
            logger.error(f"Error loading papers: {str(e)}")
            return {}
    
    def get_paper_metadata(self, paper_id: str) -> Dict:
        """Get metadata for a specific paper."""
        if not paper_id:
            raise ValueError("Paper ID cannot be empty")
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            
            papers_list = data if isinstance(data, list) else data.get('papers', [])
            paper = next((p for p in papers_list if p['id'] == paper_id), None)
            
            if not paper:
                raise ValueError(f"No metadata found for ID: {paper_id}")
            
            return {k: v for k, v in paper.items() if v}
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Papers file not found: {self.filepath}")
    
    def add_paper(self, paper_metadata: Dict) -> None:
        """Add a new paper to the database."""
        required_fields = ['id', 'title', 'abstract', 'authors', 'created']
        missing_fields = [field for field in required_fields if not paper_metadata.get(field)]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
        
        try:
            # Read existing data
            try:
                with open(self.filepath, 'r') as f:
                    data = json.load(f)
            except FileNotFoundError:
                data = {"papers": []}
            
            papers_list = data.get('papers', [])
            
            # Check if paper already exists
            if any(p.get('id') == paper_metadata['id'] for p in papers_list):
                raise ValueError(f"Paper with ID {paper_metadata['id']} already exists")
            
            # Add new paper
            papers_list.append(paper_metadata)
            data['papers'] = papers_list
            
            # Write back to file
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Added paper {paper_metadata['id']} to database")
            
        except Exception as e:
            logger.error(f"Failed to add paper to database: {str(e)}")
            raise

class MetadataExtractor:
    """Handles metadata extraction from filenames and database lookups."""
    
    def __init__(self, json_file: str = "final_papers.json"):
        self.json_file = json_file
    
    def extract_metadata(self, filename: str) -> Dict:
        """
        Extract metadata for a paper from its filename using a JSON database.
        
        Args:
            filename (str): Filename containing paper ID (format: "XXXX.XXXXX Title")
            
        Returns:
            Dict: Paper metadata
            
        Raises:
            ValueError: If filename format is invalid or paper not found
            FileNotFoundError: If JSON file not found
        """
        # Extract ID from filename
        id_match = re.match(r'(\d{4}\.\d+)', filename)
        if not id_match:
            raise ValueError("Invalid filename format. Expected format: 'XXXX.XXXXX Title'")
        
        arxiv_id = id_match.group(1)
        
        try:
            # Get project root directory
            project_root = Path(__file__).parent.parent.parent
            abs_path = project_root / self.json_file
            
            if not abs_path.exists():
                abs_path = Path.cwd() / self.json_file
                
            with open(abs_path, 'r') as f:
                data = json.load(f)
            
            # Handle both possible JSON structures
            papers_list = data if isinstance(data, list) else data.get('papers', [])
            
            # Find paper with matching ID
            paper = next((p for p in papers_list if p.get('id') == arxiv_id), None)
            
            if not paper:
                raise ValueError(f"No metadata found for ID: {arxiv_id}")
                
            # Define columns to check
            columns = ['id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors']
            
            # Only add non-empty values to metadata
            metadata = {
                col: paper.get(col) 
                for col in columns 
                if paper.get(col) and str(paper.get(col)).strip()
            }
            
            return metadata
            
        except FileNotFoundError:
            logger.error(f"Papers list not found at: {abs_path}")
            raise

class TextProcessor:
    """Handles text cleaning and processing."""
    
    @staticmethod
    def clean_arxiv_md_text(text: str) -> str:
        """Clean academic text by removing citations, references, etc."""
        # Remove citation patterns like [1], [2], [3,4], etc.
        text = re.sub(r'\[[\d,\s-]+\]', '', text)
        
        # Remove inline citations like (Author, Year) or (Author et al., Year)
        text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
        
        # Remove mathematical expressions in parentheses with numbers
        text = re.sub(r'\([^)]*\d+[^)]*\)', '', text)
        
        # Remove references section entirely
        text = re.sub(r'## References.*$', '', text, flags=re.DOTALL | re.MULTILINE)
        text = re.sub(r'# References.*$', '', text, flags=re.DOTALL | re.MULTILINE)
        
        # Remove acknowledgments section
        text = re.sub(r'## Acknowledgments.*$', '', text, flags=re.DOTALL | re.MULTILINE)
        text = re.sub(r'# Acknowledgments.*$', '', text, flags=re.DOTALL | re.MULTILINE)
        
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove mathematical notation patterns
        text = re.sub(r'<!-- formula-not-decoded -->', '', text)
        text = re.sub(r'<!-- image -->', '', text)
        
        # Remove author affiliations and institutional info
        text = re.sub(r'\d+\s*Department of[^,]*,?[^,]*,?[^,]*', '', text)
        text = re.sub(r'\*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
        
        # Remove extra spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove lines that are just numbers or contain only special characters
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines, lines with only numbers, or lines with only special chars
            if (line and 
                not re.match(r'^[\d\s\-\.\,\(\)]+$', line) and
                not re.match(r'^[^\w\s]*$', line) and
                len(line) > 2):
                cleaned_lines.append(line)
        
        # Join lines back together
        text = '\n'.join(cleaned_lines)
        
        # Final cleanup - remove any remaining isolated citations or references
        text = re.sub(r'\n\s*\[\d+\].*\n', '\n', text)
        text = re.sub(r'^- \[.*?\].*$', '', text, flags=re.MULTILINE)
        
        return text

class DocumentProcessor:
    """Handles document processing and merging."""
    
    @staticmethod
    def merge_same_heading_docs(docs: List[Document]) -> List[Document]:
        """
        Merge documents with the same heading to reduce redundancy.
        
        Args:
            docs (List[Document]): List of documents to merge
            
        Returns:
            List[Document]: Merged documents
        """
        if not docs:
            return []

        merged_docs = []
        current_heading = None
        current_doc = None

        for doc in docs:
            # Check if doc is None
            if doc is None:
                logger.warning("Found None document, skipping...")
                continue
                
            # Check if doc has required attributes
            if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
                logger.warning("Document missing required attributes, skipping...")
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

# Global instances for backward compatibility
metadata_extractor = ArxivMetadataExtractor()
paper_database = PaperDatabase()
text_processor = TextProcessor()
file_metadata_extractor = MetadataExtractor()
document_processor = DocumentProcessor()

# Legacy function names
get_arxiv_metadata_from_paper_id = metadata_extractor.get_arxiv_metadata_from_paper_id
add_paper_to_json = paper_database.add_paper
clean_arxiv_md_text = text_processor.clean_arxiv_md_text
extract_metadata = file_metadata_extractor.extract_metadata
merge_same_heading_docs = document_processor.merge_same_heading_docs
