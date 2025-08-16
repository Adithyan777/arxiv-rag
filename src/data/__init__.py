"""Data processing utilities for ArXiv papers."""

import os
import json
import requests
import xml.etree.ElementTree as ET
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging

from ..core.config import app_config

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
        self.filepath = filepath or app_config.PAPERS_JSON
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

# Global instances for backward compatibility
metadata_extractor = ArxivMetadataExtractor()
paper_database = PaperDatabase()
text_processor = TextProcessor()

# Legacy function names
get_arxiv_metadata_from_paper_id = metadata_extractor.get_arxiv_metadata_from_paper_id
add_paper_to_json = paper_database.add_paper
clean_arxiv_md_text = text_processor.clean_arxiv_md_text
