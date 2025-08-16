# ArXiv RAG System 📚

A comprehensive Retrieval Augmented Generation (RAG) system designed specifically for ArXiv research papers, pre-loaded with CVPR (Computer Vision and Pattern Recognition) papers and enhanced with fine-tuned models for research assistance.

## 🚀 Features

- **Advanced RAG Pipeline**: Built with LangChain and Qdrant for efficient document retrieval
- **Multiple LLM Providers**: Support for LMStudio, OpenRouter, and IO.NET
- **Fine-tuned Models**: Custom models trained on CVPR papers for domain-specific responses
- **Interactive Web UI**: Streamlit-based interface for easy interaction
- **Specialized Agents**: Compare papers, generate summaries, and perform targeted analysis
- **Vector Search**: Semantic search across abstracts and full paper content
- **Automatic Paper Processing**: Add new papers via ArXiv links with automatic processing

## 🏗️ Architecture

```
├── Docker Services (Qdrant + Ollama)
├── Embedding Layer (Nomic Embed Text)
├── Vector Stores (Abstracts + Papers)
├── LLM Layer (Multiple Providers)
├── RAG Pipeline (LangChain)
└── Streamlit UI
```

## 🤖 Fine-tuned Models

Two specialized models have been fine-tuned on CVPR papers using synthetic data generation and QLoRA:

- **[adithyn/qwen3-8b-4bit-cvpr-lora-full2.0](https://huggingface.co/adithyn/qwen3-8b-4bit-cvpr-lora-full2.0)** - 8B parameter model with 4-bit quantization
- **[adithyn/qwen3-14b-cvpr-chat-full](https://huggingface.co/adithyn/qwen3-14b-cvpr-chat-full)** - 14B parameter model

## 🛠️ Setup & Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.12+
- 8GB+ RAM (16GB recommended)
- CUDA GPU (optional, for faster embedding)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/arxiv-rag.git
cd arxiv-rag
```

### 2. Start Docker Services

```bash
# Start Qdrant and Ollama services
docker-compose up -d

# Wait for services to initialize (especially embedding model download)
# This may take 5-10 minutes on first run
```

The docker-compose will:
- Start Qdrant vector database on port 6333
- Start Ollama service and download `nomic-embed-text:latest` model

### 3. Install Python Dependencies

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file with your API keys:

```env
# For OpenRouter provider
OPENROUTER_API_KEY=your_openrouter_api_key

# For IO.NET provider  
OPENAI_API_KEY=your_ionet_api_key
```

## 📊 Usage

### Option A: Using Pre-loaded CVPR Papers

If you want to use the already processed CVPR papers:

#### 1. Load Abstract Collection
```bash
python init_vector_stores/abstracts.py
```

#### 2. Load Full Papers to Vector Store
```bash
python init_vector_stores/papers.py
```

### Option B: Starting Fresh

#### 1. Create Vector Store Collections
Make sure Docker services are running, then create the collections:
'arxiv-abstracts' for abstracts and 'arxiv-cvpr-main' for full papers inside the qdrant vector database.

#### 2. Add Papers via UI
- Start the application
- Use the "Add Research Paper" section
- Provide ArXiv PDF links
- Papers will be automatically processed and added to vector store

### 3. Start the Application

```bash
# From project root
streamlit run src/streamlit_ui/app.py
```

Access the application at `http://localhost:8501`

## 🖥️ Interface Overview

### 1. RAG Q&A
- Ask research questions about papers
- **Global Context**: Search across all papers
- **Specific Paper**: Target individual papers (manual selection or auto-routing)
- Advanced filtering and metadata display

### 2. Add Research Paper
- Input ArXiv PDF links
- Automatic processing and vector store integration
- Metadata extraction and storage

### 3. Agents (WIP⌛️)
- **Compare Agent**: Side-by-side comparison of papers (methodology, results, architecture)
- **Summarize Agent**: Generate focused summaries with optional areas of emphasis

## 🔧 Configuration

### LLM Providers

**LMStudio (Local)**
- Run models locally
- Supports fine-tuned CVPR models
- Base URL: `http://127.0.0.1:1234/v1`

**OpenRouter (Cloud)**
- Access to various open-source models
- Requires API key
- Free tier available for some models

**IO.NET (Cloud)**
- High-performance inference
- Requires API key
- Access to latest models

### Vector Store Collections

- **arxiv-abstracts**: Paper abstracts for quick overview searches
- **arxiv-cvpr-main**: Full paper content for detailed analysis

## 📁 Project Structure

```
arxiv-rag/
├── docker-compose.yaml         # Docker services configuration
├── pyproject.toml             # Python dependencies
├── README.md                  # This file
├── utils.py                   # Utility functions
├── final_papers.json          # Paper metadata storage
├── data/                      # Processed data
│   ├── markdown/             # Converted papers in Markdown
│   ├── papers-pdf/           # Original PDF files
│   └── qa_datasets/          # Generated Q&A datasets
├── engine/                    # Core RAG components
│   ├── embedding.py          # Embedding configuration
│   └── llm.py               # LLM providers setup
├── init_vector_stores/        # Vector store initialization
│   ├── abstracts.py         # Load abstracts
│   └── papers.py            # Load full papers
├── src/                      # Main application code
│   ├── streamlit_ui/        # Web interface
│   └── data/                # Data processing
└── scraping/                 # Paper collection tools
    ├── download.py          # PDF download
    └── find_papers.py       # Paper discovery
```

## 🔍 Key Features Explained

### Intelligent Paper Routing
The system can automatically identify the most relevant paper for your question, or you can manually select specific papers for targeted queries.

### Multi-Modal Search
- **Abstract Search**: Quick overview and paper discovery
- **Full Content Search**: Deep dive into methodologies and results
- **Metadata Filtering**: Search by authors, categories, publication dates

### Specialized Agents
- **Comparison**: Analyze multiple papers across different dimensions
- **Summarization**: Generate focused summaries with customizable emphasis areas

### Automatic Processing Pipeline
1. **PDF Download**: Fetch papers from ArXiv
2. **Content Extraction**: Convert to structured markdown
3. **Chunking**: Split into optimal sizes for retrieval
4. **Embedding**: Generate semantic embeddings
5. **Storage**: Index in Qdrant vector database

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 Roadmap

- [ ] Add fine-tuning code using synthetic-data-kit
- [ ] Include unsloth QLoRA training scripts
- [ ] Implement Agents

## 🙏 Acknowledgments

- **LangChain**: For the RAG framework
- **Qdrant**: For vector database capabilities
- **Ollama**: For local embedding models
- **Streamlit**: For the web interface
- **Unsloth**: For efficient fine-tuning
- **Synthetic Data Kit**: For training data generation

---