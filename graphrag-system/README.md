# GraphRAG System

A Knowledge Graph based Retrieval Augmented Generation system with local and global search capabilities.

## Features

- **Knowledge Graph Construction**: Extract entities and relationships from documents using LLM
- **Community Detection**: Leiden algorithm for discovering clusters of related entities
- **Local Search**: Vector similarity search over documents and entities
- **Global Search**: Search over community summaries for high-level answers
- **REST API**: FastAPI endpoints for integration

## Project Structure

```
graphrag-system/
├── data/
│   ├── input/                  # Input documents
│   ├── processed/              # Chunks, entities, embeddings
│   └── output/                 # Graph, communities, reports
├── src/
│   ├── indexing/               # Graph building pipeline
│   ├── query/                  # Search functionality
│   ├── generation/             # Answer generation
│   └── utils/                  # Utilities
├── configs/                    # Configuration files
├── scripts/                    # Pipeline scripts
├── api/                        # REST API
└── tests/                      # Test suite
```

## Installation

```bash
# Clone the repository
cd graphrag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama (for local LLM)
# macOS: brew install ollama
# Or download from https://ollama.ai

# Pull the model
ollama pull llama3.1:8b
```

## Quick Start

### 1. Prepare your data

Add your documents to `data/input/graphrag_input.jsonl`:

```jsonl
{"id": "doc_001", "text": "Your document text here...", "metadata": {"source": "example"}}
{"id": "doc_002", "text": "Another document...", "metadata": {"source": "example"}}
```

### 2. Build the knowledge graph (Run in order)

```bash
# Step 1: Build graph (extract entities & relationships)
python scripts/1_build_graph.py

# Step 2: Detect communities
python scripts/2_detect_communities.py

# Step 3: Generate community summaries
python scripts/3_generate_reports.py

# Step 4: Create embeddings
python scripts/4_create_embeddings.py
```

### 3. Run searches

```bash
# Local search (specific details)
python scripts/run_local_search.py "Who founded Apple?"

# Global search (high-level overview)
python scripts/run_global_search.py "What are the main technology companies?"
```

### 4. Start the API

```bash
uvicorn api.app:app --reload

# API docs available at http://localhost:8000/docs
```

## Configuration

Edit `configs/graphrag_config.yaml` to customize:

- Chunk size and overlap
- LLM model (default: llama3.1:8b)
- Embedding model (default: BAAI/bge-large-en-v1.5)
- Community detection resolution
- Search parameters

## API Endpoints

- `POST /api/search/` - Search with answer generation
- `GET /api/graph/stats` - Graph statistics
- `GET /api/graph/entities` - List entities
- `GET /api/graph/communities` - List communities
- `GET /api/graph/export` - Export graph as JSON

## Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Embedding | BAAI/bge-large-en-v1.5 | Best for English text |
| LLM | llama3.1:8b (Ollama) | Entity extraction, summarization, generation |
| Community | Leiden algorithm | Community detection |

## Requirements

- Python 3.9+
- 16GB RAM (for embedding model + LLM)
- Ollama installed and running

## License

MIT
