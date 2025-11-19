# How to Run GraphRAG System

## Prerequisites

### 1. Install Python 3.9+
```bash
python --version  # Should be 3.9 or higher
```

### 2. Install Ollama (Local LLM - FREE)
```bash
# macOS
brew install ollama

# Or download from https://ollama.ai
```

### 3. Start Ollama and Pull Model
```bash
# Start Ollama server (keep this running)
ollama serve

# In another terminal, pull the model
ollama pull llama3.1:8b
```

---

## Installation

```bash
# Navigate to project
cd graphrag-system

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Step-by-Step Pipeline

### Step 0: Prepare Your Data

Add your documents to `data/input/graphrag_input.jsonl`:

```jsonl
{"id": "doc_001", "text": "Your document text here...", "metadata": {"source": "example"}}
{"id": "doc_002", "text": "Another document...", "metadata": {"source": "example"}}
```

### Step 1: Build Knowledge Graph
```bash
python scripts/1_build_graph.py
```
**What it does:**
- Loads documents from `graphrag_input.jsonl`
- Chunks text into smaller pieces
- Extracts entities (people, organizations, locations, etc.)
- Extracts relationships between entities
- Builds and saves the knowledge graph

**Output:** `data/output/graph/`

### Step 2: Detect Communities
```bash
python scripts/2_detect_communities.py
```
**What it does:**
- Loads the knowledge graph
- Runs Leiden algorithm to find communities
- Groups related entities together

**Output:** `data/output/communities/`

### Step 3: Generate Community Reports
```bash
python scripts/3_generate_reports.py
```
**What it does:**
- Generates summaries for each community
- Creates high-level descriptions of entity clusters

**Output:** `data/output/reports/`

### Step 4: Create Embeddings
```bash
python scripts/4_create_embeddings.py
```
**What it does:**
- Creates vector embeddings for chunks
- Creates embeddings for entities
- Creates embeddings for community reports

**Output:** `data/processed/embeddings/`

---

## Running Searches

### Local Search (Specific Details)
```bash
python scripts/run_local_search.py "Who founded Apple?"
python scripts/run_local_search.py "What is Microsoft Office?"
python scripts/run_local_search.py "Where is Apple headquartered?"
```

### Global Search (High-Level Overview)
```bash
python scripts/run_global_search.py "What are the main technology companies?"
python scripts/run_global_search.py "Overview of company founders"
python scripts/run_global_search.py "Summary of tech industry"
```

### Search Options
```bash
# Without answer generation (just retrieval)
python scripts/run_local_search.py "query" --no-generate

# Customize number of results
python scripts/run_local_search.py "query" --top-k 20
```

---

## Running the API

### Start the Server
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

### API Endpoints

**Swagger Docs:** http://localhost:8000/docs

**Search:**
```bash
curl -X POST "http://localhost:8000/api/search/" \
  -H "Content-Type: application/json" \
  -d '{"query": "Who founded Apple?", "search_type": "local"}'
```

**Graph Stats:**
```bash
curl "http://localhost:8000/api/graph/stats"
```

**List Entities:**
```bash
curl "http://localhost:8000/api/graph/entities?limit=50"
```

**List Communities:**
```bash
curl "http://localhost:8000/api/graph/communities"
```

---

## Quick Start (All Commands)

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run pipeline
cd graphrag-system
source venv/bin/activate

# Build (run once)
python scripts/1_build_graph.py
python scripts/2_detect_communities.py
python scripts/3_generate_reports.py
python scripts/4_create_embeddings.py

# Search
python scripts/run_local_search.py "Your question here"

# Or start API
uvicorn api.app:app --reload
```

---

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running
ollama serve
```

### "Model not found"
```bash
# Pull the model first
ollama pull llama3.1:8b
```

### "No module named 'src'"
```bash
# Make sure you're in the graphrag-system directory
cd graphrag-system
```

### "leidenalg not found"
```bash
pip install leidenalg python-igraph
```

### Memory Issues
- Use smaller model: Edit `configs/graphrag_config.yaml`, change `model: "llama3.1:8b"` to `model: "llama3.2:3b"`
- Reduce chunk size in config

---

## Configuration

Edit `configs/graphrag_config.yaml` to customize:

```yaml
# Change LLM model
llm:
  model: "llama3.1:8b"  # or "mistral:7b", "gemma2:9b"

# Adjust chunking
chunking:
  chunk_size: 1200
  chunk_overlap: 100

# Search settings
search:
  local:
    top_k: 10
```

---

## File Locations

| What | Where |
|------|-------|
| Input data | `data/input/graphrag_input.jsonl` |
| Knowledge graph | `data/output/graph/graph.graphml` |
| Communities | `data/output/communities/` |
| Embeddings | `data/processed/embeddings/` |
| Config | `configs/graphrag_config.yaml` |
| Prompts | `configs/prompts/` |
