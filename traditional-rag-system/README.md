# Traditional RAG System

A baseline traditional RAG (Retrieval-Augmented Generation) system for comparing with GraphRAG performance.

## Features

- **Vector-based retrieval** using FAISS
- **Same dataset & model** as GraphRAG for fair comparison
- **Same evaluation metrics**: Relevance, Coverage, Answer Quality, Faithfulness
- **Gradio web interface** for interactive testing
- **Comprehensive evaluation scripts** for benchmarking

## Architecture

```
Query → Embedding → Vector Search (FAISS) → Top-K Chunks → LLM Generation → Answer
```

Unlike GraphRAG, this system:
- Does NOT use knowledge graph structure
- Does NOT extract entities and relationships
- Does NOT perform community detection
- Relies purely on semantic similarity search

## Setup

### Quick Setup (Recommended)

**Windows:**
```cmd
cd traditional-rag-system
setup_venv.bat
```

**Linux/Mac:**
```bash
cd traditional-rag-system
bash setup_venv.sh
```

This will automatically:
- Create virtual environment in `venv/`
- Install all dependencies (~2-3GB)
- Takes 5-10 minutes

### Verify Setup

```bash
python check_setup.py
```

This checks if everything is properly installed.

### Manual Setup (Alternative)

```bash
cd traditional-rag-system

# Create venv
python -m venv venv

# Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

### Build Index

Build FAISS index from the same dataset as GraphRAG:

```bash
python scripts/build_index.py
```

This will:
- Load data from `../graphrag-system/data/input/graphrag_format.jsonl`
- Chunk text into 800-character pieces with 150-character overlap
- Generate embeddings using `BAAI/bge-large-en-v1.5`
- Build FAISS index for fast similarity search
- Save index to `data/processed/embeddings/faiss_index.bin`

### 3. Run Web Interface

```bash
python app.py
```

Access at `http://localhost:7861`

## Evaluation

### Single System Evaluation

Evaluate Traditional RAG on a set of queries:

```bash
python scripts/evaluate.py --queries queries.json --generate-answers --output rag_results.json
```

### Compare with GraphRAG

Compare Traditional RAG with GraphRAG:

```bash
python scripts/compare_systems.py \
  --rag-results rag_results.json \
  --graphrag-local ../graphrag-system/graphrag_local_results.json \
  --graphrag-global ../graphrag-system/graphrag_global_results.json \
  --output comparison_report.md
```

## Configuration

Edit `configs/rag_config.yaml`:

```yaml
# Data
data:
  input_file: "../graphrag-system/data/input/graphrag_format.jsonl"

# Chunking (same as GraphRAG)
chunking:
  chunk_size: 800
  chunk_overlap: 150

# Embedding (same model as GraphRAG)
embedding:
  model: "BAAI/bge-large-en-v1.5"
  dimension: 1024

# LLM (same as GraphRAG)
llm:
  model: "qwen2.5:3b"
  base_url: "http://localhost:11434"

# Search
search:
  top_k: 15
  index_type: "faiss"
```

## File Structure

```
traditional-rag-system/
├── app.py                      # Gradio web interface
├── configs/
│   └── rag_config.yaml         # Configuration
├── src/
│   ├── indexing/               # Chunking, embedding, vector store
│   ├── retrieval/              # RAG retriever
│   ├── generation/             # LLM client, prompt builder
│   ├── evaluation/             # Evaluation metrics (same as GraphRAG)
│   └── utils/                  # Config, logger
├── scripts/
│   ├── build_index.py          # Build FAISS index
│   ├── evaluate.py             # Evaluate RAG
│   └── compare_systems.py      # Compare RAG vs GraphRAG
└── data/
    └── processed/              # Generated data
        ├── embeddings/         # FAISS index
        └── chunks/             # Text chunks
```

## Comparison with GraphRAG

### Advantages of Traditional RAG:
- **Faster**: Simpler architecture, no graph construction
- **Simpler**: Easier to understand and debug
- **Less storage**: Only stores embeddings, no graph

### Advantages of GraphRAG:
- **Better context**: Uses knowledge graph structure
- **Relationships**: Captures entity relationships
- **Multi-hop reasoning**: Can traverse graph for complex queries
- **Community detection**: Groups related information

### Evaluation Metrics (Same for Both):

1. **Relevance Score**: Query-result similarity
2. **Coverage Score**: Information diversity
3. **Answer Quality**: Completeness and coherence
4. **Faithfulness**: Grounding in context

## Sample Queries

Create `queries.json`:

```json
[
  {
    "query": "What is the recommended daily iron intake for adults?",
    "ground_truth": "For adult males, the recommended daily iron intake is about 8 mg per day."
  },
  {
    "query": "What are the main health risks associated with low testosterone?",
    "ground_truth": "Low testosterone can cause lower libido, decreased muscle mass, fatigue, mood changes, and slower recovery from workouts."
  }
]
```

Then run evaluation:

```bash
python scripts/evaluate.py --queries queries.json --generate-answers
```

## Usage Example

```python
from src.retrieval import RAGRetriever
from src.generation import LLMClient, PromptBuilder

# Load retriever
retriever = RAGRetriever()
retriever.load('data/processed/embeddings/faiss_index.bin',
               'data/processed/chunks/chunks.json')

# Search
results = retriever.retrieve("What is the recommended iron intake?", top_k=10)

# Generate answer
llm = LLMClient()
prompt_builder = PromptBuilder()
context = retriever.get_context(results)
prompt = prompt_builder.build_health_prompt(query, context)
answer = llm.generate(prompt)
```

## License

MIT
