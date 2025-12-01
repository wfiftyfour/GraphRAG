# RAG vs GraphRAG Comparison Guide

This guide helps you compare Traditional RAG with GraphRAG using the same dataset and evaluation metrics.

## Project Structure

```
GraphRAG/
├── graphrag-system/              # GraphRAG implementation
│   ├── app.py                    # GraphRAG UI (port 7860)
│   ├── data/input/               # Shared dataset
│   └── src/evaluation/           # Evaluation metrics
│
└── traditional-rag-system/       # Traditional RAG baseline
    ├── app.py                    # RAG UI (port 7861)
    ├── scripts/                  # Evaluation scripts
    └── configs/                  # Configuration
```

## Quick Start

### 1. Setup GraphRAG (Already Done)

GraphRAG should already be set up with:
- Data indexed and processed
- Knowledge graph built
- Community detection completed

### 2. Setup Traditional RAG

```bash
cd traditional-rag-system

# Install dependencies
pip install -r requirements.txt

# Build FAISS index from same dataset
python scripts/build_index.py

# Start web interface
python app.py  # Runs on port 7861
```

### 3. Compare Both Systems

#### Option A: Visual Comparison (Web UI)

Run both UIs side-by-side:

```bash
# Terminal 1: GraphRAG
cd graphrag-system
python app.py  # http://localhost:7860

# Terminal 2: Traditional RAG
cd traditional-rag-system
python app.py  # http://localhost:7861
```

Then manually test the same queries on both systems and compare results.

#### Option B: Automated Evaluation

```bash
cd traditional-rag-system

# Evaluate both systems on same queries
bash run_full_evaluation.sh

# View comparison report
cat comparison_report.md
```

## Evaluation Metrics

Both systems use identical evaluation metrics for fair comparison:

| Metric | Description | Range |
|--------|-------------|-------|
| **Relevance Score** | How well results match the query | 0.0 - 1.0 |
| **Coverage Score** | Diversity of retrieved information | 0.0 - 1.0 |
| **Answer Quality** | Completeness and coherence | 0.0 - 1.0 |
| **Faithfulness** | Grounding in context | 0.0 - 1.0 |
| **Overall Score** | Average of all metrics | 0.0 - 1.0 |

## Key Differences

### Traditional RAG

**Architecture:**
```
Query → Embedding → FAISS Search → Top-K Chunks → LLM → Answer
```

**Pros:**
- Faster retrieval (simple vector search)
- Simpler architecture
- Lower storage requirements
- Easy to understand and debug

**Cons:**
- No understanding of relationships
- Limited context (just similar chunks)
- No entity or topic awareness
- Can't perform multi-hop reasoning

### GraphRAG

**Architecture:**
```
Query → Embedding → Graph Search → Entities + Relationships + Communities → LLM → Answer
```

**Pros:**
- Rich context with relationships
- Multi-hop reasoning capability
- Entity and topic awareness
- Community-based global search
- Better for complex queries

**Cons:**
- Slower (graph construction + search)
- More complex architecture
- Higher storage requirements
- Requires entity extraction

## Use Cases Comparison

### When to Use Traditional RAG:
- Simple fact-finding queries
- When speed is critical
- Resource-constrained environments
- Well-scoped questions with direct answers

### When to Use GraphRAG:
- Complex queries requiring connections
- Multi-entity questions
- Topic exploration
- When understanding context is critical

## Sample Queries for Testing

### Simple Fact-Finding (RAG should do well)
- "What is the recommended daily iron intake?"
- "What is a normal testosterone level?"

### Relationship Queries (GraphRAG should excel)
- "How do testosterone levels relate to muscle health and recovery?"
- "What factors connect age, calcium, and bone density?"

### Complex Multi-Concept (GraphRAG advantage)
- "Explain the relationship between diet, hormones, and overall health"
- "What health factors are interconnected for men over 40?"

## Running Evaluations

### Evaluate Traditional RAG Only

```bash
cd traditional-rag-system

python scripts/evaluate.py \
  --queries queries.json \
  --generate-answers \
  --output rag_results.json
```

### Evaluate GraphRAG Only

```bash
cd graphrag-system

# Local search
python scripts/evaluate.py \
  --queries ../traditional-rag-system/queries.json \
  --generate-answers \
  --search-type local \
  --output graphrag_local_results.json

# Global search
python scripts/evaluate.py \
  --queries ../traditional-rag-system/queries.json \
  --generate-answers \
  --search-type global \
  --output graphrag_global_results.json
```

### Compare Results

```bash
cd traditional-rag-system

python scripts/compare_systems.py \
  --rag-results rag_results.json \
  --graphrag-local ../graphrag-system/graphrag_local_results.json \
  --graphrag-global ../graphrag-system/graphrag_global_results.json \
  --output comparison_report.md
```

## Understanding Results

The comparison report shows:

1. **Summary Statistics**: Overall performance comparison
2. **Metric-by-Metric**: Detailed breakdown per evaluation metric
3. **Winner Per Metric**: Which system performs best for each metric
4. **Key Observations**: Analysis of strengths and weaknesses

Example interpretation:

```
| Metric            | RAG    | GraphRAG Local | GraphRAG Global |
|-------------------|--------|----------------|-----------------|
| Relevance Score   | 0.8234 | 0.8567        | 0.7891         |
| Coverage Score    | 0.6789 | 0.8123        | 0.8934         |
| Answer Quality    | 0.7456 | 0.7823        | 0.8234         |
```

**Analysis:**
- RAG: Good relevance, lower coverage (finds similar text but limited diversity)
- GraphRAG Local: Best relevance + good coverage (entities + relationships)
- GraphRAG Global: Best coverage (community-level information)

## Configuration

Both systems use the same:
- **Dataset**: `graphrag-system/data/input/graphrag_format.jsonl`
- **Embedding Model**: `BAAI/bge-large-en-v1.5`
- **LLM**: `qwen2.5:3b` (Ollama)
- **Chunking**: 800 chars, 150 overlap
- **Evaluation Metrics**: Same implementation

## Troubleshooting

### Port Already in Use

If port 7860 or 7861 is taken:

```python
# In app.py, change:
app.launch(server_port=7862)  # Use different port
```

### FAISS Index Not Found

```bash
cd traditional-rag-system
python scripts/build_index.py --force
```

### LLM Connection Error

Make sure Ollama is running:

```bash
ollama serve
ollama pull qwen2.5:3b
```

### Out of Memory

Reduce batch size in config:

```yaml
embedding:
  batch_size: 8  # Reduce from 16
```

## Next Steps

1. **Run Evaluation**: Compare both systems on your queries
2. **Analyze Results**: Understand which works better for your use case
3. **Tune Parameters**: Adjust top-k, temperature, etc.
4. **Hybrid Approach**: Consider combining both (future work)

## Future: Hybrid RAG

Potential hybrid approach combining strengths:
- Use Traditional RAG for simple queries (speed)
- Use GraphRAG for complex queries (context)
- Query router to decide which system to use
- Combine results from both systems

---

**Questions?** Check the individual README files in each system folder.
