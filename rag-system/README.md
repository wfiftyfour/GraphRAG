# GraphRAG System

A Retrieval Augmented Generation system with local and global search capabilities.

## Project Structure

```
rag-system/
├── data/                    # Data storage
├── src/                     # Source code
│   ├── ingestion/          # Document loading and processing
│   ├── indexing/           # Vector indexing
│   ├── retrieval/          # Search functionality
│   ├── query/              # Query processing
│   ├── generation/         # Response generation
│   └── utils/              # Utilities
├── tests/                   # Test suite
├── configs/                 # Configuration files
├── scripts/                 # CLI scripts
└── api/                     # REST API
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Add your documents to `data/graphrag_input.jsonl`
2. Build the index:
   ```bash
   python scripts/build_index.py
   ```
3. Run a search:
   ```bash
   python scripts/run_search.py "your query here"
   ```

## API

Start the API server:
```bash
uvicorn api.app:app --reload
```

## Configuration

- `configs/model_config.yaml` - Model settings
- `configs/index_config.yaml` - Indexing settings
- `configs/search_config.yaml` - Search settings

## License

MIT
