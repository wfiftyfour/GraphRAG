"""Load documents from graphrag_input.jsonl"""

import json
from typing import List, Dict, Any
from pathlib import Path


class DocumentLoader:
    """Load documents from JSONL input file."""

    def __init__(self, input_path: str = "data/graphrag_input.jsonl"):
        self.input_path = Path(input_path)

    def load(self) -> List[Dict[str, Any]]:
        """Load all documents from the input file."""
        documents = []

        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    documents.append(doc)

        return documents

    def load_batch(self, batch_size: int = 100):
        """Load documents in batches."""
        batch = []

        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    batch.append(json.loads(line))

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

        if batch:
            yield batch
