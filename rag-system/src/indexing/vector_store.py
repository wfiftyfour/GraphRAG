"""Vector store for embeddings."""

from typing import List, Dict, Any, Optional
import numpy as np
from pathlib import Path


class VectorStore:
    """Store and retrieve vector embeddings."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors = []
        self.metadata = []

    def add(self, vectors: np.ndarray, metadata: List[Dict[str, Any]]):
        """Add vectors to the store."""
        self.vectors.extend(vectors)
        self.metadata.extend(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if not self.vectors:
            return []

        vectors_array = np.array(self.vectors)
        similarities = np.dot(vectors_array, query_vector) / (
            np.linalg.norm(vectors_array, axis=1) * np.linalg.norm(query_vector)
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'metadata': self.metadata[idx],
                'score': float(similarities[idx])
            })

        return results

    def save(self, path: str):
        """Save the vector store to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        np.save(save_path / 'vectors.npy', np.array(self.vectors))

        import json
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(self.metadata, f)

    def load(self, path: str):
        """Load the vector store from disk."""
        load_path = Path(path)

        self.vectors = np.load(load_path / 'vectors.npy').tolist()

        import json
        with open(load_path / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
