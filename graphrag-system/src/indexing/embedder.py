"""Text embedding utilities."""

from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import json


class TextEmbedder:
    """Generate embeddings for text chunks and entities."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5",
                 output_dir: str = "data/processed/embeddings"):
        self.model_name = model_name
        self.model = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.model is None:
            self.load_model()

        # BGE models perform better with normalization
        if "bge" in self.model_name.lower():
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True
            )
        else:
            embeddings = self.model.encode(texts, show_progress_bar=True)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        if self.model is None:
            self.load_model()

        # BGE models need instruction prefix for queries
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True if "bge" in self.model_name.lower() else False
        )
        return embedding[0]

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks."""
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embed(texts)

        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()

        return chunks

    def embed_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for entities."""
        texts = [f"{e['name']}: {e.get('description', '')}" for e in entities]
        embeddings = self.embed(texts)

        for i, entity in enumerate(entities):
            entity['embedding'] = embeddings[i].tolist()

        return entities

    def embed_communities(self, reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for community reports."""
        texts = [f"{r['title']}\n{r['summary']}" for r in reports]
        embeddings = self.embed(texts)

        for i, report in enumerate(reports):
            report['embedding'] = embeddings[i].tolist()

        return reports

    def save_embeddings(self, data: List[Dict[str, Any]], name: str):
        """Save embeddings to file."""
        # Extract embeddings as numpy array
        embeddings = np.array([item['embedding'] for item in data])
        np.save(self.output_dir / f"{name}_embeddings.npy", embeddings)

        # Save metadata (without embeddings)
        metadata = []
        for item in data:
            meta = {k: v for k, v in item.items() if k != 'embedding'}
            metadata.append(meta)

        with open(self.output_dir / f"{name}_metadata.json", 'w') as f:
            json.dump(metadata, f)

        print(f"Saved {len(data)} {name} embeddings")

    def load_embeddings(self, name: str):
        """Load embeddings from file."""
        embeddings = np.load(self.output_dir / f"{name}_embeddings.npy")

        with open(self.output_dir / f"{name}_metadata.json") as f:
            metadata = json.load(f)

        # Combine
        for i, meta in enumerate(metadata):
            meta['embedding'] = embeddings[i].tolist()

        return metadata
