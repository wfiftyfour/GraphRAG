"""Text embedding utilities."""

from typing import List, Dict, Any
import numpy as np


class TextEmbedder:
    """Generate embeddings for text chunks."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model = None

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

        # BGE models perform better with instruction prefix for queries
        if "bge" in self.model_name.lower():
            # For document embedding, no prefix needed
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True  # Important for BGE
            )
        else:
            embeddings = self.model.encode(texts, show_progress_bar=True)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query (with instruction prefix for BGE)."""
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
        texts = [chunk.get('processed_text', chunk['text']) for chunk in chunks]
        embeddings = self.embed(texts)

        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()

        return chunks
