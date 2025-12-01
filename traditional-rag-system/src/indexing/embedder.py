"""Text embedding utilities."""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch


class TextEmbedder:
    """Generate embeddings using SentenceTransformer."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", batch_size: int = 16):
        """
        Initialize embedder.

        Args:
            model_name: Name of embedding model
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load embedding model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"Model loaded on {self.device}")

    def embed(self, texts: Union[str, List[str]], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: Single text or list of texts
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        if self.model is None:
            self.load_model()

        if isinstance(texts, str):
            texts = [texts]

        # Encode with batch processing
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query.

        Args:
            query: Query text

        Returns:
            Query embedding
        """
        return self.embed(query, show_progress=False)[0]

    def embed_chunks(self, chunks: List[dict], text_field: str = 'text', show_progress: bool = True) -> np.ndarray:
        """
        Embed text chunks.

        Args:
            chunks: List of chunk dicts
            text_field: Field name containing text
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        texts = [chunk[text_field] for chunk in chunks]
        return self.embed(texts, show_progress=show_progress)

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self.model is None:
            self.load_model()
        return self.model.get_sentence_embedding_dimension()
