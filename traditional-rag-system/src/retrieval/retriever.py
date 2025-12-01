"""RAG retrieval component."""

from typing import List, Dict, Any
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from indexing import TextEmbedder, VectorStore


class RAGRetriever:
    """Traditional RAG retriever using vector search."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize retriever.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.embedder = None
        self.vector_store = None
        self.loaded = False

    def load(self, index_path: str, chunks_path: str, embedding_model: str = "BAAI/bge-large-en-v1.5"):
        """
        Load retriever components.

        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks metadata
            embedding_model: Embedding model name
        """
        print("Loading RAG retriever...")

        # Load embedder
        print(f"Loading embedding model: {embedding_model}")
        self.embedder = TextEmbedder(model_name=embedding_model)
        self.embedder.load_model()

        # Load vector store
        print(f"Loading vector index from: {index_path}")
        self.vector_store = VectorStore(dimension=self.embedder.embedding_dim)
        self.vector_store.load(index_path, chunks_path)

        self.loaded = True
        print("RAG retriever loaded successfully!")

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with scores
        """
        if not self.loaded:
            raise ValueError("Retriever not loaded. Call load() first.")

        # Embed query
        query_embedding = self.embedder.embed_query(query)

        # Search
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results

    def get_context(self, results: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        """
        Build context from retrieved results.

        Args:
            results: Retrieved results
            max_tokens: Maximum context tokens (approx chars / 4)

        Returns:
            Context string
        """
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough estimate

        for i, result in enumerate(results, 1):
            text = result['text']
            if total_chars + len(text) > max_chars:
                break

            context_parts.append(f"[{i}] {text}")
            total_chars += len(text)

        return '\n\n'.join(context_parts)

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        if not self.loaded:
            return {'status': 'not loaded'}

        return {
            'status': 'loaded',
            'embedding_model': self.embedder.model_name,
            'embedding_dim': self.embedder.embedding_dim,
            'vector_store': self.vector_store.get_stats()
        }
