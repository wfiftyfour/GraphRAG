"""Local indexer for document-level search."""

from typing import List, Dict, Any
from .vector_store import VectorStore


class LocalIndexer:
    """Index documents for local search."""

    def __init__(self, index_path: str = "data/indexes/local"):
        self.index_path = index_path
        self.vector_store = VectorStore()

    def build_index(self, chunks: List[Dict[str, Any]]):
        """Build local index from chunks."""
        vectors = [chunk['embedding'] for chunk in chunks]
        metadata = [{
            'chunk_id': chunk['chunk_id'],
            'doc_id': chunk['doc_id'],
            'text': chunk['text'],
            'metadata': chunk['metadata']
        } for chunk in chunks]

        self.vector_store.add(vectors, metadata)

    def save(self):
        """Save the index."""
        self.vector_store.save(self.index_path)

    def load(self):
        """Load the index."""
        self.vector_store.load(self.index_path)

    def search(self, query_vector, top_k: int = 10):
        """Search the local index."""
        return self.vector_store.search(query_vector, top_k)
