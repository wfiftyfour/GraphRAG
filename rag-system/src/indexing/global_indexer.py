"""Global indexer for corpus-level search."""

from typing import List, Dict, Any
from .vector_store import VectorStore


class GlobalIndexer:
    """Index documents for global/community search."""

    def __init__(self, index_path: str = "data/indexes/global"):
        self.index_path = index_path
        self.vector_store = VectorStore()
        self.communities = []

    def build_index(self, chunks: List[Dict[str, Any]], communities: List[Dict[str, Any]] = None):
        """Build global index from chunks and communities."""
        if communities:
            self.communities = communities
            # Index community summaries
            vectors = [c['embedding'] for c in communities]
            metadata = communities
        else:
            # Aggregate by document
            vectors = [chunk['embedding'] for chunk in chunks]
            metadata = [{
                'chunk_id': chunk['chunk_id'],
                'doc_id': chunk['doc_id'],
                'text': chunk['text']
            } for chunk in chunks]

        self.vector_store.add(vectors, metadata)

    def save(self):
        """Save the index."""
        self.vector_store.save(self.index_path)

    def load(self):
        """Load the index."""
        self.vector_store.load(self.index_path)

    def search(self, query_vector, top_k: int = 10):
        """Search the global index."""
        return self.vector_store.search(query_vector, top_k)
