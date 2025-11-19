"""Local search for document-level retrieval."""

from typing import List, Dict, Any
from ..indexing import LocalIndexer
from ..ingestion import TextEmbedder


class LocalSearch:
    """Perform local search over indexed documents."""

    def __init__(self, index_path: str = "data/indexes/local"):
        self.indexer = LocalIndexer(index_path)
        self.embedder = TextEmbedder()

    def load(self):
        """Load the index and embedder."""
        self.indexer.load()
        self.embedder.load_model()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant documents."""
        query_embedding = self.embedder.embed([query])[0]
        results = self.indexer.search(query_embedding, top_k)
        return results
