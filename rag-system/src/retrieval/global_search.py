"""Global search for corpus-level retrieval."""

from typing import List, Dict, Any
from ..indexing import GlobalIndexer
from ..ingestion import TextEmbedder


class GlobalSearch:
    """Perform global search over communities/summaries."""

    def __init__(self, index_path: str = "data/indexes/global"):
        self.indexer = GlobalIndexer(index_path)
        self.embedder = TextEmbedder()

    def load(self):
        """Load the index and embedder."""
        self.indexer.load()
        self.embedder.load_model()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant communities/summaries."""
        query_embedding = self.embedder.embed([query])[0]
        results = self.indexer.search(query_embedding, top_k)
        return results
