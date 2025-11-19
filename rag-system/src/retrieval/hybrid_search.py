"""Hybrid search combining local and global search."""

from typing import List, Dict, Any
from .local_search import LocalSearch
from .global_search import GlobalSearch


class HybridSearch:
    """Combine local and global search results."""

    def __init__(self, local_weight: float = 0.5, global_weight: float = 0.5):
        self.local_search = LocalSearch()
        self.global_search = GlobalSearch()
        self.local_weight = local_weight
        self.global_weight = global_weight

    def load(self):
        """Load both search indexes."""
        self.local_search.load()
        self.global_search.load()

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search."""
        local_results = self.local_search.search(query, top_k)
        global_results = self.global_search.search(query, top_k)

        # Combine and reweight results
        combined = []

        for result in local_results:
            result['score'] *= self.local_weight
            result['source'] = 'local'
            combined.append(result)

        for result in global_results:
            result['score'] *= self.global_weight
            result['source'] = 'global'
            combined.append(result)

        # Sort by score
        combined.sort(key=lambda x: x['score'], reverse=True)

        return combined[:top_k]
