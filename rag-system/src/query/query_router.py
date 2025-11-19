"""Query router for selecting search strategy."""

from typing import Dict, Any


class QueryRouter:
    """Route queries to appropriate search strategies."""

    def __init__(self):
        self.thresholds = {
            'local_keywords': ['specific', 'detail', 'example', 'how to'],
            'global_keywords': ['overview', 'summary', 'general', 'what is', 'explain']
        }

    def route(self, query: str) -> str:
        """Determine the best search strategy for a query."""
        query_lower = query.lower()

        local_score = sum(1 for kw in self.thresholds['local_keywords'] if kw in query_lower)
        global_score = sum(1 for kw in self.thresholds['global_keywords'] if kw in query_lower)

        if local_score > global_score:
            return 'local'
        elif global_score > local_score:
            return 'global'
        else:
            return 'hybrid'

    def get_search_config(self, query: str) -> Dict[str, Any]:
        """Get search configuration based on query routing."""
        search_type = self.route(query)

        configs = {
            'local': {'search_type': 'local', 'top_k': 10, 'use_reranker': True},
            'global': {'search_type': 'global', 'top_k': 5, 'use_reranker': False},
            'hybrid': {'search_type': 'hybrid', 'top_k': 10, 'use_reranker': True}
        }

        return configs[search_type]
