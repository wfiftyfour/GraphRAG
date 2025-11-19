"""Process and classify queries."""

from typing import Dict, Any, Tuple
from ..indexing import TextEmbedder
import numpy as np


class QueryProcessor:
    """Process queries and determine search strategy."""

    def __init__(self):
        self.embedder = TextEmbedder()
        self.local_keywords = [
            'who', 'what is', 'where', 'when', 'specific', 'detail',
            'example', 'how does', 'define', 'describe'
        ]
        self.global_keywords = [
            'overview', 'summary', 'general', 'main themes',
            'compare', 'contrast', 'relationship between',
            'what are all', 'list all', 'categorize'
        ]

    def process(self, query: str) -> Dict[str, Any]:
        """Process a query and return embedding and metadata."""
        # Generate embedding
        embedding = self.embedder.embed_query(query)

        # Classify query type
        query_type = self.classify(query)

        return {
            'query': query,
            'embedding': embedding,
            'type': query_type,
            'search_strategy': self._get_strategy(query_type)
        }

    def classify(self, query: str) -> str:
        """Classify query as local or global."""
        query_lower = query.lower()

        local_score = sum(1 for kw in self.local_keywords if kw in query_lower)
        global_score = sum(1 for kw in self.global_keywords if kw in query_lower)

        if global_score > local_score:
            return 'global'
        elif local_score > global_score:
            return 'local'
        else:
            # Default based on query length
            if len(query.split()) > 10:
                return 'global'
            return 'local'

    def _get_strategy(self, query_type: str) -> Dict[str, Any]:
        """Get search strategy based on query type."""
        strategies = {
            'local': {
                'use_local': True,
                'use_global': False,
                'local_top_k': 10,
                'global_top_k': 0,
                'use_graph_context': True
            },
            'global': {
                'use_local': False,
                'use_global': True,
                'local_top_k': 0,
                'global_top_k': 10,
                'use_graph_context': False
            },
            'hybrid': {
                'use_local': True,
                'use_global': True,
                'local_top_k': 5,
                'global_top_k': 5,
                'use_graph_context': True
            }
        }

        return strategies.get(query_type, strategies['local'])
