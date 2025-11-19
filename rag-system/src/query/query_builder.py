"""Query builder for constructing search queries."""

from typing import Dict, Any, Optional


class QueryBuilder:
    """Build structured queries from user input."""

    def __init__(self):
        self.filters = {}
        self.options = {}

    def build(self, query_text: str, filters: Optional[Dict[str, Any]] = None,
              options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build a structured query."""
        query = {
            'text': query_text,
            'filters': filters or {},
            'options': {
                'top_k': 10,
                'use_reranker': True,
                'search_type': 'hybrid',
                **(options or {})
            }
        }
        return query

    def with_filter(self, key: str, value: Any) -> 'QueryBuilder':
        """Add a filter to the query."""
        self.filters[key] = value
        return self

    def with_option(self, key: str, value: Any) -> 'QueryBuilder':
        """Add an option to the query."""
        self.options[key] = value
        return self

    def reset(self):
        """Reset filters and options."""
        self.filters = {}
        self.options = {}
