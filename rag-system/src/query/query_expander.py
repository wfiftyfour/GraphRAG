"""Query expander for improving query coverage."""

from typing import List


class QueryExpander:
    """Expand queries with synonyms and related terms."""

    def __init__(self):
        self.synonyms = {}

    def expand(self, query: str) -> List[str]:
        """Expand a query into multiple variations."""
        expanded = [query]

        # Simple word-level expansion
        words = query.lower().split()
        for word in words:
            if word in self.synonyms:
                for synonym in self.synonyms[word]:
                    expanded_query = query.lower().replace(word, synonym)
                    expanded.append(expanded_query)

        return list(set(expanded))

    def add_synonyms(self, word: str, synonyms: List[str]):
        """Add synonyms for a word."""
        self.synonyms[word.lower()] = [s.lower() for s in synonyms]

    def expand_with_llm(self, query: str) -> List[str]:
        """Expand query using LLM (placeholder for future implementation)."""
        # This would call an LLM to generate query variations
        return [query]
