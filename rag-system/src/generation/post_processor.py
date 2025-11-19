"""Post processor for generated responses."""

import re
from typing import Dict, Any, List


class PostProcessor:
    """Post-process generated responses."""

    def __init__(self):
        self.citation_pattern = r'\[(\d+)\]'

    def process(self, response: str, contexts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process the generated response."""
        # Extract citations
        citations = self.extract_citations(response)

        # Clean response
        cleaned = self.clean_response(response)

        # Add source references
        sources = self.get_sources(citations, contexts)

        return {
            'response': cleaned,
            'citations': citations,
            'sources': sources
        }

    def extract_citations(self, response: str) -> List[int]:
        """Extract citation numbers from response."""
        matches = re.findall(self.citation_pattern, response)
        return [int(m) for m in matches]

    def clean_response(self, response: str) -> str:
        """Clean up the response text."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', response).strip()
        return cleaned

    def get_sources(self, citations: List[int], contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get source information for citations."""
        sources = []
        for cite_num in citations:
            if 1 <= cite_num <= len(contexts):
                ctx = contexts[cite_num - 1]
                sources.append({
                    'citation': cite_num,
                    'doc_id': ctx['metadata'].get('doc_id', ''),
                    'text': ctx['metadata'].get('text', '')[:200]
                })
        return sources
