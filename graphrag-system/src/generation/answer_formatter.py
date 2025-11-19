"""Format and post-process generated answers."""

import re
from typing import List, Dict, Any


class AnswerFormatter:
    """Format generated answers with citations and sources."""

    def format(self, answer: str, sources: List[Dict[str, str]]) -> Dict[str, Any]:
        """Format answer with sources."""
        # Extract citations from answer
        citations = self._extract_citations(answer)

        # Clean up answer
        cleaned_answer = self._clean_answer(answer)

        # Map citations to sources
        cited_sources = self._get_cited_sources(citations, sources)

        return {
            'answer': cleaned_answer,
            'citations': citations,
            'sources': cited_sources,
            'all_sources': sources
        }

    def _extract_citations(self, text: str) -> List[int]:
        """Extract citation numbers from text."""
        pattern = r'\[(?:Source\s*)?(\d+)\]'
        matches = re.findall(pattern, text)
        return sorted(set(int(m) for m in matches))

    def _clean_answer(self, answer: str) -> str:
        """Clean up the answer text."""
        # Remove extra whitespace
        answer = re.sub(r'\s+', ' ', answer).strip()

        # Normalize citation format
        answer = re.sub(r'\[Source\s*(\d+)\]', r'[\1]', answer)

        return answer

    def _get_cited_sources(self, citations: List[int],
                          sources: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Get sources that were cited."""
        cited = []
        for cite_num in citations:
            for source in sources:
                if source['id'] == str(cite_num):
                    cited.append(source)
                    break
        return cited

    def format_markdown(self, result: Dict[str, Any]) -> str:
        """Format result as markdown."""
        md = f"## Answer\n\n{result['answer']}\n\n"

        if result['citations']:
            md += "## Sources\n\n"
            for source in result['sources']:
                md += f"**[{source['id']}]** ({source['type']})\n"
                md += f"{source['content']}\n\n"

        return md

    def format_json(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format result as clean JSON."""
        return {
            'answer': result['answer'],
            'num_citations': len(result['citations']),
            'sources': [
                {
                    'id': s['id'],
                    'type': s['type'],
                    'excerpt': s['content'][:200]
                }
                for s in result['sources']
            ]
        }
