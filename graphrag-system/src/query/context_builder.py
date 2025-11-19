"""Build context from search results for LLM generation."""

from typing import List, Dict, Any


class ContextBuilder:
    """Build context from search results."""

    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens

    def build_local_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context from local search results."""
        context_parts = []

        for i, result in enumerate(results):
            if result['type'] == 'chunk':
                context_parts.append(f"[Source {i+1}]\n{result['content']}")

            elif result['type'] == 'entity':
                entity_context = f"[Entity: {result['metadata'].get('name', '')}]\n"
                entity_context += f"{result['content']}\n"

                # Add graph context
                if 'graph_context' in result:
                    gc = result['graph_context']
                    if gc.get('relationships'):
                        entity_context += "Related to:\n"
                        for rel in gc['relationships'][:3]:
                            entity_context += f"- {rel['neighbor']} ({rel['relationship']})\n"

                context_parts.append(entity_context)

        context = "\n\n".join(context_parts)

        # Truncate if too long (rough estimate: 4 chars per token)
        max_chars = self.max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Context truncated...]"

        return context

    def build_global_context(self, results: List[Dict[str, Any]]) -> str:
        """Build context from global search results."""
        context_parts = []

        for i, result in enumerate(results):
            community_context = f"[Community {i+1}: {result.get('title', '')}]\n"
            community_context += f"{result.get('summary', '')}\n"
            community_context += f"(Contains {result.get('num_entities', 0)} entities)"

            context_parts.append(community_context)

        context = "\n\n".join(context_parts)

        # Truncate if needed
        max_chars = self.max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Context truncated...]"

        return context

    def build_hybrid_context(self, local_results: List[Dict[str, Any]],
                             global_results: List[Dict[str, Any]]) -> str:
        """Build context from both local and global results."""
        parts = []

        # Global context first (high-level)
        if global_results:
            parts.append("## High-Level Context (Communities)")
            parts.append(self.build_global_context(global_results[:3]))

        # Local context (specific details)
        if local_results:
            parts.append("\n## Specific Details (Documents & Entities)")
            parts.append(self.build_local_context(local_results[:5]))

        return "\n\n".join(parts)

    def format_sources(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Format sources for citation."""
        sources = []

        for i, result in enumerate(results):
            if result['type'] == 'chunk':
                sources.append({
                    'id': str(i + 1),
                    'type': 'document',
                    'content': result['content'][:200] + '...'
                })
            elif result['type'] == 'entity':
                sources.append({
                    'id': str(i + 1),
                    'type': 'entity',
                    'content': result['content']
                })
            elif result['type'] == 'community':
                sources.append({
                    'id': str(i + 1),
                    'type': 'community',
                    'content': result.get('title', '')
                })

        return sources
