"""Build prompts for LLM generation."""

from typing import List, Dict, Any
from pathlib import Path


class PromptBuilder:
    """Build prompts for different search types."""

    def __init__(self, prompts_dir: str = "configs/prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.templates = {}

    def load_templates(self):
        """Load prompt templates from files."""
        for prompt_file in self.prompts_dir.glob("*.txt"):
            name = prompt_file.stem
            with open(prompt_file) as f:
                self.templates[name] = f.read()

    def build_local_prompt(self, query: str, context: str) -> Dict[str, str]:
        """Build prompt for local search."""
        system = """You are a helpful assistant that answers questions based on the provided context.
Use the specific details from the context to provide accurate, detailed answers.
Always cite your sources using [Source N] notation.
If information is not in the context, say so clearly."""

        user = f"""Context:
{context}

Question: {query}

Provide a detailed answer based on the context above. Cite sources using [Source N] notation."""

        return {'system': system, 'user': user}

    def build_global_prompt(self, query: str, context: str) -> Dict[str, str]:
        """Build prompt for global search."""
        system = """You are a helpful assistant that provides comprehensive overviews based on community summaries.
Synthesize information from multiple communities to provide a high-level understanding.
Focus on themes, patterns, and relationships across the dataset."""

        user = f"""Community Summaries:
{context}

Question: {query}

Provide a comprehensive answer that synthesizes information from the community summaries above.
Focus on the big picture and key themes."""

        return {'system': system, 'user': user}

    def build_hybrid_prompt(self, query: str, context: str) -> Dict[str, str]:
        """Build prompt for hybrid search."""
        system = """You are a helpful assistant that answers questions using both high-level summaries and specific details.
First provide the big picture from community summaries, then support with specific details.
Cite sources appropriately."""

        user = f"""{context}

Question: {query}

Answer the question by:
1. First providing the high-level understanding from community summaries
2. Then supporting with specific details from documents and entities
3. Citing sources appropriately"""

        return {'system': system, 'user': user}

    def get_template(self, name: str) -> str:
        """Get a specific template."""
        if not self.templates:
            self.load_templates()
        return self.templates.get(name, '')
