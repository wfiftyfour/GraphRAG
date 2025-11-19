"""Prompt builder for LLM generation."""

from typing import List, Dict, Any


class PromptBuilder:
    """Build prompts for LLM generation."""

    def __init__(self):
        self.system_template = """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer. If you cannot find the answer in the context, say so."""

        self.user_template = """Context:
{context}

Question: {question}

Answer:"""

    def build(self, query: str, contexts: List[Dict[str, Any]]) -> str:
        """Build a prompt from query and contexts."""
        context_text = "\n\n".join([
            f"[{i+1}] {ctx['metadata']['text']}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = self.user_template.format(
            context=context_text,
            question=query
        )

        return prompt

    def build_with_system(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, str]:
        """Build a prompt with system message."""
        user_prompt = self.build(query, contexts)

        return {
            'system': self.system_template,
            'user': user_prompt
        }

    def set_system_template(self, template: str):
        """Set custom system template."""
        self.system_template = template

    def set_user_template(self, template: str):
        """Set custom user template."""
        self.user_template = template
