"""Prompt building utilities."""


class PromptBuilder:
    """Build prompts for RAG answer generation."""

    @staticmethod
    def build_rag_prompt(query: str, context: str) -> str:
        """
        Build RAG prompt from query and context.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.

Context Information:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the information provided in the context above
- Be concise and informative
- If the context doesn't contain enough information to answer the question fully, acknowledge this
- Cite relevant parts of the context when possible

Answer:"""

        return prompt

    @staticmethod
    def build_health_prompt(query: str, context: str) -> str:
        """
        Build health-specific RAG prompt.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        prompt = f"""You are a knowledgeable health assistant. Answer the health-related question using the provided context.

Health Information Context:
{context}

Question: {query}

Instructions:
- Base your answer on the medical/health information provided in the context
- Be accurate and informative
- Use clear, understandable language
- If the context lacks sufficient information, state this clearly
- Mention that users should consult healthcare professionals for personalized advice

Answer:"""

        return prompt
