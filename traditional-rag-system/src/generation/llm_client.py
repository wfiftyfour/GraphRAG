"""LLM client for answer generation."""

import requests
from typing import Dict, Any


class LLMClient:
    """Client for Ollama LLM."""

    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"):
        """
        Initialize LLM client.

        Args:
            model: Model name
            base_url: Ollama base URL
        """
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"

    def generate(self, prompt: str, temperature: float = 0.4, max_tokens: int = 1536) -> str:
        """
        Generate answer from prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(self.generate_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()

        except requests.exceptions.RequestException as e:
            print(f"LLM generation error: {e}")
            return f"Error generating answer: {str(e)}"

    def chat(self, messages: list, temperature: float = 0.4, max_tokens: int = 1536) -> str:
        """
        Chat completion (alternative interface).

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text
        """
        # Convert to single prompt for Ollama
        prompt_parts = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            prompt_parts.append(f"{role.upper()}: {content}")

        prompt = '\n\n'.join(prompt_parts)
        return self.generate(prompt, temperature, max_tokens)
