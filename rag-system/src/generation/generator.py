"""Response generator using LLM."""

import requests
from typing import List, Dict, Any, Optional
from .prompt_builder import PromptBuilder


class ResponseGenerator:
    """Generate responses using LLM (OpenAI or Ollama)."""

    def __init__(self, model_name: str = "llama3.2", provider: str = "ollama"):
        self.model_name = model_name
        self.provider = provider  # "openai" or "ollama"
        self.prompt_builder = PromptBuilder()
        self.client = None
        self.ollama_base_url = "http://localhost:11434"

    def setup_openai(self, api_key: str):
        """Setup OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.provider = "openai"
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

    def setup_ollama(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        """Setup Ollama for local LLM inference (FREE, no quota)."""
        self.ollama_base_url = base_url
        self.model_name = model
        self.provider = "ollama"

        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags")
            if response.status_code != 200:
                raise ConnectionError(f"Cannot connect to Ollama at {base_url}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Ollama is not running at {base_url}. Start it with: ollama serve")

    def generate(self, query: str, contexts: List[Dict[str, Any]],
                 max_tokens: int = 500) -> str:
        """Generate a response."""
        prompts = self.prompt_builder.build_with_system(query, contexts)

        if self.provider == "ollama":
            return self._generate_ollama(prompts, max_tokens)
        else:
            return self._generate_openai(prompts, max_tokens)

    def _generate_openai(self, prompts: Dict[str, str], max_tokens: int) -> str:
        """Generate using OpenAI API."""
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Call setup_openai() first.")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompts['system']},
                {"role": "user", "content": prompts['user']}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )

        return response.choices[0].message.content

    def _generate_ollama(self, prompts: Dict[str, str], max_tokens: int) -> str:
        """Generate using Ollama (local, FREE)."""
        url = f"{self.ollama_base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompts['system']},
                {"role": "user", "content": prompts['user']}
            ],
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['message']['content']
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Ollama is not running. Start it with: ollama serve")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def generate_stream(self, query: str, contexts: List[Dict[str, Any]]):
        """Generate a streaming response."""
        prompts = self.prompt_builder.build_with_system(query, contexts)

        if self.provider == "ollama":
            yield from self._stream_ollama(prompts)
        else:
            yield from self._stream_openai(prompts)

    def _stream_openai(self, prompts: Dict[str, str]):
        """Stream using OpenAI API."""
        if self.client is None:
            raise ValueError("OpenAI client not initialized. Call setup_openai() first.")

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": prompts['system']},
                {"role": "user", "content": prompts['user']}
            ],
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def _stream_ollama(self, prompts: Dict[str, str]):
        """Stream using Ollama (local, FREE)."""
        url = f"{self.ollama_base_url}/api/chat"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": prompts['system']},
                {"role": "user", "content": prompts['user']}
            ],
            "stream": True
        }

        try:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        yield data['message']['content']
        except requests.exceptions.ConnectionError:
            raise ConnectionError("Ollama is not running. Start it with: ollama serve")
