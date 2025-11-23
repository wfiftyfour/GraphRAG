"""LLM client for text generation."""

import requests
import time
from typing import Dict, Any, Generator


class LLMClient:
    """Client for LLM API calls (Gemini or Ollama)."""

    def __init__(self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434", provider: str = "ollama"):
        self.provider = provider

        # Ollama config - lightweight model for RTX 3050 8GB
        self.ollama_model = model if provider == "ollama" else "qwen2.5:3b"  # 3B model - fast, low VRAM (~4GB)
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"

    def generate(self, prompt: Dict[str, str], max_tokens: int = 1024,
                 temperature: float = 0.3) -> str:
        """Generate response from LLM."""
        if self.provider == "gemini":
            return self._generate_gemini(prompt, max_tokens, temperature)
        else:
            return self._generate_ollama(prompt, max_tokens, temperature)

    def _generate_gemini(self, prompt: Dict[str, str], max_tokens: int, temperature: float) -> str:
        """Generate response from Gemini."""
        url = f"{self.gemini_url}/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        full_prompt = f"System: {prompt['system']}\n\nUser: {prompt['user']}"
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            time.sleep(2)  # 30 RPM limit
            return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            if "429" in str(e):
                time.sleep(60)
                return self._generate_gemini(prompt, max_tokens, temperature)
            raise RuntimeError(f"Gemini generation failed: {e}")

    def _generate_ollama(self, prompt: Dict[str, str], max_tokens: int, temperature: float) -> str:
        """Generate response from Ollama."""
        messages = [
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": prompt['user']}
        ]

        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json()['message']['content']
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}. Run: ollama serve")
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {e}")

    def generate_stream(self, prompt: Dict[str, str], max_tokens: int = 1024,
                        temperature: float = 0.3) -> Generator[str, None, None]:
        """Generate streaming response from LLM."""
        messages = [
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": prompt['user']}
        ]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    if 'message' in data and 'content' in data['message']:
                        yield data['message']['content']
        except requests.exceptions.ConnectionError:
            raise ConnectionError(f"Cannot connect to Ollama at {self.base_url}. Run: ollama serve")

    def check_connection(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> list:
        """List available models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return [m['name'] for m in response.json().get('models', [])]
        except:
            pass
        return []
