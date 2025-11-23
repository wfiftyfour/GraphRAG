"""Extract entities from text using LLM."""

import json
import requests
import time
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class EntityExtractor:
    """Extract named entities from text chunks using LLM."""

    def __init__(self, llm_client=None, provider="ollama"):
        self.llm_client = llm_client
        self.provider = provider

        # Ollama config - lightweight model for RTX 3050 8GB
        self.ollama_url = "http://localhost:11434/api/chat"
        self.ollama_model = "qwen2.5:3b"  # 3B model - fast, low VRAM (~4GB)

    def extract(self, chunk: Dict[str, Any], prompt_template: str = None) -> List[Dict[str, Any]]:
        """Extract entities from a single chunk."""
        text = chunk.get('text', '')
        chunk_id = chunk.get('chunk_id', '')

        prompt = prompt_template or self._default_prompt()
        full_prompt = prompt.format(text=text)

        # Call LLM
        response = self._call_llm(full_prompt)

        # Parse entities from response
        entities = self._parse_entities(response, chunk_id)

        return entities

    def extract_batch(self, chunks: List[Dict[str, Any]], prompt_template: str = None,
                       max_workers: int = 1) -> List[Dict[str, Any]]:
        """Extract entities from multiple chunks with parallel processing."""
        all_entities = []

        def process_chunk(chunk):
            return self.extract(chunk, prompt_template)

        # Use ThreadPoolExecutor for parallel LLM calls
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}

            # Process results with progress bar
            for future in tqdm(as_completed(future_to_chunk), total=len(chunks), desc="Extracting entities"):
                try:
                    entities = future.result(timeout=300)  # 5 minute timeout per chunk
                    all_entities.extend(entities)
                except Exception as e:
                    print(f"Error processing chunk: {e}")

        # Deduplicate entities by name
        unique_entities = self._deduplicate_entities(all_entities)

        return unique_entities

    def _call_llm(self, prompt: str) -> str:
        """Call LLM (Gemini or Ollama)."""
        if self.provider == "gemini":
            return self._call_gemini(prompt)
        else:
            return self._call_ollama(prompt)

    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API."""
        url = f"{self.gemini_url}/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0}
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            time.sleep(2)  # 30 RPM limit
            return result['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"Gemini call failed: {e}")
            if "429" in str(e):
                time.sleep(60)
                return self._call_gemini(prompt)
            return "[]"

    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama LLM."""
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0}
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)
            response.raise_for_status()
            time.sleep(0.5)  # Small delay between requests to prevent overload
            return response.json()['message']['content']
        except requests.exceptions.Timeout:
            print(f"Ollama timeout after 120s, skipping this chunk...")
            return "[]"  # Skip instead of retry to avoid infinite loop
        except Exception as e:
            print(f"LLM call failed: {e}")
            return "[]"

    def _parse_entities(self, response: str, chunk_id: str) -> List[Dict[str, Any]]:
        """Parse entities from LLM response."""
        entities = []

        try:
            # Try to extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)

                for entity in parsed:
                    # Handle both dict and string formats
                    if isinstance(entity, dict):
                        entities.append({
                            'name': entity.get('name', ''),
                            'type': entity.get('type', 'UNKNOWN'),
                            'description': entity.get('description', ''),
                            'source_chunk': chunk_id
                        })
                    elif isinstance(entity, str):
                        # If entity is a string, create a simple entity
                        entities.append({
                            'name': entity,
                            'type': 'UNKNOWN',
                            'description': '',
                            'source_chunk': chunk_id
                        })
        except json.JSONDecodeError:
            # Fallback: simple extraction
            pass

        return entities

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate entities by name."""
        seen = {}
        for entity in entities:
            # Skip entities with empty or None names
            if not entity.get('name'):
                continue

            name = entity['name'].lower().strip()

            # Skip empty names after stripping
            if not name:
                continue

            if name not in seen:
                seen[name] = entity
            else:
                # Merge descriptions
                if entity.get('description'):
                    existing = seen[name].get('description', '')
                    seen[name]['description'] = f"{existing} {entity['description']}".strip()

        return list(seen.values())

    def _default_prompt(self) -> str:
        return """Extract all named entities from the following text.
For each entity, provide:
- name: The entity name
- type: One of [PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, PRODUCT, DATE]
- description: Brief description based on context

Return as JSON array:
[{{"name": "...", "type": "...", "description": "..."}}]

Text:
{text}

Entities (JSON only):"""
