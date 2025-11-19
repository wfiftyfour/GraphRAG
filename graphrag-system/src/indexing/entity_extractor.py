"""Extract entities from text using LLM."""

import json
import requests
from typing import List, Dict, Any


class EntityExtractor:
    """Extract named entities from text chunks using LLM."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model = "llama3.1:8b"

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

    def extract_batch(self, chunks: List[Dict[str, Any]], prompt_template: str = None) -> List[Dict[str, Any]]:
        """Extract entities from multiple chunks."""
        all_entities = []

        for chunk in chunks:
            entities = self.extract(chunk, prompt_template)
            all_entities.extend(entities)

        # Deduplicate entities by name
        unique_entities = self._deduplicate_entities(all_entities)

        return unique_entities

    def _call_llm(self, prompt: str) -> str:
        """Call Ollama LLM."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0}
        }

        try:
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return response.json()['message']['content']
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
                    entities.append({
                        'name': entity.get('name', ''),
                        'type': entity.get('type', 'UNKNOWN'),
                        'description': entity.get('description', ''),
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
            name = entity['name'].lower()
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
