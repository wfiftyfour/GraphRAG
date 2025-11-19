"""Extract relationships between entities using LLM."""

import json
import requests
from typing import List, Dict, Any


class RelationshipExtractor:
    """Extract relationships between entities using LLM."""

    def __init__(self):
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model = "llama3.1:8b"

    def extract(self, chunk: Dict[str, Any], entities: List[Dict[str, Any]],
                prompt_template: str = None) -> List[Dict[str, Any]]:
        """Extract relationships from a chunk given its entities."""
        text = chunk.get('text', '')
        chunk_id = chunk.get('chunk_id', '')

        # Filter entities for this chunk
        chunk_entities = [e for e in entities if e.get('source_chunk') == chunk_id]

        if len(chunk_entities) < 2:
            return []

        entity_names = [e['name'] for e in chunk_entities]

        prompt = prompt_template or self._default_prompt()
        full_prompt = prompt.format(
            text=text,
            entities=', '.join(entity_names)
        )

        response = self._call_llm(full_prompt)
        relationships = self._parse_relationships(response, chunk_id)

        return relationships

    def extract_batch(self, chunks: List[Dict[str, Any]],
                      entities: List[Dict[str, Any]],
                      prompt_template: str = None) -> List[Dict[str, Any]]:
        """Extract relationships from multiple chunks."""
        all_relationships = []

        for chunk in chunks:
            relationships = self.extract(chunk, entities, prompt_template)
            all_relationships.extend(relationships)

        # Deduplicate and aggregate
        unique_relationships = self._deduplicate_relationships(all_relationships)

        return unique_relationships

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

    def _parse_relationships(self, response: str, chunk_id: str) -> List[Dict[str, Any]]:
        """Parse relationships from LLM response."""
        relationships = []

        try:
            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)

                for rel in parsed:
                    relationships.append({
                        'source': rel.get('source', ''),
                        'target': rel.get('target', ''),
                        'relationship': rel.get('relationship', ''),
                        'description': rel.get('description', ''),
                        'weight': rel.get('weight', 1.0),
                        'source_chunk': chunk_id
                    })
        except json.JSONDecodeError:
            pass

        return relationships

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate relationships."""
        seen = {}

        for rel in relationships:
            key = (rel['source'].lower(), rel['target'].lower(), rel['relationship'].lower())
            if key not in seen:
                seen[key] = rel
            else:
                # Increase weight for repeated relationships
                seen[key]['weight'] = seen[key].get('weight', 1) + 1

        return list(seen.values())

    def _default_prompt(self) -> str:
        return """Given the text and entities, extract relationships between them.

For each relationship provide:
- source: Source entity name
- target: Target entity name
- relationship: Type of relationship (e.g., FOUNDED, WORKS_AT, LOCATED_IN, OWNS, CREATED)
- description: Brief description of the relationship

Text:
{text}

Entities: {entities}

Return as JSON array:
[{{"source": "...", "target": "...", "relationship": "...", "description": "..."}}]

Relationships (JSON only):"""
