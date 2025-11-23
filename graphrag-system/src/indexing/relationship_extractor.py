"""Extract relationships between entities using LLM."""

import json
import requests
import time
from typing import List, Dict, Any
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class RelationshipExtractor:
    """Extract relationships between entities using LLM."""

    def __init__(self, provider="ollama"):
        self.provider = provider

        # Ollama config - lightweight model for RTX 3050 8GB
        self.ollama_url = "http://localhost:11434/api/chat"
        self.ollama_model = "qwen2.5:3b"  # 3B model - fast, low VRAM (~4GB)

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
                      prompt_template: str = None,
                      max_workers: int = 1,
                      batch_size: int = 30) -> List[Dict[str, Any]]:
        """Extract relationships from multiple chunks with batching to prevent memory issues."""
        all_relationships = []
        failed_chunk_ids = []  # Track failed chunk IDs

        # Process in smaller batches to prevent Ollama crashes
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)")

            def process_chunk(chunk):
                return self.extract(chunk, entities, prompt_template)

            # Use ThreadPoolExecutor for parallel LLM calls
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in batch}

                for future in tqdm(as_completed(future_to_chunk), total=len(batch), desc=f"Batch {i//batch_size + 1}"):
                    try:
                        relationships = future.result(timeout=300)  # 5 minute timeout
                        all_relationships.extend(relationships)
                    except Exception as e:
                        chunk = future_to_chunk[future]
                        chunk_id = chunk.get('chunk_id', 'unknown')
                        failed_chunk_ids.append(chunk_id)
                        print(f"Error processing chunk {chunk_id}: {e}")

            # Clear Ollama model from memory after each batch
            if i + batch_size < len(chunks):
                print("Unloading model to free VRAM...")
                try:
                    requests.post("http://localhost:11434/api/generate",
                                json={"model": self.ollama_model, "keep_alive": 0}, timeout=5)
                    time.sleep(2)
                except:
                    pass

        # Save failed chunk IDs for retry
        if failed_chunk_ids:
            print(f"\n⚠️  {len(failed_chunk_ids)} chunks failed/timeout")
            from pathlib import Path
            failed_file = Path("data/processed/relationships/failed_chunks.json")
            failed_file.parent.mkdir(parents=True, exist_ok=True)
            with open(failed_file, 'w') as f:
                json.dump(failed_chunk_ids, f, indent=2)
            print(f"Failed chunk IDs saved to: {failed_file}")

        # Deduplicate and aggregate
        unique_relationships = self._deduplicate_relationships(all_relationships)

        return unique_relationships

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
                    # Handle both dict and other formats
                    if isinstance(rel, dict):
                        relationships.append({
                            'source': rel.get('source', ''),
                            'target': rel.get('target', ''),
                            'relationship': rel.get('relationship', ''),
                            'description': rel.get('description', ''),
                            'weight': rel.get('weight', 1.0),
                            'source_chunk': chunk_id
                        })
                    # Skip non-dict formats for relationships as they need more structure
        except json.JSONDecodeError:
            pass

        return relationships

    def _deduplicate_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate relationships."""
        seen = {}

        for rel in relationships:
            # Skip relationships with missing required fields
            source = rel.get('source', '').strip()
            target = rel.get('target', '').strip()
            relationship = rel.get('relationship', '').strip()

            if not source or not target or not relationship:
                continue

            key = (source.lower(), target.lower(), relationship.lower())
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
