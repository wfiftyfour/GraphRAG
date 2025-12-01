"""Text chunking utilities."""

from typing import List, Dict, Any
import json


class TextChunker:
    """Chunk text into smaller pieces for embedding."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initialize chunker.

        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to each chunk

        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            # Get chunk
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for delimiter in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_delim = chunk_text.rfind(delimiter)
                    if last_delim > self.chunk_size * 0.5:  # At least 50% of chunk
                        end = start + last_delim + len(delimiter)
                        chunk_text = text[start:end]
                        break

            # Create chunk dict
            chunk = {
                'id': chunk_id,
                'text': chunk_text.strip(),
                'start': start,
                'end': end,
                'metadata': metadata or {}
            }

            chunks.append(chunk)
            chunk_id += 1

            # Move start for next chunk
            start = end - self.chunk_overlap

            # Prevent infinite loop
            if start <= chunks[-1]['start']:
                start = end

        return chunks

    def chunk_conversations(self, conversations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk conversation data (like GraphRAG format).

        Args:
            conversations: List of conversation dicts with 'conversation' field

        Returns:
            List of chunks with metadata
        """
        all_chunks = []

        for conv in conversations:
            # Extract conversation text
            conversation = conv.get('conversation', [])
            conv_id = conv.get('id', 'unknown')
            session_id = conv.get('session_id', 'unknown')

            # Build full conversation text
            text_parts = []
            for turn in conversation:
                role = turn.get('role', 'unknown')
                content = turn.get('content', '')
                text_parts.append(f"{role.upper()}: {content}")

            full_text = '\n\n'.join(text_parts)

            # Metadata
            metadata = {
                'conversation_id': conv_id,
                'session_id': session_id,
                'num_turns': len(conversation),
                'entities': conv.get('entities', []),
                'topics': conv.get('topics', [])
            }

            # Chunk the conversation
            chunks = self.chunk_text(full_text, metadata)

            # Add conversation-level ID
            for chunk in chunks:
                chunk['conversation_id'] = conv_id
                chunk['global_chunk_id'] = f"{conv_id}_chunk_{chunk['id']}"

            all_chunks.extend(chunks)

        return all_chunks

    def load_and_chunk_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load JSONL file and chunk conversations.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of chunks
        """
        conversations = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))

        return self.chunk_conversations(conversations)
