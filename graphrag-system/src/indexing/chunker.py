"""Text chunking utilities."""

from typing import List, Dict, Any
import re


class TextChunker:
    """Split documents into chunks for processing."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a document into chunks."""
        text = document.get('text', '')
        doc_id = document.get('id', '')
        metadata = document.get('metadata', {})

        # Split by sentences first for better chunks
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'chunk_id': f"{doc_id}_chunk_{chunk_index}",
                    'doc_id': doc_id,
                    'text': chunk_text,
                    'metadata': metadata,
                    'chunk_index': chunk_index
                })

                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else ''
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text)
                chunk_index += 1

            current_chunk.append(sentence)
            current_length += sentence_length

        # Don't forget last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'chunk_id': f"{doc_id}_chunk_{chunk_index}",
                'doc_id': doc_id,
                'text': chunk_text,
                'metadata': metadata,
                'chunk_index': chunk_index
            })

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
