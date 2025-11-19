"""Text chunking utilities."""

from typing import List, Dict, Any


class TextChunker:
    """Split documents into chunks for processing."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split a document into chunks."""
        text = document.get('text', '')
        doc_id = document.get('id', '')
        metadata = document.get('metadata', {})

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append({
                'chunk_id': f"{doc_id}_chunk_{chunk_index}",
                'doc_id': doc_id,
                'text': chunk_text,
                'metadata': metadata,
                'start_pos': start,
                'end_pos': end
            })

            start = end - self.chunk_overlap
            chunk_index += 1

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
