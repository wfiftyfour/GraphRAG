"""Text preprocessing utilities."""

import re
from typing import List, Dict, Any


class TextPreprocessor:
    """Preprocess text for embedding and indexing."""

    def __init__(self, lowercase: bool = False, remove_special_chars: bool = False):
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars

    def preprocess(self, text: str) -> str:
        """Preprocess a single text string."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if self.lowercase:
            text = text.lower()

        if self.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        return text

    def preprocess_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess multiple chunks."""
        for chunk in chunks:
            chunk['processed_text'] = self.preprocess(chunk['text'])
        return chunks
