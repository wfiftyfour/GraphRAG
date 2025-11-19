"""Reranker for improving search results."""

from typing import List, Dict, Any


class Reranker:
    """Rerank search results for better relevance."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        """Load the reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, max_length=512)
        except ImportError:
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """Rerank results based on query relevance."""
        if self.model is None:
            self.load_model()

        if not results:
            return []

        # Prepare pairs for scoring
        pairs = [(query, r['metadata']['text']) for r in results]

        # Get rerank scores
        scores = self.model.predict(pairs)

        # Update results with new scores
        for i, result in enumerate(results):
            result['rerank_score'] = float(scores[i])

        # Sort by rerank score
        results.sort(key=lambda x: x['rerank_score'], reverse=True)

        return results[:top_k]
