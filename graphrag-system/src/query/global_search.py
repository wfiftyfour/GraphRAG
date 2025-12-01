"""Global search using community summaries."""

from typing import List, Dict, Any
import numpy as np
from pathlib import Path
import json


class GlobalSearch:
    """Perform global search over community summaries."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.community_embeddings = None
        self.community_reports = None

    def load(self):
        """Load community data."""
        embeddings_dir = self.data_dir / "processed/embeddings"

        # Load community embeddings
        emb_path = embeddings_dir / "communities_embeddings.npy"
        if emb_path.exists():
            self.community_embeddings = np.load(emb_path)
            with open(embeddings_dir / "communities_metadata.json") as f:
                self.community_reports = json.load(f)
        else:
            # Fallback to reports directory
            import pandas as pd
            reports_path = self.data_dir / "output/reports/community_reports.parquet"
            if reports_path.exists():
                df = pd.read_parquet(reports_path)
                self.community_reports = df.to_dict('records')

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant communities."""
        if self.community_embeddings is None:
            return []

        # Vector search on community summaries
        scores = self._cosine_similarity(query_embedding, self.community_embeddings)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            report = self.community_reports[idx]

            # Extract entity names from title (format: "entity1, entity2, and X others")
            title = report.get('title', '')
            entities = []
            if title:
                # Split by comma and "and", take first few entities
                parts = title.replace(' and ', ', ').split(', ')
                entities = [e.strip() for e in parts if not e.strip().endswith('others')]

            results.append({
                'type': 'community',
                'score': float(scores[idx]),
                'community_id': report.get('community_id', idx),
                'title': title,
                'summary': report.get('summary', ''),
                'num_entities': report.get('num_entities', 0),
                'rank': report.get('rank', 0),
                'metadata': {
                    'entities': entities,
                    'name': entities[0] if entities else None
                }
            })

        return results

    def _cosine_similarity(self, query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        query_norm = query / np.linalg.norm(query)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.dot(embeddings_norm, query_norm)

    def get_all_summaries(self, top_k: int = None) -> List[Dict[str, Any]]:
        """Get all community summaries, optionally limited."""
        if self.community_reports is None:
            return []

        # Sort by rank
        sorted_reports = sorted(
            self.community_reports,
            key=lambda x: x.get('rank', 0),
            reverse=True
        )

        if top_k:
            return sorted_reports[:top_k]
        return sorted_reports
