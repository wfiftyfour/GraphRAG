"""Local search using vector similarity and graph traversal."""

from typing import List, Dict, Any
import numpy as np
from pathlib import Path


class LocalSearch:
    """Perform local search combining vector similarity and graph context."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.entity_embeddings = None
        self.entity_metadata = None
        self.graph = None

    def load(self):
        """Load all necessary data."""
        embeddings_dir = self.data_dir / "processed/embeddings"

        # Load chunk embeddings
        self.chunk_embeddings = np.load(embeddings_dir / "chunks_embeddings.npy")
        import json
        with open(embeddings_dir / "chunks_metadata.json") as f:
            self.chunk_metadata = json.load(f)

        # Load entity embeddings
        entity_emb_path = embeddings_dir / "entities_embeddings.npy"
        if entity_emb_path.exists():
            self.entity_embeddings = np.load(entity_emb_path)
            with open(embeddings_dir / "entities_metadata.json") as f:
                self.entity_metadata = json.load(f)

        # Load graph
        graph_path = self.data_dir / "output/graph/graph.graphml"
        if graph_path.exists():
            import networkx as nx
            self.graph = nx.read_graphml(graph_path)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search for relevant chunks and entities."""
        results = []

        # 1. Vector search on chunks
        chunk_scores = self._cosine_similarity(query_embedding, self.chunk_embeddings)
        top_chunk_indices = np.argsort(chunk_scores)[-top_k:][::-1]

        for idx in top_chunk_indices:
            results.append({
                'type': 'chunk',
                'score': float(chunk_scores[idx]),
                'content': self.chunk_metadata[idx]['text'],
                'metadata': self.chunk_metadata[idx]
            })

        # 2. Vector search on entities (if available)
        if self.entity_embeddings is not None:
            entity_scores = self._cosine_similarity(query_embedding, self.entity_embeddings)
            top_entity_indices = np.argsort(entity_scores)[-5:][::-1]

            for idx in top_entity_indices:
                entity = self.entity_metadata[idx]
                # Get graph context
                graph_context = self._get_entity_context(entity['name'])

                results.append({
                    'type': 'entity',
                    'score': float(entity_scores[idx]),
                    'content': f"{entity['name']}: {entity.get('description', '')}",
                    'metadata': entity,
                    'graph_context': graph_context
                })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    def _cosine_similarity(self, query: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity."""
        query_norm = query / np.linalg.norm(query)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return np.dot(embeddings_norm, query_norm)

    def _get_entity_context(self, entity_name: str) -> Dict[str, Any]:
        """Get graph context for an entity."""
        if self.graph is None or not self.graph.has_node(entity_name):
            return {}

        neighbors = list(self.graph.neighbors(entity_name))
        relationships = []

        for neighbor in neighbors[:5]:  # Limit neighbors
            edge_data = self.graph.edges[entity_name, neighbor]
            relationships.append({
                'neighbor': neighbor,
                'relationship': edge_data.get('relationship', '')
            })

        return {
            'neighbors': neighbors[:10],
            'relationships': relationships,
            'degree': self.graph.degree(entity_name)
        }
