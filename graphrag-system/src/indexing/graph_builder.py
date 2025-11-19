"""Build knowledge graph from entities and relationships."""

import json
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd


class GraphBuilder:
    """Build and manage knowledge graph."""

    def __init__(self, output_dir: str = "data/output/graph"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.graph = None

    def build(self, entities: List[Dict[str, Any]],
              relationships: List[Dict[str, Any]]) -> 'GraphBuilder':
        """Build graph from entities and relationships."""
        try:
            import networkx as nx
            self.graph = nx.Graph()
        except ImportError:
            raise ImportError("Please install networkx: pip install networkx")

        # Add nodes (entities)
        for entity in entities:
            self.graph.add_node(
                entity['name'],
                type=entity.get('type', 'UNKNOWN'),
                description=entity.get('description', ''),
                source_chunk=entity.get('source_chunk', '')
            )

        # Add edges (relationships)
        for rel in relationships:
            self.graph.add_edge(
                rel['source'],
                rel['target'],
                relationship=rel.get('relationship', ''),
                description=rel.get('description', ''),
                weight=rel.get('weight', 1.0),
                source_chunk=rel.get('source_chunk', '')
            )

        return self

    def save(self):
        """Save graph to files."""
        if self.graph is None:
            raise ValueError("Graph not built. Call build() first.")

        # Save as GraphML
        import networkx as nx
        nx.write_graphml(self.graph, self.output_dir / "graph.graphml")

        # Save entities as parquet
        entities_data = []
        for node, attrs in self.graph.nodes(data=True):
            entities_data.append({
                'name': node,
                'type': attrs.get('type', ''),
                'description': attrs.get('description', ''),
                'degree': self.graph.degree(node)
            })

        entities_df = pd.DataFrame(entities_data)
        entities_df.to_parquet(self.output_dir / "entities.parquet", index=False)

        # Save relationships as parquet
        relationships_data = []
        for source, target, attrs in self.graph.edges(data=True):
            relationships_data.append({
                'source': source,
                'target': target,
                'relationship': attrs.get('relationship', ''),
                'description': attrs.get('description', ''),
                'weight': attrs.get('weight', 1.0)
            })

        relationships_df = pd.DataFrame(relationships_data)
        relationships_df.to_parquet(self.output_dir / "relationships.parquet", index=False)

        print(f"Graph saved: {len(entities_data)} entities, {len(relationships_data)} relationships")

    def load(self):
        """Load graph from files."""
        import networkx as nx
        graph_path = self.output_dir / "graph.graphml"

        if graph_path.exists():
            self.graph = nx.read_graphml(graph_path)
        else:
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        return self

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        if self.graph is None:
            return {}

        import networkx as nx

        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'num_components': nx.number_connected_components(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }
