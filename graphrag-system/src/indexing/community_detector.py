"""Detect communities in knowledge graph using Leiden algorithm."""

import json
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd


class CommunityDetector:
    """Detect communities using Leiden algorithm."""

    def __init__(self, output_dir: str = "data/output/communities"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.communities = {}
        self.hierarchy = {}

    def detect(self, graph, resolution: float = 1.0) -> Dict[int, List[str]]:
        """Detect communities in the graph."""
        try:
            import leidenalg
            import igraph as ig
        except ImportError:
            raise ImportError("Please install: pip install leidenalg python-igraph")

        # Convert networkx to igraph
        import networkx as nx

        # Create igraph from networkx
        edges = list(graph.edges())
        nodes = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}

        ig_graph = ig.Graph()
        ig_graph.add_vertices(len(nodes))
        ig_graph.add_edges([(node_to_idx[e[0]], node_to_idx[e[1]]) for e in edges])

        # Run Leiden algorithm
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution
        )

        # Map results back to node names
        self.communities = {}
        for community_id, community in enumerate(partition):
            members = [nodes[idx] for idx in community]
            self.communities[community_id] = members

        # Build hierarchy (simplified - single level)
        self.hierarchy = {
            'num_levels': 1,
            'level_0': {
                'num_communities': len(self.communities),
                'communities': self.communities
            }
        }

        return self.communities

    def save(self):
        """Save communities to files."""
        # Save communities as parquet
        communities_data = []
        for community_id, members in self.communities.items():
            for member in members:
                communities_data.append({
                    'entity': member,
                    'community_id': community_id,
                    'community_size': len(members)
                })

        df = pd.DataFrame(communities_data)
        df.to_parquet(self.output_dir / "communities.parquet", index=False)

        # Save hierarchy as JSON
        with open(self.output_dir / "community_hierarchy.json", 'w') as f:
            json.dump(self.hierarchy, f, indent=2)

        print(f"Communities saved: {len(self.communities)} communities")

    def load(self) -> Dict[int, List[str]]:
        """Load communities from files."""
        df = pd.read_parquet(self.output_dir / "communities.parquet")

        self.communities = {}
        for community_id in df['community_id'].unique():
            members = df[df['community_id'] == community_id]['entity'].tolist()
            self.communities[community_id] = members

        # Load hierarchy
        hierarchy_path = self.output_dir / "community_hierarchy.json"
        if hierarchy_path.exists():
            with open(hierarchy_path) as f:
                self.hierarchy = json.load(f)

        return self.communities

    def get_community_for_entity(self, entity: str) -> int:
        """Get community ID for an entity."""
        for community_id, members in self.communities.items():
            if entity in members:
                return community_id
        return -1
