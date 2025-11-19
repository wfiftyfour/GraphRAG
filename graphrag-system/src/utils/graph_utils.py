"""Graph utility functions."""

from typing import List, Dict, Any, Optional


class GraphUtils:
    """Utility functions for graph operations."""

    @staticmethod
    def get_subgraph(graph, nodes: List[str], depth: int = 1):
        """Get subgraph around specified nodes."""
        import networkx as nx

        all_nodes = set(nodes)

        # Expand to neighbors
        for _ in range(depth):
            new_nodes = set()
            for node in all_nodes:
                if graph.has_node(node):
                    new_nodes.update(graph.neighbors(node))
            all_nodes.update(new_nodes)

        return graph.subgraph(all_nodes).copy()

    @staticmethod
    def get_paths(graph, source: str, target: str, max_length: int = 3) -> List[List[str]]:
        """Get all paths between two nodes."""
        import networkx as nx

        if not graph.has_node(source) or not graph.has_node(target):
            return []

        try:
            paths = list(nx.all_simple_paths(
                graph, source, target, cutoff=max_length
            ))
            return paths
        except:
            return []

    @staticmethod
    def get_central_nodes(graph, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get most central nodes by various metrics."""
        import networkx as nx

        # Degree centrality
        degree_cent = nx.degree_centrality(graph)

        # Sort by degree
        sorted_nodes = sorted(
            degree_cent.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for node, centrality in sorted_nodes:
            results.append({
                'node': node,
                'degree_centrality': centrality,
                'degree': graph.degree(node)
            })

        return results

    @staticmethod
    def export_to_json(graph) -> Dict[str, Any]:
        """Export graph to JSON format."""
        nodes = []
        for node, attrs in graph.nodes(data=True):
            nodes.append({
                'id': node,
                **attrs
            })

        edges = []
        for source, target, attrs in graph.edges(data=True):
            edges.append({
                'source': source,
                'target': target,
                **attrs
            })

        return {
            'nodes': nodes,
            'edges': edges,
            'stats': {
                'num_nodes': graph.number_of_nodes(),
                'num_edges': graph.number_of_edges()
            }
        }
