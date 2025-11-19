"""Graph API routes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from src.indexing import GraphBuilder, CommunityDetector
from src.utils import GraphUtils, Config

router = APIRouter()

# Initialize components
_graph = None
_communities = None


def get_graph():
    global _graph, _communities

    if _graph is None:
        config = Config()
        cfg = config.load()

        graph_builder = GraphBuilder(cfg['graph']['output_dir'])
        graph_builder.load()
        _graph = graph_builder.graph

        detector = CommunityDetector(cfg['community']['output_dir'])
        _communities = detector.load()

    return _graph, _communities


class EntityResponse(BaseModel):
    name: str
    type: str
    description: str
    degree: int
    neighbors: List[str]


class GraphStatsResponse(BaseModel):
    num_nodes: int
    num_edges: int
    num_communities: int
    density: float


@router.get("/stats", response_model=GraphStatsResponse)
async def get_stats():
    """Get graph statistics."""
    try:
        graph, communities = get_graph()
        import networkx as nx

        return GraphStatsResponse(
            num_nodes=graph.number_of_nodes(),
            num_edges=graph.number_of_edges(),
            num_communities=len(communities),
            density=nx.density(graph)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entity/{name}", response_model=EntityResponse)
async def get_entity(name: str):
    """Get entity details."""
    try:
        graph, _ = get_graph()

        if not graph.has_node(name):
            raise HTTPException(status_code=404, detail=f"Entity not found: {name}")

        attrs = graph.nodes[name]
        neighbors = list(graph.neighbors(name))

        return EntityResponse(
            name=name,
            type=attrs.get('type', 'UNKNOWN'),
            description=attrs.get('description', ''),
            degree=graph.degree(name),
            neighbors=neighbors[:20]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities")
async def list_entities(limit: int = 100, type: Optional[str] = None):
    """List entities, optionally filtered by type."""
    try:
        graph, _ = get_graph()

        entities = []
        for node, attrs in graph.nodes(data=True):
            if type and attrs.get('type', '').upper() != type.upper():
                continue

            entities.append({
                'name': node,
                'type': attrs.get('type', 'UNKNOWN'),
                'degree': graph.degree(node)
            })

            if len(entities) >= limit:
                break

        # Sort by degree
        entities.sort(key=lambda x: x['degree'], reverse=True)

        return {"entities": entities, "total": len(entities)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/communities")
async def list_communities():
    """List all communities."""
    try:
        _, communities = get_graph()

        result = []
        for community_id, members in communities.items():
            result.append({
                'id': community_id,
                'size': len(members),
                'members': members[:10]  # First 10 members
            })

        # Sort by size
        result.sort(key=lambda x: x['size'], reverse=True)

        return {"communities": result, "total": len(result)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export")
async def export_graph():
    """Export graph as JSON."""
    try:
        graph, _ = get_graph()
        return GraphUtils.export_to_json(graph)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
