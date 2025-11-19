"""Search API routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval import HybridSearch, Reranker


router = APIRouter()

# Initialize search (singleton pattern)
search_engine = None
reranker = None


def get_search_engine():
    global search_engine
    if search_engine is None:
        search_engine = HybridSearch()
        search_engine.load()
    return search_engine


def get_reranker():
    global reranker
    if reranker is None:
        reranker = Reranker()
    return reranker


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_reranker: bool = True


class SearchResult(BaseModel):
    text: str
    score: float
    doc_id: str


class SearchResponse(BaseModel):
    results: List[SearchResult]
    total: int


@router.post("/", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform a search query."""
    try:
        engine = get_search_engine()
        results = engine.search(request.query, request.top_k)

        if request.use_reranker:
            reranker = get_reranker()
            results = reranker.rerank(request.query, results, request.top_k)

        formatted_results = [
            SearchResult(
                text=r['metadata']['text'],
                score=r.get('rerank_score', r['score']),
                doc_id=r['metadata'].get('doc_id', '')
            )
            for r in results
        ]

        return SearchResponse(results=formatted_results, total=len(formatted_results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
