"""Query API routes with generation."""

import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval import HybridSearch, Reranker
from src.generation import ResponseGenerator, PostProcessor


router = APIRouter()

# Initialize components
search_engine = None
reranker = None
generator = None
post_processor = None


def get_search_engine():
    global search_engine
    if search_engine is None:
        search_engine = HybridSearch()
        search_engine.load()
    return search_engine


def get_generator():
    global generator
    if generator is None:
        generator = ResponseGenerator()
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            generator.setup_openai(api_key)
    return generator


class QueryRequest(BaseModel):
    query: str
    top_k: int = 10
    generate: bool = True


class Source(BaseModel):
    doc_id: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


@router.post("/", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query with optional response generation."""
    try:
        # Search
        engine = get_search_engine()
        results = engine.search(request.query, request.top_k)

        # Rerank
        reranker = Reranker()
        results = reranker.rerank(request.query, results, request.top_k)

        if request.generate:
            # Generate response
            gen = get_generator()
            if gen.client is None:
                raise HTTPException(status_code=500, detail="OpenAI API key not configured")

            response = gen.generate(request.query, results)

            # Post-process
            processor = PostProcessor()
            processed = processor.process(response, results)

            sources = [
                Source(
                    doc_id=r['metadata'].get('doc_id', ''),
                    text=r['metadata']['text'][:200]
                )
                for r in results[:3]
            ]

            return QueryResponse(answer=processed['response'], sources=sources)
        else:
            # Return concatenated context
            context = "\n\n".join([r['metadata']['text'] for r in results[:3]])
            sources = [
                Source(
                    doc_id=r['metadata'].get('doc_id', ''),
                    text=r['metadata']['text'][:200]
                )
                for r in results[:3]
            ]

            return QueryResponse(answer=context, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
