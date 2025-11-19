"""Search API routes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from src.query import LocalSearch, GlobalSearch, QueryProcessor, ContextBuilder
from src.generation import PromptBuilder, LLMClient, AnswerFormatter
from src.utils import Config

router = APIRouter()

# Initialize components (singleton)
_local_search = None
_global_search = None
_processor = None
_config = None


def get_components():
    global _local_search, _global_search, _processor, _config

    if _processor is None:
        _processor = QueryProcessor()
        _local_search = LocalSearch()
        _global_search = GlobalSearch()

        _local_search.load()
        _global_search.load()

        _config = Config()
        _config.load()

    return _local_search, _global_search, _processor, _config


class SearchRequest(BaseModel):
    query: str
    search_type: str = "auto"  # local, global, auto
    top_k: int = 10
    generate: bool = True


class Source(BaseModel):
    id: str
    type: str
    content: str


class SearchResponse(BaseModel):
    answer: str
    search_type: str
    sources: List[Source]
    num_results: int


@router.post("/", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Perform search with optional answer generation."""
    try:
        local_search, global_search, processor, config = get_components()

        # Process query
        query_data = processor.process(request.query)

        # Determine search type
        if request.search_type == "auto":
            search_type = query_data['type']
        else:
            search_type = request.search_type

        # Perform search
        if search_type == "local":
            results = local_search.search(query_data['embedding'], request.top_k)
        else:
            results = global_search.search(query_data['embedding'], request.top_k)

        # Build context
        context_builder = ContextBuilder()

        if search_type == "local":
            context = context_builder.build_local_context(results)
        else:
            context = context_builder.build_global_context(results)

        sources = context_builder.format_sources(results)

        # Generate answer if requested
        if request.generate:
            prompt_builder = PromptBuilder()

            if search_type == "local":
                prompt = prompt_builder.build_local_prompt(request.query, context)
            else:
                prompt = prompt_builder.build_global_prompt(request.query, context)

            llm = LLMClient(model=config.llm.get('model', 'llama3.1:8b'))
            answer = llm.generate(prompt)

            formatter = AnswerFormatter()
            formatted = formatter.format(answer, sources)
            answer = formatted['answer']
        else:
            answer = context

        return SearchResponse(
            answer=answer,
            search_type=search_type,
            sources=[Source(**s) for s in sources[:5]],
            num_results=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types")
async def get_search_types():
    """Get available search types."""
    return {
        "types": ["local", "global", "auto"],
        "descriptions": {
            "local": "Search specific documents and entities",
            "global": "Search community summaries for high-level overview",
            "auto": "Automatically determine best search type"
        }
    }
