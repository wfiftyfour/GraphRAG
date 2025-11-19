"""FastAPI application for GraphRAG system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import search, graph

app = FastAPI(
    title="GraphRAG API",
    description="API for GraphRAG knowledge graph based retrieval system",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(graph.router, prefix="/api/graph", tags=["graph"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "GraphRAG API is running",
        "docs": "/docs",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
