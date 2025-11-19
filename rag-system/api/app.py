"""FastAPI application for GraphRAG system."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import search, query

app = FastAPI(
    title="GraphRAG API",
    description="API for GraphRAG retrieval and generation system",
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
app.include_router(query.router, prefix="/api/query", tags=["query"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "GraphRAG API is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}
