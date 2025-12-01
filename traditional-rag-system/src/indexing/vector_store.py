"""Vector storage and similarity search using FAISS."""

from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
import json
from pathlib import Path


class VectorStore:
    """FAISS-based vector store for similarity search."""

    def __init__(self, dimension: int = 1024, index_type: str = "flat"):
        """
        Initialize vector store.

        Args:
            dimension: Embedding dimension
            index_type: FAISS index type ('flat' or 'ivf')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.chunks = []  # Store chunk metadata

    def build_index(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]):
        """
        Build FAISS index from embeddings.

        Args:
            embeddings: Numpy array of embeddings (N x D)
            chunks: List of chunk metadata
        """
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        # Ensure float32 for FAISS
        embeddings = embeddings.astype('float32')

        # Create index
        if self.index_type == "flat":
            # Simple flat index (exact search)
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product (cosine for normalized vectors)
        elif self.index_type == "ivf":
            # IVF index for faster search (approximate)
            n_list = min(100, len(embeddings) // 10)
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_list)
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Add vectors
        self.index.add(embeddings)
        self.chunks = chunks

        print(f"Built FAISS index with {self.index.ntotal} vectors")

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return

        Returns:
            List of results with chunks and scores
        """
        if self.index is None:
            raise ValueError("Index not built yet")

        # Ensure correct shape and type
        query_embedding = query_embedding.astype('float32').reshape(1, -1)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for invalid results
                continue

            result = {
                'rank': i + 1,
                'score': float(score),
                'chunk_id': idx,
                'text': self.chunks[idx]['text'],
                'content': self.chunks[idx]['text'],  # Add for evaluator compatibility
                'metadata': self.chunks[idx].get('metadata', {}),
                'conversation_id': self.chunks[idx].get('conversation_id', 'unknown'),
                'type': 'chunk'  # For compatibility with GraphRAG format
            }
            results.append(result)

        return results

    def save(self, index_path: str, chunks_path: str):
        """
        Save index and chunks to disk.

        Args:
            index_path: Path to save FAISS index
            chunks_path: Path to save chunks metadata
        """
        if self.index is None:
            raise ValueError("No index to save")

        # Save FAISS index
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)

        # Save chunks
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

        print(f"Saved index to {index_path}")
        print(f"Saved chunks to {chunks_path}")

    def load(self, index_path: str, chunks_path: str):
        """
        Load index and chunks from disk.

        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks metadata
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Load chunks
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        print(f"Loaded index with {self.index.ntotal} vectors")
        print(f"Loaded {len(self.chunks)} chunks")

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'num_vectors': self.index.ntotal if self.index else 0,
            'num_chunks': len(self.chunks),
            'dimension': self.dimension,
            'index_type': self.index_type
        }
