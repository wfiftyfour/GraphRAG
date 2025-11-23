"""Text embedding utilities."""

from typing import List, Dict, Any
from pathlib import Path
import numpy as np
import json


class TextEmbedder:
    """Generate embeddings for text chunks and entities."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5",
                 output_dir: str = "data/processed/embeddings"):
        self.model_name = model_name
        self.model = None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self):
        """Load the embedding model."""
        import sys
        print("[BGE] === ENTERING load_model() ===", flush=True)
        sys.stdout.flush()

        try:
            print("[BGE] Importing dependencies (should be fast if pre-loaded)...", flush=True)
            from sentence_transformers import SentenceTransformer
            import torch
            import os
            print("[BGE] Dependencies imported/accessed successfully", flush=True)

            # Determine device (GPU if available, else CPU)
            print("[BGE] Checking CUDA availability...", flush=True)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[BGE] Using device: {device}", flush=True)

            # Try loading from cache first (offline mode)
            try:
                print(f"[BGE] Loading model '{self.model_name}' from cache...", flush=True)
                print("[BGE] This may take 10-30 seconds for large models...", flush=True)
                sys.stdout.flush()

                self.model = SentenceTransformer(
                    self.model_name,
                    device=device,
                    local_files_only=True
                )
                print(f"[BGE] Successfully loaded cached model on {device}", flush=True)

            except Exception as cache_error:
                print(f"[BGE] Cache load failed: {cache_error}")
                print(f"[BGE] Downloading model '{self.model_name}' from HuggingFace...")
                print("[BGE] This will take 5-10 minutes for 1.34GB model...")

                # If offline fails, try online download
                self.model = SentenceTransformer(self.model_name, device=device)
                print(f"[BGE] Successfully downloaded model to {device}")

        except ImportError as e:
            raise ImportError(f"Please install sentence-transformers: pip install sentence-transformers\nError: {e}")
        except Exception as e:
            print(f"[BGE] ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.model is None:
            self.load_model()

        # BGE models perform better with normalization
        if "bge" in self.model_name.lower():
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True
            )
        else:
            embeddings = self.model.encode(texts, show_progress_bar=True)

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a query."""
        if self.model is None:
            self.load_model()

        # BGE models need instruction prefix for queries
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True if "bge" in self.model_name.lower() else False
        )
        return embedding[0]

    def embed_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Generate embeddings for chunks with batching."""
        if not chunks:
            return chunks

        print(f"Embedding {len(chunks)} chunks in batches of {batch_size}...")

        # Process in batches
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            texts = [chunk['text'] for chunk in batch]

            print(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} ({len(batch)} chunks)...")
            batch_embeddings = self.embed(texts)
            all_embeddings.extend(batch_embeddings)

        # Assign embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = all_embeddings[i].tolist()

        return chunks

    def embed_entities(self, entities: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Generate embeddings for entities with batching."""
        if not entities:
            return entities

        print(f"Embedding {len(entities)} entities in batches of {batch_size}...")

        # Process in batches
        all_embeddings = []
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i+batch_size]
            texts = [f"{e.get('name', '')}: {e.get('description', '')}" for e in batch]

            print(f"Processing batch {i//batch_size + 1}/{(len(entities)-1)//batch_size + 1} ({len(batch)} entities)...")
            batch_embeddings = self.embed(texts)
            all_embeddings.extend(batch_embeddings)

        # Assign embeddings to entities
        for i, entity in enumerate(entities):
            entity['embedding'] = all_embeddings[i].tolist()

        return entities

    def embed_communities(self, reports: List[Dict[str, Any]], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Generate embeddings for community reports with batching."""
        if not reports:
            return reports

        print(f"Embedding {len(reports)} community reports in batches of {batch_size}...")

        # Process in batches
        all_embeddings = []
        for i in range(0, len(reports), batch_size):
            batch = reports[i:i+batch_size]
            texts = [f"{r.get('title', '')}\n{r.get('summary', '')}" for r in batch]

            print(f"Processing batch {i//batch_size + 1}/{(len(reports)-1)//batch_size + 1} ({len(batch)} reports)...")
            batch_embeddings = self.embed(texts)
            all_embeddings.extend(batch_embeddings)

        # Assign embeddings to reports
        for i, report in enumerate(reports):
            report['embedding'] = all_embeddings[i].tolist()

        return reports

    def save_embeddings(self, data: List[Dict[str, Any]], name: str):
        """Save embeddings to file."""
        if not data:
            print(f"Warning: No {name} data to save")
            return

        # Extract embeddings as numpy array
        embeddings = np.array([item['embedding'] for item in data])
        np.save(self.output_dir / f"{name}_embeddings.npy", embeddings)

        # Save metadata (without embeddings)
        metadata = []
        for item in data:
            meta = {k: v for k, v in item.items() if k != 'embedding'}
            metadata.append(meta)

        with open(self.output_dir / f"{name}_metadata.json", 'w') as f:
            json.dump(metadata, f)

        print(f"Saved {len(data)} {name} embeddings")

    def load_embeddings(self, name: str):
        """Load embeddings from file."""
        embeddings = np.load(self.output_dir / f"{name}_embeddings.npy")

        with open(self.output_dir / f"{name}_metadata.json") as f:
            metadata = json.load(f)

        # Combine
        for i, meta in enumerate(metadata):
            meta['embedding'] = embeddings[i].tolist()

        return metadata
