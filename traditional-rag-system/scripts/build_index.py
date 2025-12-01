#!/usr/bin/env python3
"""Build RAG index from data."""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from indexing import TextChunker, TextEmbedder, VectorStore
from utils import setup_logger, Config


def main():
    parser = argparse.ArgumentParser(description='Build RAG index from JSONL data')
    parser.add_argument('--config', default='configs/rag_config.yaml', help='Config file path')
    parser.add_argument('--force', action='store_true', help='Force rebuild even if index exists')
    args = parser.parse_args()

    logger = setup_logger('build_index')

    # Load config
    config = Config(args.config)
    cfg = config.load()

    logger.info("="*60)
    logger.info("Building RAG Index")
    logger.info("="*60)

    # Paths
    input_file = Path(cfg['data']['input_file'])
    if not input_file.is_absolute():
        input_file = Path(__file__).parent.parent / input_file

    output_dir = Path(cfg['data']['processed_dir'])
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / output_dir

    index_path = output_dir / 'embeddings' / 'faiss_index.bin'
    chunks_path = output_dir / 'chunks' / 'chunks.json'

    # Check if index already exists
    if index_path.exists() and not args.force:
        logger.warning(f"Index already exists at {index_path}")
        logger.warning("Use --force to rebuild")
        return

    # Create output dirs
    index_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Step 1: Chunk text
    logger.info(f"\nStep 1: Loading and chunking data from {input_file}")
    chunker = TextChunker(
        chunk_size=cfg['chunking']['chunk_size'],
        chunk_overlap=cfg['chunking']['chunk_overlap']
    )

    chunks = chunker.load_and_chunk_jsonl(str(input_file))
    logger.info(f"✓ Created {len(chunks)} chunks")

    # Save chunks
    import json
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ Saved chunks to {chunks_path}")

    # Step 2: Generate embeddings
    logger.info(f"\nStep 2: Generating embeddings with {cfg['embedding']['model']}")
    embedder = TextEmbedder(
        model_name=cfg['embedding']['model'],
        batch_size=cfg['embedding']['batch_size']
    )

    embeddings = embedder.embed_chunks(chunks, show_progress=True)
    logger.info(f"✓ Generated {len(embeddings)} embeddings (dimension={embeddings.shape[1]})")

    # Step 3: Build FAISS index
    logger.info(f"\nStep 3: Building FAISS index")
    vector_store = VectorStore(
        dimension=cfg['embedding']['dimension'],
        index_type=cfg['search']['index_type']
    )

    vector_store.build_index(embeddings, chunks)
    logger.info(f"✓ Built FAISS index with {vector_store.index.ntotal} vectors")

    # Step 4: Save index
    logger.info(f"\nStep 4: Saving index")
    vector_store.save(str(index_path), str(chunks_path))

    elapsed = time.time() - start_time

    logger.info("\n" + "="*60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("="*60)
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"Index type: {cfg['search']['index_type']}")
    logger.info(f"Index saved to: {index_path}")
    logger.info(f"Chunks saved to: {chunks_path}")
    logger.info(f"Total time: {elapsed:.2f}s")
    logger.info("="*60)


if __name__ == '__main__':
    main()
