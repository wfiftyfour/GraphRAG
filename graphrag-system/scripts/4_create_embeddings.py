#!/usr/bin/env python3
"""Step 4: Create embeddings for chunks, entities, and communities."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing import TextEmbedder, CommunitySummarizer
from src.utils import setup_logger, Config


def main():
    logger = setup_logger('create_embeddings')
    config = Config()
    cfg = config.load()

    try:
        # Initialize embedder
        embedder = TextEmbedder(
            model_name=cfg['embedding']['model'],
            output_dir=f"{cfg['data']['processed_dir']}/embeddings"
        )

        # 1. Embed chunks
        logger.info("Loading and embedding chunks...")
        chunks_path = Path(cfg['data']['processed_dir']) / 'chunks/chunks.json'
        with open(chunks_path) as f:
            chunks = json.load(f)

        chunks = embedder.embed_chunks(chunks)
        embedder.save_embeddings(chunks, 'chunks')
        logger.info(f"Embedded {len(chunks)} chunks")

        # 2. Embed entities
        logger.info("Loading and embedding entities...")
        entities_path = Path(cfg['data']['processed_dir']) / 'entities/entities.json'
        with open(entities_path) as f:
            entities = json.load(f)

        entities = embedder.embed_entities(entities)
        embedder.save_embeddings(entities, 'entities')
        logger.info(f"Embedded {len(entities)} entities")

        # 3. Embed community reports
        logger.info("Loading and embedding community reports...")
        summarizer = CommunitySummarizer()
        reports = summarizer.load()

        reports = embedder.embed_communities(reports)
        embedder.save_embeddings(reports, 'communities')
        logger.info(f"Embedded {len(reports)} community reports")

        logger.info("Step 4 complete! Indexing finished. You can now run searches.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
