#!/usr/bin/env python3
"""Retry failed/timeout chunks from relationship extraction."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing import RelationshipExtractor
from src.utils import setup_logger, Config


def main():
    logger = setup_logger('retry_failed_chunks')
    config = Config()
    cfg = config.load()

    # Load chunks and entities
    logger.info("Loading chunks and entities...")
    with open('data/processed/chunks/chunks.json') as f:
        all_chunks = json.load(f)

    with open('data/processed/entities/entities.json') as f:
        entities = json.load(f)

    # Load failed chunk IDs from file
    failed_chunks_file = Path('data/processed/relationships/failed_chunks.json')
    if not failed_chunks_file.exists():
        logger.info("No failed_chunks.json found. All chunks processed successfully!")
        return

    with open(failed_chunks_file) as f:
        failed_chunk_ids = json.load(f)

    if not failed_chunk_ids:
        logger.info("No failed chunks found. All chunks processed successfully!")
        return

    # Load existing relationships
    rels_file = Path('data/processed/relationships/relationships.json')
    if not rels_file.exists():
        logger.error("No relationships file found. Run 1_build_graph.py first.")
        return

    with open(rels_file) as f:
        existing_rels = json.load(f)

    # Get failed chunks from all chunks
    failed_chunks = [c for c in all_chunks if c.get('chunk_id') in failed_chunk_ids]

    logger.info(f"Found {len(failed_chunks)} failed/timeout chunks to retry")
    logger.info(f"Failed chunk IDs: {failed_chunk_ids[:10]}")
    if len(failed_chunk_ids) > 10:
        logger.info(f"... and {len(failed_chunk_ids) - 10} more")

    # Ask for confirmation
    print(f"\nDo you want to retry {len(failed_chunks)} chunks? (y/n): ", end='')
    response = input().strip().lower()
    if response != 'y':
        logger.info("Retry cancelled by user")
        return

    # Extract relationships from failed chunks
    logger.info("Extracting relationships from failed chunks...")
    rel_extractor = RelationshipExtractor()
    rel_extractor.model = cfg['llm']['model']

    with open('configs/prompts/relationship_extraction.txt') as f:
        rel_prompt = f.read()

    # Process with smaller batch size for stability
    new_relationships = rel_extractor.extract_batch(
        failed_chunks,
        entities,
        rel_prompt,
        max_workers=1,
        batch_size=20  # Smaller batch for retry
    )

    logger.info(f"Extracted {len(new_relationships)} new relationships from failed chunks")

    # Merge with existing relationships
    all_relationships = existing_rels + new_relationships

    # Save merged relationships
    logger.info("Saving merged relationships...")
    with open(rels_file, 'w') as f:
        json.dump(all_relationships, f)

    logger.info(f"Total relationships: {len(all_relationships)} (was {len(existing_rels)}, added {len(new_relationships)})")
    logger.info("Retry complete! You can now run 1_build_graph.py to rebuild the graph with new relationships.")


if __name__ == '__main__':
    main()
