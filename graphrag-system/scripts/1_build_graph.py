#!/usr/bin/env python3
"""Step 1: Build knowledge graph from documents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing import (
    DocumentLoader, TextChunker, EntityExtractor,
    RelationshipExtractor, GraphBuilder
)
from src.utils import setup_logger, Config


def main():
    logger = setup_logger('build_graph')
    config = Config()
    cfg = config.load()

    try:
        # 1. Load documents
        logger.info("Loading documents...")
        loader = DocumentLoader(cfg['data']['input_file'])
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        # 2. Chunk documents
        logger.info("Chunking documents...")
        chunker = TextChunker(
            chunk_size=cfg['chunking']['chunk_size'],
            chunk_overlap=cfg['chunking']['chunk_overlap']
        )
        chunks = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Save chunks
        import json
        chunks_dir = Path(cfg['data']['processed_dir']) / 'chunks'
        chunks_dir.mkdir(parents=True, exist_ok=True)
        with open(chunks_dir / 'chunks.json', 'w') as f:
            json.dump(chunks, f)

        # 3. Extract entities
        logger.info("Extracting entities...")
        entity_extractor = EntityExtractor()
        entity_extractor.model = cfg['llm']['model']

        # Load prompt template
        with open('configs/prompts/entity_extraction.txt') as f:
            entity_prompt = f.read()

        entities = entity_extractor.extract_batch(chunks, entity_prompt)
        logger.info(f"Extracted {len(entities)} entities")

        # Save entities
        entities_dir = Path(cfg['data']['processed_dir']) / 'entities'
        entities_dir.mkdir(parents=True, exist_ok=True)
        with open(entities_dir / 'entities.json', 'w') as f:
            json.dump(entities, f)

        # 4. Extract relationships
        logger.info("Extracting relationships...")
        rel_extractor = RelationshipExtractor()
        rel_extractor.model = cfg['llm']['model']

        with open('configs/prompts/relationship_extraction.txt') as f:
            rel_prompt = f.read()

        relationships = rel_extractor.extract_batch(chunks, entities, rel_prompt)
        logger.info(f"Extracted {len(relationships)} relationships")

        # Save relationships
        rels_dir = Path(cfg['data']['processed_dir']) / 'relationships'
        rels_dir.mkdir(parents=True, exist_ok=True)
        with open(rels_dir / 'relationships.json', 'w') as f:
            json.dump(relationships, f)

        # 5. Build graph
        logger.info("Building knowledge graph...")
        graph_builder = GraphBuilder(cfg['graph']['output_dir'])
        graph_builder.build(entities, relationships)
        graph_builder.save()

        stats = graph_builder.get_stats()
        logger.info(f"Graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

        logger.info("Step 1 complete! Run 2_detect_communities.py next.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
