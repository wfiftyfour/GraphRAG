#!/usr/bin/env python3
"""Build index from graphrag_input.jsonl."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import DocumentLoader, TextChunker, TextPreprocessor, TextEmbedder
from src.indexing import LocalIndexer, GlobalIndexer
from src.utils import setup_logger, Config


def main():
    parser = argparse.ArgumentParser(description='Build GraphRAG index')
    parser.add_argument('--input', default='data/graphrag_input.jsonl', help='Input file path')
    parser.add_argument('--config', default='configs', help='Config directory')
    args = parser.parse_args()

    logger = setup_logger('build_index')
    config = Config(args.config)

    try:
        # Load configurations
        index_config = config.load('index_config')
        model_config = config.load('model_config')

        # Load documents
        logger.info(f"Loading documents from {args.input}")
        loader = DocumentLoader(args.input)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents")

        # Chunk documents
        logger.info("Chunking documents...")
        chunker = TextChunker(
            chunk_size=index_config['chunking']['chunk_size'],
            chunk_overlap=index_config['chunking']['chunk_overlap']
        )
        chunks = chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Preprocess
        logger.info("Preprocessing chunks...")
        preprocessor = TextPreprocessor(
            lowercase=index_config['preprocessing']['lowercase'],
            remove_special_chars=index_config['preprocessing']['remove_special_chars']
        )
        chunks = preprocessor.preprocess_chunks(chunks)

        # Generate embeddings
        logger.info("Generating embeddings...")
        embedder = TextEmbedder(model_name=model_config['embedding']['model_name'])
        chunks = embedder.embed_chunks(chunks)

        # Build local index
        logger.info("Building local index...")
        local_indexer = LocalIndexer(index_config['local_index']['path'])
        local_indexer.build_index(chunks)
        local_indexer.save()
        logger.info(f"Local index saved to {index_config['local_index']['path']}")

        # Build global index
        logger.info("Building global index...")
        global_indexer = GlobalIndexer(index_config['global_index']['path'])
        global_indexer.build_index(chunks)
        global_indexer.save()
        logger.info(f"Global index saved to {index_config['global_index']['path']}")

        logger.info("Index building complete!")

    except Exception as e:
        logger.error(f"Error building index: {e}")
        raise


if __name__ == '__main__':
    main()
