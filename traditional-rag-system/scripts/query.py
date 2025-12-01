#!/usr/bin/env python3
"""Interactive query script for Traditional RAG system."""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retrieval import RAGRetriever
from generation import LLMClient, PromptBuilder
from utils import setup_logger, Config


def main():
    parser = argparse.ArgumentParser(description='Query Traditional RAG system')
    parser.add_argument('query', help='Query string')
    parser.add_argument('--config', default='configs/rag_config.yaml', help='Config file')
    parser.add_argument('--top-k', type=int, help='Number of results to retrieve')
    parser.add_argument('--no-answer', action='store_true', help='Only retrieve, do not generate answer')
    parser.add_argument('--show-sources', action='store_true', help='Show source chunks')
    args = parser.parse_args()

    logger = setup_logger('query')

    # Load config
    config = Config(args.config)
    cfg = config.load()

    # Override top_k if specified
    top_k = args.top_k if args.top_k else cfg['search']['top_k']

    try:
        # Initialize components
        logger.info("Loading RAG index...")

        # Paths
        processed_dir = Path(__file__).parent.parent / cfg['data']['processed_dir']
        index_path = processed_dir / 'embeddings' / 'faiss_index.bin'
        chunks_path = processed_dir / 'chunks' / 'chunks.json'

        # Load retriever
        retriever = RAGRetriever(cfg)
        retriever.load(
            str(index_path),
            str(chunks_path),
            cfg['embedding']['model']
        )
        logger.info("RAG index loaded")

        # Initialize LLM
        llm = None
        prompt_builder = None
        if not args.no_answer:
            llm = LLMClient(
                model=cfg['llm']['model'],
                base_url=cfg['llm']['base_url']
            )
            prompt_builder = PromptBuilder()
            logger.info(f"LLM initialized: {cfg['llm']['model']}")

        # Execute query
        print("\n" + "="*60)
        print(f"QUERY: {args.query}")
        print("="*60)

        # Retrieve
        start = time.time()
        search_results = retriever.retrieve(args.query, top_k=top_k)
        retrieval_time = time.time() - start

        print(f"\nRetrieved {len(search_results)} chunks in {retrieval_time:.3f}s")

        # Show sources if requested
        if args.show_sources:
            print("\n" + "-"*60)
            print("RETRIEVED SOURCES:")
            print("-"*60)
            for i, result in enumerate(search_results[:5], 1):  # Show top 5
                print(f"\n[{i}] Score: {result['score']:.4f}")
                print(f"Conversation: {result.get('conversation_id', 'unknown')}")
                print(f"Text: {result['text'][:200]}...")

        # Generate answer
        if not args.no_answer and llm:
            print("\n" + "-"*60)
            print("GENERATING ANSWER...")
            print("-"*60)

            context = retriever.get_context(search_results, max_tokens=2000)
            prompt = prompt_builder.build_health_prompt(args.query, context)

            start = time.time()
            answer = llm.generate(
                prompt,
                temperature=cfg['generation']['temperature'],
                max_tokens=cfg['generation']['max_tokens']
            )
            generation_time = time.time() - start

            print(f"\nAnswer generated in {generation_time:.3f}s\n")
            print("ANSWER:")
            print("-"*60)
            print(answer)
            print("-"*60)

            # Show statistics
            print(f"\nSTATISTICS:")
            print(f"  Retrieval time: {retrieval_time:.3f}s")
            print(f"  Generation time: {generation_time:.3f}s")
            print(f"  Total time: {retrieval_time + generation_time:.3f}s")
            print(f"  Chunks retrieved: {len(search_results)}")
            print(f"  Top score: {search_results[0]['score']:.4f}")

        print("\n" + "="*60 + "\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
