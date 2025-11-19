#!/usr/bin/env python3
"""Run search queries."""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import LocalSearch, GlobalSearch, HybridSearch, Reranker
from src.query import QueryBuilder, QueryRouter
from src.generation import ResponseGenerator, PostProcessor
from src.utils import setup_logger, Config


def main():
    parser = argparse.ArgumentParser(description='Run GraphRAG search')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--type', choices=['local', 'global', 'hybrid', 'auto'],
                        default='auto', help='Search type')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    parser.add_argument('--generate', action='store_true', help='Generate response')
    args = parser.parse_args()

    logger = setup_logger('run_search')
    config = Config('configs')

    try:
        # Load configurations
        config.load('search_config')
        config.load('model_config')
        config.load_env()

        # Determine search type
        if args.type == 'auto':
            router = QueryRouter()
            search_type = router.route(args.query)
            logger.info(f"Auto-routed to {search_type} search")
        else:
            search_type = args.type

        # Run search
        if search_type == 'local':
            searcher = LocalSearch()
        elif search_type == 'global':
            searcher = GlobalSearch()
        else:
            searcher = HybridSearch()

        searcher.load()
        results = searcher.search(args.query, args.top_k)

        # Rerank results
        reranker = Reranker()
        results = reranker.rerank(args.query, results, args.top_k)

        # Display results
        print(f"\nSearch Results for: '{args.query}'")
        print("=" * 50)
        for i, result in enumerate(results):
            print(f"\n[{i+1}] Score: {result.get('rerank_score', result['score']):.4f}")
            print(f"    Text: {result['metadata']['text'][:200]}...")

        # Generate response if requested
        if args.generate:
            api_key = config.get_env('OPENAI_API_KEY')
            if not api_key:
                logger.error("OPENAI_API_KEY not set")
                return

            generator = ResponseGenerator()
            generator.setup_openai(api_key)
            response = generator.generate(args.query, results)

            processor = PostProcessor()
            processed = processor.process(response, results)

            print("\n" + "=" * 50)
            print("Generated Response:")
            print(processed['response'])

    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise


if __name__ == '__main__':
    main()
