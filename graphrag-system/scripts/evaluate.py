#!/usr/bin/env python3
"""Evaluate the GraphRAG system."""

import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query import LocalSearch, GlobalSearch, QueryProcessor, ContextBuilder
from src.generation import LLMClient, PromptBuilder
from src.utils import setup_logger, Config


def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphRAG system')
    parser.add_argument('--queries', required=True, help='Path to queries JSON file')
    parser.add_argument('--output', default='evaluation_results.json', help='Output file')
    parser.add_argument('--search-type', choices=['local', 'global', 'both'], default='both')
    args = parser.parse_args()

    logger = setup_logger('evaluate')
    config = Config()
    cfg = config.load()

    try:
        # Load queries
        with open(args.queries) as f:
            queries = json.load(f)

        logger.info(f"Loaded {len(queries)} queries for evaluation")

        # Initialize components
        processor = QueryProcessor()
        local_search = LocalSearch()
        global_search = GlobalSearch()

        local_search.load()
        global_search.load()

        results = []

        for query_item in queries:
            query = query_item['query']
            logger.info(f"Evaluating: {query}")

            # Process query
            query_data = processor.process(query)

            eval_result = {
                'query': query,
                'query_type': query_data['type']
            }

            # Local search
            if args.search_type in ['local', 'both']:
                start = time.time()
                local_results = local_search.search(query_data['embedding'], top_k=10)
                local_time = time.time() - start

                eval_result['local'] = {
                    'time': local_time,
                    'num_results': len(local_results),
                    'top_score': local_results[0]['score'] if local_results else 0
                }

            # Global search
            if args.search_type in ['global', 'both']:
                start = time.time()
                global_results = global_search.search(query_data['embedding'], top_k=5)
                global_time = time.time() - start

                eval_result['global'] = {
                    'time': global_time,
                    'num_results': len(global_results),
                    'top_score': global_results[0]['score'] if global_results else 0
                }

            results.append(eval_result)

        # Calculate summary stats
        summary = {
            'total_queries': len(results),
            'avg_local_time': sum(r.get('local', {}).get('time', 0) for r in results) / len(results),
            'avg_global_time': sum(r.get('global', {}).get('time', 0) for r in results) / len(results)
        }

        output = {
            'summary': summary,
            'results': results
        }

        # Save results
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Evaluation complete. Results saved to {args.output}")
        logger.info(f"Avg local search time: {summary['avg_local_time']:.4f}s")
        logger.info(f"Avg global search time: {summary['avg_global_time']:.4f}s")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
