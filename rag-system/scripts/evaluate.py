#!/usr/bin/env python3
"""Evaluate the RAG system."""

import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import HybridSearch, Reranker
from src.utils import setup_logger, Metrics, Timer


def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphRAG system')
    parser.add_argument('--queries', required=True, help='Path to queries JSON file')
    parser.add_argument('--output', default='evaluation_results.json', help='Output file')
    args = parser.parse_args()

    logger = setup_logger('evaluate')
    metrics = Metrics()

    try:
        # Load queries
        with open(args.queries, 'r') as f:
            queries = json.load(f)

        logger.info(f"Loaded {len(queries)} queries for evaluation")

        # Initialize search
        searcher = HybridSearch()
        searcher.load()

        reranker = Reranker()

        results = []

        for query_item in queries:
            query = query_item['query']
            expected = query_item.get('expected', [])

            # Search
            with Timer() as t:
                search_results = searcher.search(query, top_k=10)
                search_results = reranker.rerank(query, search_results, top_k=10)

            metrics.add_retrieval_time(t.duration)

            # Calculate relevance (simple overlap)
            retrieved_ids = [r['metadata'].get('doc_id', '') for r in search_results]
            relevant = len(set(retrieved_ids) & set(expected))
            precision = relevant / len(retrieved_ids) if retrieved_ids else 0
            recall = relevant / len(expected) if expected else 0

            metrics.add_relevance_score(precision)

            results.append({
                'query': query,
                'precision': precision,
                'recall': recall,
                'time': t.duration,
                'retrieved': len(search_results)
            })

        # Save results
        summary = metrics.get_summary()
        output = {
            'summary': summary,
            'results': results
        }

        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Evaluation complete. Results saved to {args.output}")
        logger.info(f"Average retrieval time: {summary['avg_retrieval_time']:.4f}s")
        logger.info(f"Average relevance score: {summary['avg_relevance_score']:.4f}")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise


if __name__ == '__main__':
    main()
