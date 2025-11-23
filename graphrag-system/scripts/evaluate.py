#!/usr/bin/env python3
"""Evaluate the GraphRAG system with comprehensive metrics."""

import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query import LocalSearch, GlobalSearch, QueryProcessor, ContextBuilder
from src.generation import LLMClient, PromptBuilder, AnswerFormatter
from src.evaluation import SearchEvaluator
from src.utils import setup_logger, Config


def main():
    parser = argparse.ArgumentParser(description='Evaluate GraphRAG system with quality metrics')
    parser.add_argument('--queries', required=True, help='Path to queries JSON file (with optional ground_truth)')
    parser.add_argument('--output', default='evaluation_results.json', help='Output file')
    parser.add_argument('--search-type', choices=['local', 'global', 'both'], default='both')
    parser.add_argument('--generate-answers', action='store_true', help='Generate answers for evaluation')
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
        context_builder = ContextBuilder()
        prompt_builder = PromptBuilder()

        local_search.load()
        global_search.load()

        # Initialize LLM if generating answers
        llm = None
        if args.generate_answers:
            llm = LLMClient(model=cfg['llm']['model'])
            formatter = AnswerFormatter()

        results = []

        for i, query_item in enumerate(queries, 1):
            query = query_item['query']
            ground_truth = query_item.get('ground_truth', None)

            logger.info(f"[{i}/{len(queries)}] Evaluating: {query}")

            # Process query
            query_data = processor.process(query)

            eval_result = {
                'query': query,
                'query_type': query_data['type'],
                'has_ground_truth': ground_truth is not None
            }

            # Local search
            if args.search_type in ['local', 'both']:
                start = time.time()
                local_results = local_search.search(query_data['embedding'], top_k=10)
                local_time = time.time() - start

                # Generate answer if requested
                local_answer = ""
                if args.generate_answers and llm:
                    context = context_builder.build_local_context(local_results)
                    prompt = prompt_builder.build_local_prompt(query, context)
                    answer = llm.generate(prompt)
                    sources = context_builder.format_sources(local_results)
                    formatted = formatter.format(answer, sources)
                    local_answer = formatted['answer']

                # Evaluate with SearchEvaluator
                evaluator = SearchEvaluator()
                metrics = evaluator.evaluate(
                    query=query,
                    answer=local_answer if local_answer else "No answer generated",
                    search_results=local_results,
                    ground_truth=ground_truth
                )

                eval_result['local'] = {
                    'time': local_time,
                    'num_results': len(local_results),
                    'top_score': local_results[0]['score'] if local_results else 0,
                    'metrics': metrics,
                    'answer_generated': bool(local_answer)
                }

            # Global search
            if args.search_type in ['global', 'both']:
                start = time.time()
                global_results = global_search.search(query_data['embedding'], top_k=5)
                global_time = time.time() - start

                # Generate answer if requested
                global_answer = ""
                if args.generate_answers and llm:
                    context = context_builder.build_global_context(global_results)
                    prompt = prompt_builder.build_global_prompt(query, context)
                    answer = llm.generate(prompt)
                    sources = context_builder.format_sources(global_results)
                    formatted = formatter.format(answer, sources)
                    global_answer = formatted['answer']

                # Evaluate with SearchEvaluator
                evaluator = SearchEvaluator()
                metrics = evaluator.evaluate(
                    query=query,
                    answer=global_answer if global_answer else "No answer generated",
                    search_results=global_results,
                    ground_truth=ground_truth
                )

                eval_result['global'] = {
                    'time': global_time,
                    'num_results': len(global_results),
                    'top_score': global_results[0]['score'] if global_results else 0,
                    'metrics': metrics,
                    'answer_generated': bool(global_answer)
                }

            results.append(eval_result)

            # Print progress
            if args.search_type in ['local', 'both']:
                logger.info(f"  Local  - Relevance: {eval_result['local']['metrics']['relevance_score']:.4f}, "
                           f"Quality: {eval_result['local']['metrics']['answer_quality']:.4f}")
            if args.search_type in ['global', 'both']:
                logger.info(f"  Global - Relevance: {eval_result['global']['metrics']['relevance_score']:.4f}, "
                           f"Quality: {eval_result['global']['metrics']['answer_quality']:.4f}")

        # Calculate summary stats
        summary = {
            'total_queries': len(results),
            'search_type': args.search_type,
            'answers_generated': args.generate_answers
        }

        # Calculate average metrics for local search
        if args.search_type in ['local', 'both']:
            local_metrics = [r['local']['metrics'] for r in results if 'local' in r]
            summary['local'] = {
                'avg_time': sum(r['local']['time'] for r in results if 'local' in r) / len(results),
                'avg_relevance_score': sum(m['relevance_score'] for m in local_metrics) / len(local_metrics),
                'avg_coverage_score': sum(m['coverage_score'] for m in local_metrics) / len(local_metrics),
                'avg_answer_quality': sum(m['answer_quality'] for m in local_metrics) / len(local_metrics),
                'avg_faithfulness': sum(m['faithfulness'] for m in local_metrics) / len(local_metrics),
                'overall_score': sum(
                    sum(m.values()) / len(m) for m in local_metrics
                ) / len(local_metrics)
            }

        # Calculate average metrics for global search
        if args.search_type in ['global', 'both']:
            global_metrics = [r['global']['metrics'] for r in results if 'global' in r]
            summary['global'] = {
                'avg_time': sum(r['global']['time'] for r in results if 'global' in r) / len(results),
                'avg_relevance_score': sum(m['relevance_score'] for m in global_metrics) / len(global_metrics),
                'avg_coverage_score': sum(m['coverage_score'] for m in global_metrics) / len(global_metrics),
                'avg_answer_quality': sum(m['answer_quality'] for m in global_metrics) / len(global_metrics),
                'avg_faithfulness': sum(m['faithfulness'] for m in global_metrics) / len(global_metrics),
                'overall_score': sum(
                    sum(m.values()) / len(m) for m in global_metrics
                ) / len(global_metrics)
            }

        output = {
            'summary': summary,
            'results': results
        }

        # Save results
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Results saved to: {args.output}")
        logger.info(f"Total queries evaluated: {len(results)}")

        if args.search_type in ['local', 'both']:
            logger.info(f"\nLOCAL SEARCH PERFORMANCE:")
            logger.info(f"  Avg Time: {summary['local']['avg_time']:.4f}s")
            logger.info(f"  Avg Relevance Score: {summary['local']['avg_relevance_score']:.4f}")
            logger.info(f"  Avg Coverage Score: {summary['local']['avg_coverage_score']:.4f}")
            logger.info(f"  Avg Answer Quality: {summary['local']['avg_answer_quality']:.4f}")
            logger.info(f"  Avg Faithfulness: {summary['local']['avg_faithfulness']:.4f}")
            logger.info(f"  Overall Score: {summary['local']['overall_score']:.4f}")

        if args.search_type in ['global', 'both']:
            logger.info(f"\nGLOBAL SEARCH PERFORMANCE:")
            logger.info(f"  Avg Time: {summary['global']['avg_time']:.4f}s")
            logger.info(f"  Avg Relevance Score: {summary['global']['avg_relevance_score']:.4f}")
            logger.info(f"  Avg Coverage Score: {summary['global']['avg_coverage_score']:.4f}")
            logger.info(f"  Avg Answer Quality: {summary['global']['avg_answer_quality']:.4f}")
            logger.info(f"  Avg Faithfulness: {summary['global']['avg_faithfulness']:.4f}")
            logger.info(f"  Overall Score: {summary['global']['overall_score']:.4f}")

        logger.info(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
