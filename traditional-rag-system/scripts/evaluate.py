#!/usr/bin/env python3
"""Evaluate the Traditional RAG system."""

import sys
import json
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retrieval import RAGRetriever
from generation import LLMClient, PromptBuilder
from evaluation import SearchEvaluator
from utils import setup_logger, Config


def main():
    parser = argparse.ArgumentParser(description='Evaluate Traditional RAG system')
    parser.add_argument('--queries', required=True, help='Path to queries JSON file')
    parser.add_argument('--output', default='evaluation_results.json', help='Output file')
    parser.add_argument('--generate-answers', action='store_true', help='Generate answers for evaluation')
    parser.add_argument('--config', default='configs/rag_config.yaml', help='Config file')
    args = parser.parse_args()

    logger = setup_logger('evaluate')

    # Load config
    config = Config(args.config)
    cfg = config.load()

    try:
        # Load queries
        with open(args.queries) as f:
            queries = json.load(f)

        logger.info(f"Loaded {len(queries)} queries for evaluation")

        # Initialize components
        logger.info("Initializing RAG components...")

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

        # Initialize LLM if generating answers
        llm = None
        prompt_builder = None
        if args.generate_answers:
            llm = LLMClient(
                model=cfg['llm']['model'],
                base_url=cfg['llm']['base_url']
            )
            prompt_builder = PromptBuilder()
            logger.info(f"LLM initialized: {cfg['llm']['model']}")

        results = []

        # Evaluate each query
        for i, query_item in enumerate(queries, 1):
            query = query_item['query']
            ground_truth = query_item.get('ground_truth', None)

            logger.info(f"[{i}/{len(queries)}] Evaluating: {query}")

            # Retrieve
            start = time.time()
            search_results = retriever.retrieve(query, top_k=cfg['search']['top_k'])
            retrieval_time = time.time() - start

            # Generate answer if requested
            answer = ""
            if args.generate_answers and llm:
                context = retriever.get_context(search_results, max_tokens=2000)
                prompt = prompt_builder.build_health_prompt(query, context)
                answer = llm.generate(
                    prompt,
                    temperature=cfg['generation']['temperature'],
                    max_tokens=cfg['generation']['max_tokens']
                )

            # Evaluate
            evaluator = SearchEvaluator()
            metrics = evaluator.evaluate(
                query=query,
                answer=answer if answer else "No answer generated",
                search_results=search_results,
                ground_truth=ground_truth
            )

            # Store result
            eval_result = {
                'query': query,
                'has_ground_truth': ground_truth is not None,
                'retrieval_time': retrieval_time,
                'num_results': len(search_results),
                'top_score': search_results[0]['score'] if search_results else 0,
                'metrics': metrics,
                'answer_generated': bool(answer)
            }

            results.append(eval_result)

            # Print progress
            logger.info(f"  Relevance: {metrics['relevance_score']:.4f}, "
                       f"Quality: {metrics['answer_quality']:.4f}")

        # Calculate summary statistics
        all_metrics = [r['metrics'] for r in results]

        summary = {
            'total_queries': len(results),
            'answers_generated': args.generate_answers,
            'avg_time': sum(r['retrieval_time'] for r in results) / len(results),
            'avg_relevance_score': sum(m['relevance_score'] for m in all_metrics) / len(all_metrics),
            'avg_coverage_score': sum(m['coverage_score'] for m in all_metrics) / len(all_metrics),
            'avg_answer_quality': sum(m['answer_quality'] for m in all_metrics) / len(all_metrics),
            'avg_faithfulness': sum(m['faithfulness'] for m in all_metrics) / len(all_metrics),
            'overall_score': sum(
                sum(m.values()) / len(m) for m in all_metrics
            ) / len(all_metrics)
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
        logger.info(f"\nTRADITIONAL RAG PERFORMANCE:")
        logger.info(f"  Avg Time: {summary['avg_time']:.4f}s")
        logger.info(f"  Avg Relevance Score: {summary['avg_relevance_score']:.4f}")
        logger.info(f"  Avg Coverage Score: {summary['avg_coverage_score']:.4f}")
        logger.info(f"  Avg Answer Quality: {summary['avg_answer_quality']:.4f}")
        logger.info(f"  Avg Faithfulness: {summary['avg_faithfulness']:.4f}")
        logger.info(f"  Overall Score: {summary['overall_score']:.4f}")
        logger.info(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
