#!/usr/bin/env python3
"""Run local search query."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query import LocalSearch, QueryProcessor, ContextBuilder
from src.generation import PromptBuilder, LLMClient, AnswerFormatter
from src.evaluation import SearchEvaluator
from src.utils import setup_logger, Config


def main():
    parser = argparse.ArgumentParser(description='Run local search')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--top-k', type=int, default=10, help='Number of results')
    parser.add_argument('--no-generate', action='store_true', help='Skip answer generation')
    parser.add_argument('--ground-truth', type=str, default=None, help='Ground truth answer for evaluation')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation metrics')
    args = parser.parse_args()

    logger = setup_logger('local_search')
    config = Config()
    cfg = config.load()

    try:
        # Process query
        logger.info(f"Processing query: {args.query}")
        processor = QueryProcessor()
        query_data = processor.process(args.query)

        # Perform local search
        logger.info("Performing local search...")
        local_search = LocalSearch()
        local_search.load()

        results = local_search.search(query_data['embedding'], args.top_k)

        # Display results
        print(f"\n{'='*60}")
        print(f"Local Search Results for: '{args.query}'")
        print(f"{'='*60}\n")

        for i, result in enumerate(results):
            print(f"[{i+1}] Score: {result['score']:.4f} | Type: {result['type']}")
            print(f"    {result['content'][:200]}...")
            print()

        # Generate answer if requested
        answer_text = ""
        if not args.no_generate:
            logger.info("Generating answer...")

            # Build context
            context_builder = ContextBuilder()
            context = context_builder.build_local_context(results)
            sources = context_builder.format_sources(results)

            # Build prompt
            prompt_builder = PromptBuilder()
            prompt = prompt_builder.build_local_prompt(args.query, context)

            # Generate
            llm = LLMClient(model=cfg['llm']['model'])
            answer = llm.generate(prompt)

            # Format
            formatter = AnswerFormatter()
            formatted = formatter.format(answer, sources)
            answer_text = formatted['answer']

            print(f"\n{'='*60}")
            print("Generated Answer:")
            print(f"{'='*60}\n")
            print(answer_text)

            if formatted['citations']:
                print(f"\nSources cited: {formatted['citations']}")

        # Evaluate results
        if not args.no_eval:
            logger.info("Evaluating search results...")
            evaluator = SearchEvaluator()

            # Use generated answer or empty string
            metrics = evaluator.evaluate(
                query=args.query,
                answer=answer_text if answer_text else "No answer generated",
                search_results=results,
                ground_truth=args.ground_truth
            )

            # Display metrics
            print(evaluator.get_summary())

            # Log metrics
            logger.info(f"Evaluation metrics: {metrics}")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
