#!/usr/bin/env python3
"""Run global search query."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.query import GlobalSearch, QueryProcessor, ContextBuilder
from src.generation import PromptBuilder, LLMClient, AnswerFormatter
from src.utils import setup_logger, Config


def main():
    parser = argparse.ArgumentParser(description='Run global search')
    parser.add_argument('query', help='Search query')
    parser.add_argument('--top-k', type=int, default=5, help='Number of communities')
    parser.add_argument('--no-generate', action='store_true', help='Skip answer generation')
    args = parser.parse_args()

    logger = setup_logger('global_search')
    config = Config()
    cfg = config.load()

    try:
        # Process query
        logger.info(f"Processing query: {args.query}")
        processor = QueryProcessor()
        query_data = processor.process(args.query)

        # Perform global search
        logger.info("Performing global search...")
        global_search = GlobalSearch()
        global_search.load()

        results = global_search.search(query_data['embedding'], args.top_k)

        # Display results
        print(f"\n{'='*60}")
        print(f"Global Search Results for: '{args.query}'")
        print(f"{'='*60}\n")

        for i, result in enumerate(results):
            print(f"[Community {result['community_id']}] Score: {result['score']:.4f}")
            print(f"Title: {result['title']}")
            print(f"Entities: {result['num_entities']}")
            print(f"Summary: {result['summary'][:300]}...")
            print()

        # Generate answer if requested
        if not args.no_generate:
            logger.info("Generating answer...")

            # Build context
            context_builder = ContextBuilder()
            context = context_builder.build_global_context(results)
            sources = context_builder.format_sources(results)

            # Build prompt
            prompt_builder = PromptBuilder()
            prompt = prompt_builder.build_global_prompt(args.query, context)

            # Generate
            llm = LLMClient(model=cfg['llm']['model'])
            answer = llm.generate(prompt)

            # Format
            formatter = AnswerFormatter()
            formatted = formatter.format(answer, sources)

            print(f"\n{'='*60}")
            print("Generated Answer:")
            print(f"{'='*60}\n")
            print(formatted['answer'])

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
