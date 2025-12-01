#!/usr/bin/env python3
"""Query script with evaluation metrics for Traditional RAG system."""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retrieval import RAGRetriever
from generation import LLMClient, PromptBuilder
from evaluation import SearchEvaluator
from utils import setup_logger, Config


def main():
    parser = argparse.ArgumentParser(description='Query Traditional RAG with evaluation metrics')
    parser.add_argument('query', help='Query string')
    parser.add_argument('--config', default='configs/rag_config.yaml', help='Config file')
    parser.add_argument('--top-k', type=int, help='Number of results to retrieve')
    parser.add_argument('--show-sources', action='store_true', help='Show source chunks')
    args = parser.parse_args()

    logger = setup_logger('query_eval')

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
        logger.info("✓ RAG index loaded")

        # Initialize LLM
        llm = LLMClient(
            model=cfg['llm']['model'],
            base_url=cfg['llm']['base_url']
        )
        prompt_builder = PromptBuilder()
        logger.info(f"✓ LLM initialized: {cfg['llm']['model']}")

        # Initialize evaluator
        evaluator = SearchEvaluator()
        logger.info("✓ Evaluator initialized")

        # Execute query
        print("\n" + "="*80)
        print(f"QUERY: {args.query}")
        print("="*80)

        # Step 1: Retrieve
        print("\n[1/3] Retrieving relevant chunks...")
        start = time.time()
        search_results = retriever.retrieve(args.query, top_k=top_k)
        retrieval_time = time.time() - start
        print(f"✓ Retrieved {len(search_results)} chunks in {retrieval_time:.3f}s")

        # Show sources if requested
        if args.show_sources:
            print("\n" + "-"*80)
            print("RETRIEVED SOURCES:")
            print("-"*80)
            for i, result in enumerate(search_results[:5], 1):  # Show top 5
                print(f"\n[{i}] Score: {result['score']:.4f}")
                print(f"Conversation: {result.get('conversation_id', 'unknown')}")
                print(f"Text: {result['text'][:200]}...")

        # Step 2: Generate answer
        print("\n[2/3] Generating answer...")
        context = retriever.get_context(search_results, max_tokens=2000)
        prompt = prompt_builder.build_health_prompt(args.query, context)

        start = time.time()
        answer = llm.generate(
            prompt,
            temperature=cfg['generation']['temperature'],
            max_tokens=cfg['generation']['max_tokens']
        )
        generation_time = time.time() - start
        print(f"✓ Answer generated in {generation_time:.3f}s")

        # Step 3: Evaluate
        print("\n[3/3] Evaluating response quality...")
        start = time.time()

        # Prepare search results for evaluation (need 'content' field)
        eval_results = []
        for r in search_results:
            eval_results.append({
                'content': r['text'],
                'score': r['score'],
                'type': 'chunk',
                'metadata': {
                    'conversation_id': r.get('conversation_id', 'unknown'),
                    'chunk_id': r.get('chunk_id', 'unknown')
                }
            })

        # Call evaluator with correct signature
        metrics = evaluator.evaluate(
            query=args.query,
            answer=answer,
            search_results=eval_results
        )
        eval_time = time.time() - start
        print(f"✓ Evaluation completed in {eval_time:.3f}s")

        # Display results
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(answer)

        print("\n" + "="*80)
        print("EVALUATION METRICS:")
        print("="*80)
        print(f"  1. Relevance Score:    {metrics['relevance_score']:.4f} (0-1)")
        print(f"     → How relevant is the answer to the query")
        print(f"\n  2. Coverage Score:     {metrics['coverage_score']:.4f} (0-1)")
        print(f"     → How comprehensively the answer covers the topic")
        print(f"\n  3. Answer Quality:     {metrics['answer_quality']:.4f} (0-1)")
        print(f"     → Overall quality of the answer (clarity, accuracy)")
        print(f"\n  4. Faithfulness Score: {metrics['faithfulness']:.4f} (0-1)")
        print(f"     → How faithful the answer is to the source context")

        # Average score
        avg_score = (
            metrics['relevance_score'] +
            metrics['coverage_score'] +
            metrics['answer_quality'] +
            metrics['faithfulness']
        ) / 4
        print(f"\n  → Average Score:       {avg_score:.4f} (0-1)")

        # Statistics
        print("\n" + "="*80)
        print("STATISTICS:")
        print("="*80)
        print(f"  Retrieval time:   {retrieval_time:.3f}s")
        print(f"  Generation time:  {generation_time:.3f}s")
        print(f"  Evaluation time:  {eval_time:.3f}s")
        print(f"  Total time:       {retrieval_time + generation_time + eval_time:.3f}s")
        print(f"  Chunks retrieved: {len(search_results)}")
        print(f"  Top score:        {search_results[0]['score']:.4f}")
        print("="*80 + "\n")

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
