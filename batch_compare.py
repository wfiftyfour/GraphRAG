#!/usr/bin/env python3
"""
Batch Comparison Script
Run multiple queries from a file and save metrics to a text file
No answers are saved, only performance metrics
"""

import sys
import os
from pathlib import Path
import time
import argparse
from datetime import datetime

# Pre-load heavy dependencies
print("Loading dependencies...", flush=True)
import torch
from sentence_transformers import SentenceTransformer

# Add both systems to path
graphrag_root = Path(__file__).parent / 'graphrag-system'
trad_rag_root = Path(__file__).parent / 'traditional-rag-system'

# Temporarily change directory to graphrag-system to import
original_dir = Path.cwd()
os.chdir(graphrag_root)
sys.path.insert(0, str(graphrag_root))

# Import GraphRAG components
from src.query import LocalSearch, GlobalSearch
from src.generation import PromptBuilder as GraphPromptBuilder
from src.generation import LLMClient as GraphLLMClient
from src.evaluation import SearchEvaluator
from src.utils import setup_logger, Config as GraphConfig

# Change back to original directory
os.chdir(original_dir)

# Import Traditional RAG components
sys.path.insert(0, str(trad_rag_root / 'src'))
from retrieval import RAGRetriever
from generation import LLMClient as RAGLLMClient
from generation import PromptBuilder as RAGPromptBuilder
from utils import Config as RAGConfig

# Initialize logger
logger = setup_logger('batch_comparison')
print("Imports successful!", flush=True)


def initialize_systems():
    """Initialize both RAG and GraphRAG systems."""
    logger.info("="*80)
    logger.info("INITIALIZING COMPARISON SYSTEMS")
    logger.info("="*80)

    # Get absolute paths
    root_dir = Path(__file__).parent
    graphrag_dir = root_dir / 'graphrag-system'
    trad_rag_dir = root_dir / 'traditional-rag-system'

    # Initialize GraphRAG
    logger.info("\n[1/2] Initializing GraphRAG...")
    graph_config = GraphConfig(str(graphrag_dir / 'configs'))
    graph_cfg = graph_config.load('graphrag_config.yaml')

    # Initialize and LOAD GraphRAG data
    graphrag_local = LocalSearch(data_dir=str(graphrag_dir / 'data'))
    logger.info("  Loading GraphRAG local search embeddings...")
    graphrag_local.load(load_entities=False, load_graph=False)

    graphrag_global = GlobalSearch(data_dir=str(graphrag_dir / 'data'))
    logger.info("  Loading GraphRAG global search community data...")
    graphrag_global.load()

    graph_llm = GraphLLMClient(model=graph_cfg['llm']['model'])
    graph_prompt = GraphPromptBuilder()
    logger.info("  GraphRAG initialized and data loaded")

    # Initialize Traditional RAG
    logger.info("\n[2/2] Initializing Traditional RAG...")
    rag_config = RAGConfig(str(trad_rag_dir / 'configs' / 'rag_config.yaml'))
    rag_cfg = rag_config.load()

    # Load RAG index
    processed_dir = trad_rag_dir / rag_cfg['data']['processed_dir']
    index_path = processed_dir / 'embeddings' / 'faiss_index.bin'
    chunks_path = processed_dir / 'chunks' / 'chunks.json'

    rag_retriever = RAGRetriever(rag_cfg)
    rag_retriever.load(
        str(index_path),
        str(chunks_path),
        rag_cfg['embedding']['model']
    )
    rag_llm = RAGLLMClient(
        model=rag_cfg['llm']['model'],
        base_url=rag_cfg['llm']['base_url']
    )
    rag_prompt = RAGPromptBuilder()
    logger.info("  Traditional RAG initialized")

    # Initialize evaluator
    evaluator = SearchEvaluator()

    logger.info("\n" + "="*80)
    logger.info("INITIALIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")

    return {
        'graphrag_local': graphrag_local,
        'graphrag_global': graphrag_global,
        'rag_retriever': rag_retriever,
        'graph_llm': graph_llm,
        'rag_llm': rag_llm,
        'graph_prompt': graph_prompt,
        'rag_prompt': rag_prompt,
        'evaluator': evaluator
    }


def run_query(query, systems, run_local=True, run_global=True):
    """
    Run a single query on all systems and return metrics only.

    Args:
        query: The query string
        systems: Dictionary of initialized system components
        run_local: Whether to run GraphRAG local search
        run_global: Whether to run GraphRAG global search

    Returns:
        Dictionary with metrics for each system
    """
    graphrag_local = systems['graphrag_local']
    graphrag_global = systems['graphrag_global']
    rag_retriever = systems['rag_retriever']
    graph_llm = systems['graph_llm']
    rag_llm = systems['rag_llm']
    graph_prompt = systems['graph_prompt']
    rag_prompt = systems['rag_prompt']
    evaluator = systems['evaluator']

    results = {
        'query': query,
        'traditional_rag': None,
        'graphrag_local': None,
        'graphrag_global': None
    }

    # ===== TRADITIONAL RAG =====
    logger.info(f"\n  Running Traditional RAG...")
    rag_start = time.time()

    # Retrieve
    rag_search_results = rag_retriever.retrieve(query, top_k=15)
    rag_retrieval_time = time.time() - rag_start

    # Generate answer
    context = rag_retriever.get_context(rag_search_results, max_tokens=2000)
    prompt = rag_prompt.build_health_prompt(query, context)

    gen_start = time.time()
    rag_answer = rag_llm.generate(prompt, temperature=0.4, max_tokens=1536)
    rag_generation_time = time.time() - gen_start

    rag_total_time = time.time() - rag_start

    # Evaluate
    rag_metrics = evaluator.evaluate(
        query=query,
        answer=rag_answer,
        search_results=rag_search_results
    )
    rag_metrics['overall_score'] = sum(rag_metrics.values()) / len(rag_metrics)

    results['traditional_rag'] = {
        'retrieval_time': rag_retrieval_time,
        'generation_time': rag_generation_time,
        'total_time': rag_total_time,
        'num_chunks': len(rag_search_results),
        'metrics': rag_metrics
    }
    logger.info(f"    Completed in {rag_total_time:.2f}s")

    # ===== GRAPHRAG LOCAL =====
    if run_local:
        logger.info(f"  Running GraphRAG Local...")
        local_start = time.time()

        # Embed query and search
        query_embedding = rag_retriever.embedder.embed_query(query)
        local_results = graphrag_local.search(query_embedding, top_k=20)
        local_search_time = time.time() - local_start

        # Format context from results
        local_context = "\n\n".join([
            f"[Source {i+1}] {result['content']}"
            for i, result in enumerate(local_results)
        ])

        # Generate
        local_prompt = graph_prompt.build_local_prompt(query, local_context)
        gen_start = time.time()
        local_answer = graph_llm.generate(local_prompt, temperature=0.4, max_tokens=1536)
        local_generation_time = time.time() - gen_start

        local_total_time = time.time() - local_start

        # Evaluate
        local_metrics = evaluator.evaluate(
            query=query,
            answer=local_answer,
            search_results=local_results
        )
        local_metrics['overall_score'] = sum(local_metrics.values()) / len(local_metrics)

        results['graphrag_local'] = {
            'search_time': local_search_time,
            'generation_time': local_generation_time,
            'total_time': local_total_time,
            'num_results': len(local_results),
            'metrics': local_metrics
        }
        logger.info(f"    Completed in {local_total_time:.2f}s")

    # ===== GRAPHRAG GLOBAL =====
    if run_global:
        logger.info(f"  Running GraphRAG Global...")
        global_start = time.time()

        # Embed query and search communities
        query_embedding = rag_retriever.embedder.embed_query(query)
        global_results = graphrag_global.search(query_embedding, top_k=5)
        global_search_time = time.time() - global_start

        # Format context from community summaries
        global_context = "\n\n".join([
            f"Community {result.get('community_id', i)}: {result.get('summary', '')}"
            for i, result in enumerate(global_results)
        ])

        # Generate
        global_prompt = graph_prompt.build_global_prompt(query, global_context)
        gen_start = time.time()
        global_answer = graph_llm.generate(global_prompt, temperature=0.4, max_tokens=1536)
        global_generation_time = time.time() - gen_start

        global_total_time = time.time() - global_start

        # Evaluate - Add 'content' field to global results for evaluation
        global_results_for_eval = []
        for result in global_results:
            eval_result = result.copy()
            eval_result['content'] = result.get('summary', '')
            global_results_for_eval.append(eval_result)

        global_metrics = evaluator.evaluate(
            query=query,
            answer=global_answer,
            search_results=global_results_for_eval
        )
        global_metrics['overall_score'] = sum(global_metrics.values()) / len(global_metrics)

        results['graphrag_global'] = {
            'search_time': global_search_time,
            'generation_time': global_generation_time,
            'total_time': global_total_time,
            'metrics': global_metrics
        }
        logger.info(f"    Completed in {global_total_time:.2f}s")

    return results


def format_metrics_output(all_results):
    """
    Format all query results into a readable metrics file.

    Args:
        all_results: List of result dictionaries from run_query

    Returns:
        Formatted string for metrics.txt
    """
    output = []
    output.append("="*80)
    output.append("BATCH COMPARISON METRICS")
    output.append("="*80)
    output.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output.append(f"Total Queries: {len(all_results)}")
    output.append("="*80)
    output.append("")

    for i, result in enumerate(all_results, 1):
        query = result['query']
        output.append(f"\n{'='*80}")
        output.append(f"Query {i}: {query}")
        output.append('='*80)
        output.append("")

        # Traditional RAG
        if result['traditional_rag']:
            rag = result['traditional_rag']
            output.append("TRADITIONAL RAG:")
            output.append(f"  Retrieval Time:    {rag['retrieval_time']:.3f}s")
            output.append(f"  Generation Time:   {rag['generation_time']:.3f}s")
            output.append(f"  Total Time:        {rag['total_time']:.3f}s")
            output.append(f"  Chunks Retrieved:  {rag['num_chunks']}")
            output.append(f"  Metrics:")
            output.append(f"    - Relevance:     {rag['metrics']['relevance_score']:.4f}")
            output.append(f"    - Coverage:      {rag['metrics']['coverage_score']:.4f}")
            output.append(f"    - Quality:       {rag['metrics']['answer_quality']:.4f}")
            output.append(f"    - Faithfulness:  {rag['metrics']['faithfulness']:.4f}")
            output.append(f"    - Overall Score: {rag['metrics']['overall_score']:.4f}")
            output.append("")

        # GraphRAG Local
        if result['graphrag_local']:
            local = result['graphrag_local']
            output.append("GRAPHRAG LOCAL:")
            output.append(f"  Search Time:       {local['search_time']:.3f}s")
            output.append(f"  Generation Time:   {local['generation_time']:.3f}s")
            output.append(f"  Total Time:        {local['total_time']:.3f}s")
            output.append(f"  Results Used:      {local['num_results']}")
            output.append(f"  Metrics:")
            output.append(f"    - Relevance:     {local['metrics']['relevance_score']:.4f}")
            output.append(f"    - Coverage:      {local['metrics']['coverage_score']:.4f}")
            output.append(f"    - Quality:       {local['metrics']['answer_quality']:.4f}")
            output.append(f"    - Faithfulness:  {local['metrics']['faithfulness']:.4f}")
            output.append(f"    - Overall Score: {local['metrics']['overall_score']:.4f}")
            output.append("")

        # GraphRAG Global
        if result['graphrag_global']:
            glob = result['graphrag_global']
            output.append("GRAPHRAG GLOBAL:")
            output.append(f"  Search Time:       {glob['search_time']:.3f}s")
            output.append(f"  Generation Time:   {glob['generation_time']:.3f}s")
            output.append(f"  Total Time:        {glob['total_time']:.3f}s")
            output.append(f"  Metrics:")
            output.append(f"    - Relevance:     {glob['metrics']['relevance_score']:.4f}")
            output.append(f"    - Coverage:      {glob['metrics']['coverage_score']:.4f}")
            output.append(f"    - Quality:       {glob['metrics']['answer_quality']:.4f}")
            output.append(f"    - Faithfulness:  {glob['metrics']['faithfulness']:.4f}")
            output.append(f"    - Overall Score: {glob['metrics']['overall_score']:.4f}")
            output.append("")

    # Summary statistics
    output.append("\n" + "="*80)
    output.append("SUMMARY STATISTICS")
    output.append("="*80)
    output.append("")

    # Calculate averages for each system
    for system_name in ['traditional_rag', 'graphrag_local', 'graphrag_global']:
        system_results = [r[system_name] for r in all_results if r[system_name]]

        if not system_results:
            continue

        display_name = system_name.replace('_', ' ').upper()
        output.append(f"{display_name}:")

        avg_total_time = sum(r['total_time'] for r in system_results) / len(system_results)
        avg_relevance = sum(r['metrics']['relevance_score'] for r in system_results) / len(system_results)
        avg_coverage = sum(r['metrics']['coverage_score'] for r in system_results) / len(system_results)
        avg_quality = sum(r['metrics']['answer_quality'] for r in system_results) / len(system_results)
        avg_faithfulness = sum(r['metrics']['faithfulness'] for r in system_results) / len(system_results)
        avg_overall = sum(r['metrics']['overall_score'] for r in system_results) / len(system_results)

        output.append(f"  Avg Total Time:    {avg_total_time:.3f}s")
        output.append(f"  Avg Relevance:     {avg_relevance:.4f}")
        output.append(f"  Avg Coverage:      {avg_coverage:.4f}")
        output.append(f"  Avg Quality:       {avg_quality:.4f}")
        output.append(f"  Avg Faithfulness:  {avg_faithfulness:.4f}")
        output.append(f"  Avg Overall Score: {avg_overall:.4f}")
        output.append("")

    output.append("="*80)
    output.append("END OF REPORT")
    output.append("="*80)

    return "\n".join(output)


def load_queries(queries_file):
    """Load queries from a text file, ignoring comments and empty lines."""
    queries = []
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                queries.append(line)
    return queries


def main():
    parser = argparse.ArgumentParser(description='Run batch comparison of RAG systems')
    parser.add_argument('--queries', '-q', default='queries.txt',
                        help='Path to queries file (default: queries.txt)')
    parser.add_argument('--output', '-o', default='metrics.txt',
                        help='Path to output metrics file (default: metrics.txt)')
    parser.add_argument('--skip-local', action='store_true',
                        help='Skip GraphRAG local search')
    parser.add_argument('--skip-global', action='store_true',
                        help='Skip GraphRAG global search')

    args = parser.parse_args()

    # Load queries
    queries_path = Path(args.queries)
    if not queries_path.exists():
        print(f"Error: Queries file not found: {queries_path}")
        sys.exit(1)

    queries = load_queries(queries_path)
    print(f"Loaded {len(queries)} queries from {queries_path}")

    if not queries:
        print("Error: No queries found in file")
        sys.exit(1)

    # Initialize systems
    print("\nInitializing systems...")
    systems = initialize_systems()

    # Run all queries
    print(f"\nProcessing {len(queries)} queries...")
    all_results = []

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(queries)}: {query}")
        print('='*80)

        result = run_query(
            query,
            systems,
            run_local=not args.skip_local,
            run_global=not args.skip_global
        )
        all_results.append(result)

    # Format and save metrics
    print(f"\n\nFormatting metrics...")
    metrics_output = format_metrics_output(all_results)

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(metrics_output)

    print(f"\nMetrics saved to: {output_path}")
    print(f"Processed {len(queries)} queries successfully!")


if __name__ == "__main__":
    main()
