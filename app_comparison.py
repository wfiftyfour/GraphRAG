#!/usr/bin/env python3
"""
RAG vs GraphRAG Comparison Interface
Compare Traditional RAG and GraphRAG side-by-side with the same query
"""

import sys
import os
from pathlib import Path
import time
import json
import importlib.util

# Pre-load heavy dependencies
print("Loading dependencies (this may take 20-30 seconds)...", flush=True)
import torch
print("  - torch loaded", flush=True)
from sentence_transformers import SentenceTransformer
print("  - sentence_transformers loaded", flush=True)
import gradio as gr
print("  - gradio loaded", flush=True)

# Add both systems to path
graphrag_root = Path(__file__).parent / 'graphrag-system'
trad_rag_root = Path(__file__).parent / 'traditional-rag-system'

# Temporarily change directory to graphrag-system to import
original_dir = Path.cwd()
os.chdir(graphrag_root)
sys.path.insert(0, str(graphrag_root))

# Import GraphRAG components
print("Importing GraphRAG components...", flush=True)
from src.query import LocalSearch, GlobalSearch
from src.generation import PromptBuilder as GraphPromptBuilder
from src.generation import LLMClient as GraphLLMClient
from src.evaluation import SearchEvaluator
from src.utils import setup_logger, Config as GraphConfig

# Change back to original directory
os.chdir(original_dir)

# Import Traditional RAG components
print("Importing Traditional RAG components...", flush=True)
sys.path.insert(0, str(trad_rag_root / 'src'))
from retrieval import RAGRetriever
from generation import LLMClient as RAGLLMClient
from generation import PromptBuilder as RAGPromptBuilder
from utils import Config as RAGConfig

# Initialize logger
logger = setup_logger('comparison_app')
print("Imports successful!", flush=True)

# Global components
graphrag_local = None
graphrag_global = None
rag_retriever = None
graph_llm = None
rag_llm = None
graph_prompt = None
rag_prompt = None
evaluator = None


def initialize_systems():
    """Initialize both RAG and GraphRAG systems."""
    global graphrag_local, graphrag_global, rag_retriever
    global graph_llm, rag_llm, graph_prompt, rag_prompt, evaluator

    try:
        logger.info("="*60)
        logger.info("INITIALIZING COMPARISON SYSTEMS")
        logger.info("="*60)

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
        graphrag_local.load(load_entities=False, load_graph=False)  # Load chunk embeddings only

        graphrag_global = GlobalSearch(data_dir=str(graphrag_dir / 'data'))
        logger.info("  Loading GraphRAG global search community data...")
        graphrag_global.load()  # Load community embeddings and reports

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

        logger.info("\n" + "="*60)
        logger.info("INITIALIZATION COMPLETE")
        logger.info("="*60)
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")

        return "System ready for comparison!"

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Initialization failed: {str(e)}"


def compare_systems(query: str, use_graphrag_local: bool = True, use_graphrag_global: bool = False):
    """
    Run the same query on both systems and compare results.

    Args:
        query: User query
        use_graphrag_local: Use GraphRAG local search
        use_graphrag_global: Use GraphRAG global search

    Returns:
        Formatted comparison results
    """
    if not query.strip():
        return "Please enter a query", "", "", ""

    try:
        results = {
            'query': query,
            'traditional_rag': {},
            'graphrag_local': {},
            'graphrag_global': {}
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"QUERY: {query}")
        logger.info(f"{'='*60}")

        # ===== TRADITIONAL RAG =====
        logger.info("\n[1/3] Running Traditional RAG...")
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

        # Calculate overall score
        rag_metrics['overall_score'] = sum(rag_metrics.values()) / len(rag_metrics)

        results['traditional_rag'] = {
            'answer': rag_answer,
            'retrieval_time': rag_retrieval_time,
            'generation_time': rag_generation_time,
            'total_time': rag_total_time,
            'num_chunks': len(rag_search_results),
            'top_score': rag_search_results[0]['score'] if rag_search_results else 0,
            'metrics': rag_metrics
        }

        logger.info(f"  Completed in {rag_total_time:.2f}s")

        # ===== GRAPHRAG LOCAL =====
        if use_graphrag_local:
            logger.info("\n[2/3] Running GraphRAG Local Search...")
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

            # Calculate overall score
            local_metrics['overall_score'] = sum(local_metrics.values()) / len(local_metrics)

            results['graphrag_local'] = {
                'answer': local_answer,
                'search_time': local_search_time,
                'generation_time': local_generation_time,
                'total_time': local_total_time,
                'num_results': len(local_results),
                'metrics': local_metrics
            }

            logger.info(f"  Completed in {local_total_time:.2f}s")

        # ===== GRAPHRAG GLOBAL =====
        if use_graphrag_global:
            logger.info("\n[3/3] Running GraphRAG Global Search...")
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
            # Keep the original results with metadata and entities!
            global_results_for_eval = []
            for result in global_results:
                eval_result = result.copy()
                # Add 'content' field from summary for evaluator
                eval_result['content'] = result.get('summary', '')
                global_results_for_eval.append(eval_result)

            global_metrics = evaluator.evaluate(
                query=query,
                answer=global_answer,
                search_results=global_results_for_eval
            )

            # Calculate overall score
            global_metrics['overall_score'] = sum(global_metrics.values()) / len(global_metrics)

            results['graphrag_global'] = {
                'answer': global_answer,
                'search_time': global_search_time,
                'generation_time': global_generation_time,
                'total_time': global_total_time,
                'metrics': global_metrics
            }

            logger.info(f"  Completed in {global_total_time:.2f}s")

        # Format output
        return format_comparison_results(results, use_graphrag_local, use_graphrag_global)

    except Exception as e:
        logger.error(f"Query error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"Error: {str(e)}", "", "", ""


def format_comparison_results(results, use_local, use_global):
    """Format results for display in Gradio."""

    # Summary
    summary = f"# Query Comparison Results\n\n"
    summary += f"**Query:** {results['query']}\n\n"
    summary += "## Performance Summary\n\n"
    summary += "| System | Total Time | Retrieval/Search | Generation | Score |\n"
    summary += "|--------|-----------|------------------|------------|-------|\n"

    rag = results['traditional_rag']
    summary += f"| Traditional RAG | {rag['total_time']:.2f}s | {rag['retrieval_time']:.2f}s | {rag['generation_time']:.2f}s | {rag['metrics'].get('overall_score', 0):.3f} |\n"

    if use_local and results['graphrag_local']:
        local = results['graphrag_local']
        summary += f"| GraphRAG Local | {local['total_time']:.2f}s | {local['search_time']:.2f}s | {local['generation_time']:.2f}s | {local['metrics'].get('overall_score', 0):.3f} |\n"

    if use_global and results['graphrag_global']:
        glob = results['graphrag_global']
        summary += f"| GraphRAG Global | {glob['total_time']:.2f}s | {glob['search_time']:.2f}s | {glob['generation_time']:.2f}s | {glob['metrics'].get('overall_score', 0):.3f} |\n"

    # Traditional RAG Output
    rag_output = "# Traditional RAG\n\n"
    rag_output += f"**Answer:**\n\n{rag['answer']}\n\n"
    rag_output += "---\n\n"
    rag_output += f"**Metrics:**\n"
    rag_output += f"- Relevance: {rag['metrics'].get('relevance_score', 0):.3f}\n"
    rag_output += f"- Coverage: {rag['metrics'].get('coverage_score', 0):.3f}\n"
    rag_output += f"- Quality: {rag['metrics'].get('answer_quality', 0):.3f}\n"
    rag_output += f"- Faithfulness: {rag['metrics'].get('faithfulness', 0):.3f}\n"
    rag_output += f"- Chunks Retrieved: {rag['num_chunks']}\n"
    rag_output += f"- Top Similarity: {rag['top_score']:.3f}\n"

    # GraphRAG Local Output
    local_output = ""
    if use_local and results['graphrag_local']:
        local = results['graphrag_local']
        local_output = "# GraphRAG Local Search\n\n"
        local_output += f"**Answer:**\n\n{local['answer']}\n\n"
        local_output += "---\n\n"
        local_output += f"**Metrics:**\n"
        local_output += f"- Relevance: {local['metrics'].get('relevance_score', 0):.3f}\n"
        local_output += f"- Coverage: {local['metrics'].get('coverage_score', 0):.3f}\n"
        local_output += f"- Quality: {local['metrics'].get('answer_quality', 0):.3f}\n"
        local_output += f"- Faithfulness: {local['metrics'].get('faithfulness', 0):.3f}\n"
        local_output += f"- Results Used: {local['num_results']}\n"

    # GraphRAG Global Output
    global_output = ""
    if use_global and results['graphrag_global']:
        glob = results['graphrag_global']
        global_output = "# GraphRAG Global Search\n\n"
        global_output += f"**Answer:**\n\n{glob['answer']}\n\n"
        global_output += "---\n\n"
        global_output += f"**Metrics:**\n"
        global_output += f"- Relevance: {glob['metrics'].get('relevance_score', 0):.3f}\n"
        global_output += f"- Coverage: {glob['metrics'].get('coverage_score', 0):.3f}\n"
        global_output += f"- Quality: {glob['metrics'].get('answer_quality', 0):.3f}\n"
        global_output += f"- Faithfulness: {glob['metrics'].get('faithfulness', 0):.3f}\n"

    return summary, rag_output, local_output, global_output


# Create Gradio Interface
with gr.Blocks(title="RAG vs GraphRAG Comparison") as demo:
    gr.Markdown("# RAG vs GraphRAG Side-by-Side Comparison")
    gr.Markdown("Compare Traditional RAG and GraphRAG systems with the same query")

    with gr.Row():
        with gr.Column():
            init_btn = gr.Button("Initialize Systems", variant="primary")
            init_status = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        query_input = gr.Textbox(
            label="Enter your query",
            placeholder="e.g., What supplements are recommended for males?",
            lines=2
        )

    with gr.Row():
        use_local = gr.Checkbox(label="Use GraphRAG Local Search", value=True)
        use_global = gr.Checkbox(label="Use GraphRAG Global Search", value=False)

    compare_btn = gr.Button("Compare Systems", variant="primary", size="lg")

    with gr.Row():
        summary_output = gr.Markdown(label="Summary")

    with gr.Row():
        with gr.Column():
            rag_output = gr.Markdown(label="Traditional RAG")
        with gr.Column():
            graphrag_local_output = gr.Markdown(label="GraphRAG Local")
        with gr.Column():
            graphrag_global_output = gr.Markdown(label="GraphRAG Global")

    # Event handlers
    init_btn.click(
        fn=initialize_systems,
        outputs=init_status
    )

    compare_btn.click(
        fn=compare_systems,
        inputs=[query_input, use_local, use_global],
        outputs=[summary_output, rag_output, graphrag_local_output, graphrag_global_output]
    )

    # Example queries
    gr.Examples(
        examples=[
            ["What supplements are recommended for males?"],
            ["How does age affect calcium requirements?"],
            ["What is the recommended protein intake for maintaining muscle health?"],
            ["What are normal testosterone levels for adult men?"],
        ],
        inputs=query_input
    )

if __name__ == "__main__":
    logger.info("Starting RAG vs GraphRAG Comparison Interface")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=True
    )
