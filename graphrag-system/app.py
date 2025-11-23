#!/usr/bin/env python3
"""
GraphRAG Web Interface
A user-friendly GUI for running local and global search with evaluation metrics.
"""

import sys
from pathlib import Path

# IMPORTANT: Import heavy dependencies BEFORE Gradio to avoid threading issues
print("Loading dependencies (this may take 20-30 seconds)...", flush=True)
sys.stdout.flush()

import torch  # ~1 second
print("  - torch loaded", flush=True)

from sentence_transformers import SentenceTransformer  # ~23 seconds
print("  - sentence_transformers loaded", flush=True)

import gradio as gr
print("  - gradio loaded", flush=True)

sys.path.insert(0, str(Path(__file__).parent))

from src.query import LocalSearch, GlobalSearch, QueryProcessor, ContextBuilder
from src.generation import PromptBuilder, LLMClient, AnswerFormatter
from src.evaluation import SearchEvaluator
from src.utils import setup_logger, Config

# Initialize logger
logger = setup_logger('graphrag_app')
logger.info("Heavy dependencies pre-loaded successfully")

# Global variables for loaded components
local_search = None
global_search = None
processor = None
context_builder = None
prompt_builder = None
llm = None
formatter = None
config = None


def initialize_components():
    """Initialize all GraphRAG components (lazy loading for speed)."""
    global local_search, global_search, processor, context_builder, prompt_builder, llm, formatter, config

    try:
        logger.info("Initializing GraphRAG components...")

        # Load config
        config = Config()
        cfg = config.load()

        # Initialize search components (lightweight, no data loading yet)
        processor = QueryProcessor()
        local_search = LocalSearch()
        global_search = GlobalSearch()
        context_builder = ContextBuilder()
        prompt_builder = PromptBuilder()
        formatter = AnswerFormatter()

        # Initialize LLM
        llm = LLMClient(model=cfg['llm']['model'])

        # IMPORTANT: Pre-load BGE model to avoid first-query delay
        logger.info("Pre-loading BGE embedding model...")
        logger.info("Step 1/2: Initializing TextEmbedder...")
        if processor.embedder is None:
            from src.indexing import TextEmbedder
            processor.embedder = TextEmbedder()
            logger.info("✓ TextEmbedder initialized")

        logger.info("Step 2/2: Loading BGE model into memory (may take 10-30s)...")
        logger.info(f"Calling load_model() on embedder: {processor.embedder}")
        try:
            import sys
            sys.stdout.flush()  # Flush stdout to ensure logs appear
            processor.embedder.load_model()  # Force load model now
            logger.info("✓ BGE model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"❌ BGE model loading failed: {str(e)}"

        logger.info("=" * 60)
        logger.info("✅ INITIALIZATION SUCCESSFUL!")
        logger.info("=" * 60)
        logger.info("System ready for queries. Data will load on first search.")
        return "✅ System ready! (CUDA: {}, BGE model loaded)".format(
            "Enabled" if torch.cuda.is_available() else "Disabled"
        )

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"❌ Initialization failed: {str(e)}"


def run_local_search(query, top_k, generate_answer, ground_truth):
    """Run local search with evaluation."""
    import time
    try:
        if local_search is None:
            return "❌ Please initialize the system first!", "", ""

        logger.info(f"Starting local search for query: '{query}'")
        total_start = time.time()

        # Lazy load data on first search (only chunks for speed)
        if not hasattr(local_search, 'chunk_embeddings') or local_search.chunk_embeddings is None:
            logger.info("First search - loading chunk embeddings only (fast mode)...")
            t0 = time.time()
            local_search.load(load_entities=False, load_graph=False)  # Only load chunks
            logger.info(f"✓ Chunk data loaded in {time.time()-t0:.2f}s!")

        # Process query (embed query)
        logger.info("Processing query (generating embedding)...")
        t1 = time.time()
        query_data = processor.process(query)
        logger.info(f"✓ Query embedded in {time.time()-t1:.2f}s")

        # Perform search (chunks only for speed)
        logger.info(f"Searching top {top_k} results...")
        t2 = time.time()
        results = local_search.search(query_data['embedding'], top_k=top_k, include_entities=False)
        logger.info(f"✓ Search completed in {time.time()-t2:.2f}s")

        # Format search results
        results_text = f"### 🔍 Local Search Results for: '{query}'\n\n"
        results_text += f"Found {len(results)} results:\n\n"

        for i, result in enumerate(results, 1):
            results_text += f"**[{i}]** Score: `{result['score']:.4f}` | Type: `{result['type']}`\n"
            content = result.get('content', '')[:300]
            results_text += f"{content}...\n\n"

        # Generate answer if requested
        answer_text = ""
        if generate_answer:
            logger.info("Generating answer with LLM...")
            t3 = time.time()
            context = context_builder.build_local_context(results)
            prompt = prompt_builder.build_local_prompt(query, context)
            answer = llm.generate(prompt)
            logger.info(f"✓ LLM answer generated in {time.time()-t3:.2f}s")
            sources = context_builder.format_sources(results)
            formatted = formatter.format(answer, sources)
            answer_text = formatted['answer']

            if formatted.get('citations'):
                answer_text += f"\n\n**Sources cited:** {formatted['citations']}"
        else:
            answer_text = "Answer generation skipped."

        # Evaluate
        logger.info("Evaluating results...")
        t4 = time.time()
        evaluator = SearchEvaluator()
        metrics = evaluator.evaluate(
            query=query,
            answer=answer_text if generate_answer else "No answer generated",
            search_results=results,
            ground_truth=ground_truth if ground_truth else None
        )
        logger.info(f"✓ Evaluation completed in {time.time()-t4:.2f}s")

        total_time = time.time() - total_start
        logger.info(f"✅ Total search time: {total_time:.2f}s")

        # Format metrics
        metrics_text = "### 📊 Evaluation Metrics\n\n"
        metrics_text += f"- **Relevance Score**: {metrics['relevance_score']:.4f}\n"
        metrics_text += f"- **Coverage Score**: {metrics['coverage_score']:.4f}\n"
        metrics_text += f"- **Answer Quality**: {metrics['answer_quality']:.4f}\n"
        metrics_text += f"- **Faithfulness**: {metrics['faithfulness']:.4f}\n"

        overall = sum(metrics.values()) / len(metrics)
        metrics_text += f"\n**Overall Score**: {overall:.4f}"
        metrics_text += f"\n\n_Total time: {total_time:.1f}s_"

        return results_text, answer_text, metrics_text

    except Exception as e:
        logger.error(f"Local search error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"❌ Error: {str(e)}", "", ""


def run_global_search(query, top_k, generate_answer, ground_truth):
    """Run global search with evaluation."""
    import time
    try:
        if global_search is None:
            return "❌ Please initialize the system first!", "", ""

        logger.info(f"Starting global search for query: '{query}'")
        total_start = time.time()

        # Lazy load data on first search
        if not hasattr(global_search, 'community_embeddings') or global_search.community_embeddings is None:
            logger.info("First search - loading community embeddings (21MB)...")
            t0 = time.time()
            global_search.load()
            logger.info(f"✓ Community data loaded in {time.time()-t0:.2f}s!")

        # Process query (embed query)
        logger.info("Processing query (generating embedding)...")
        t1 = time.time()
        query_data = processor.process(query)
        logger.info(f"✓ Query embedded in {time.time()-t1:.2f}s")

        # Perform search
        logger.info(f"Searching top {top_k} communities...")
        t2 = time.time()
        results = global_search.search(query_data['embedding'], top_k=top_k)
        logger.info(f"✓ Search completed in {time.time()-t2:.2f}s")

        # Format search results
        results_text = f"### 🌍 Global Search Results for: '{query}'\n\n"
        results_text += f"Found {len(results)} communities:\n\n"

        for i, result in enumerate(results, 1):
            results_text += f"**[Community {result.get('community_id', i)}]** Score: `{result['score']:.4f}`\n"
            results_text += f"- **Title**: {result.get('title', 'N/A')}\n"
            results_text += f"- **Entities**: {result.get('num_entities', 0)}\n"
            summary = result.get('summary', '')[:300]
            results_text += f"- **Summary**: {summary}...\n\n"

        # Generate answer if requested
        answer_text = ""
        if generate_answer:
            logger.info("Generating answer with LLM...")
            t3 = time.time()
            context = context_builder.build_global_context(results)
            prompt = prompt_builder.build_global_prompt(query, context)
            answer = llm.generate(prompt)
            logger.info(f"✓ LLM answer generated in {time.time()-t3:.2f}s")
            sources = context_builder.format_sources(results)
            formatted = formatter.format(answer, sources)
            answer_text = formatted['answer']

            if formatted.get('citations'):
                answer_text += f"\n\n**Sources cited:** {formatted['citations']}"
        else:
            answer_text = "Answer generation skipped."

        # Evaluate
        logger.info("Evaluating results...")
        t4 = time.time()
        evaluator = SearchEvaluator()
        metrics = evaluator.evaluate(
            query=query,
            answer=answer_text if generate_answer else "No answer generated",
            search_results=results,
            ground_truth=ground_truth if ground_truth else None
        )
        logger.info(f"✓ Evaluation completed in {time.time()-t4:.2f}s")

        total_time = time.time() - total_start
        logger.info(f"✅ Total search time: {total_time:.2f}s")

        # Format metrics
        metrics_text = "### 📊 Evaluation Metrics\n\n"
        metrics_text += f"- **Relevance Score**: {metrics['relevance_score']:.4f}\n"
        metrics_text += f"- **Coverage Score**: {metrics['coverage_score']:.4f}\n"
        metrics_text += f"- **Answer Quality**: {metrics['answer_quality']:.4f}\n"
        metrics_text += f"- **Faithfulness**: {metrics['faithfulness']:.4f}\n"

        overall = sum(metrics.values()) / len(metrics)
        metrics_text += f"\n**Overall Score**: {overall:.4f}"
        metrics_text += f"\n\n_Total time: {total_time:.1f}s_"

        return results_text, answer_text, metrics_text

    except Exception as e:
        logger.error(f"Global search error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"❌ Error: {str(e)}", "", ""


# Create Gradio Interface
with gr.Blocks(title="GraphRAG Search System") as app:
    gr.Markdown(
        """
        # 🔍 GraphRAG Search System
        *Knowledge Graph-based Retrieval Augmented Generation*
        """
    )

    # System status section (compact)
    with gr.Row():
        init_status = gr.Textbox(
            label="System Status",
            value="⏳ Initializing...",
            interactive=False,
            show_label=False,
            container=False,
            lines=1,
            scale=4
        )
        init_btn = gr.Button("🔄 Re-initialize", variant="secondary", size="sm", scale=1)

    init_btn.click(fn=initialize_components, outputs=init_status)

    gr.Markdown("---")

    # Main interface with tabs
    with gr.Tabs():
        # Local Search Tab
        with gr.Tab("🔍 Local Search"):
            gr.Markdown("**Best for**: Specific questions, detailed information about entities and relationships")

            # Input section
            with gr.Row():
                local_query = gr.Textbox(
                    label="Query",
                    placeholder="e.g., Who are the key people mentioned in the documents?",
                    lines=2,
                    scale=4
                )
                with gr.Column(scale=1):
                    local_top_k = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=10,
                        step=1,
                        label="Top-k Results"
                    )
                    local_generate = gr.Checkbox(
                        label="Generate Answer",
                        value=True
                    )

            with gr.Row():
                local_ground_truth = gr.Textbox(
                    label="Ground Truth (optional - for evaluation)",
                    placeholder="Expected answer for better evaluation...",
                    lines=2
                )

            local_search_btn = gr.Button("🔍 Search", variant="primary", size="lg")

            gr.Markdown("---")

            # Results section - reordered for better flow
            with gr.Column():
                gr.Markdown("## 📝 Results")
                local_results = gr.Markdown("*Search results will appear here...*")

                gr.Markdown("---")
                gr.Markdown("## 💬 Generated Answer")
                local_answer = gr.Markdown("*Generated answer will appear here...*")

                gr.Markdown("---")
                gr.Markdown("## 📊 Evaluation Metrics")
                local_metrics = gr.Markdown("*Evaluation metrics will appear here...*")

            local_search_btn.click(
                fn=run_local_search,
                inputs=[local_query, local_top_k, local_generate, local_ground_truth],
                outputs=[local_results, local_answer, local_metrics]
            )

        # Global Search Tab
        with gr.Tab("🌍 Global Search"):
            gr.Markdown("**Best for**: High-level questions, overall themes, comprehensive summaries")

            # Input section
            with gr.Row():
                global_query = gr.Textbox(
                    label="Query",
                    placeholder="e.g., What are the main themes in the documents?",
                    lines=2,
                    scale=4
                )
                with gr.Column(scale=1):
                    global_top_k = gr.Slider(
                        minimum=3,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Top-k Communities"
                    )
                    global_generate = gr.Checkbox(
                        label="Generate Answer",
                        value=True
                    )

            with gr.Row():
                global_ground_truth = gr.Textbox(
                    label="Ground Truth (optional - for evaluation)",
                    placeholder="Expected answer for better evaluation...",
                    lines=2
                )

            global_search_btn = gr.Button("🌍 Search", variant="primary", size="lg")

            gr.Markdown("---")

            # Results section - reordered for better flow
            with gr.Column():
                gr.Markdown("## 📝 Results")
                global_results = gr.Markdown("*Search results will appear here...*")

                gr.Markdown("---")
                gr.Markdown("## 💬 Generated Answer")
                global_answer = gr.Markdown("*Generated answer will appear here...*")

                gr.Markdown("---")
                gr.Markdown("## 📊 Evaluation Metrics")
                global_metrics = gr.Markdown("*Evaluation metrics will appear here...*")

            global_search_btn.click(
                fn=run_global_search,
                inputs=[global_query, global_top_k, global_generate, global_ground_truth],
                outputs=[global_results, global_answer, global_metrics]
            )

    # Footer - Compact help section
    with gr.Accordion("📖 Help & Metrics Guide", open=False):
        gr.Markdown(
            """
            ### 🔍 Search Types:
            - **Local Search**: Best for specific questions, detailed information about entities and relationships
            - **Global Search**: Best for high-level questions, overall themes, comprehensive summaries

            ### 📊 Evaluation Metrics:
            - **Relevance Score**: How well search results match the query
            - **Coverage Score**: Diversity and comprehensiveness of information
            - **Answer Quality**: Completeness, coherence, and informativeness
            - **Faithfulness**: How well the answer is grounded in retrieved context

            ### ⚡ Performance Tips:
            - First search loads data (~15-25s for local, ~10-15s for global)
            - Subsequent searches are much faster (~5-15s)
            - Disable "Generate Answer" for faster results (evaluation only)
            """
        )



if __name__ == "__main__":
    logger.info("Starting GraphRAG Web Interface...")

    # IMPORTANT: Initialize BEFORE launching Gradio to avoid threading issues
    logger.info("Initializing components before starting UI...")
    init_result = initialize_components()
    logger.info(f"Initialization result: {init_result}")

    # Update status in UI
    init_status.value = init_result

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        inbrowser=True
    )
