#!/usr/bin/env python3
"""
Traditional RAG Web Interface
A user-friendly GUI for traditional vector-based RAG with evaluation metrics.
"""

import sys
from pathlib import Path

# IMPORTANT: Import heavy dependencies BEFORE Gradio to avoid threading issues
print("Loading dependencies (this may take 20-30 seconds)...", flush=True)
sys.stdout.flush()

import torch
print("  - torch loaded", flush=True)

from sentence_transformers import SentenceTransformer
print("  - sentence_transformers loaded", flush=True)

import gradio as gr
print("  - gradio loaded", flush=True)

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from retrieval import RAGRetriever
from generation import LLMClient, PromptBuilder
from evaluation import SearchEvaluator
from utils import setup_logger, Config

# Initialize logger
logger = setup_logger('rag_app')
logger.info("Heavy dependencies pre-loaded successfully")

# Global variables
retriever = None
llm = None
prompt_builder = None
config = None


def initialize_components():
    """Initialize all RAG components."""
    global retriever, llm, prompt_builder, config

    try:
        logger.info("Initializing Traditional RAG components...")

        # Load config
        config = Config()
        cfg = config.load()

        # Initialize retriever
        logger.info("Loading retriever...")
        retriever = RAGRetriever(cfg)

        # Paths
        processed_dir = Path(__file__).parent / cfg['data']['processed_dir']
        index_path = processed_dir / 'embeddings' / 'faiss_index.bin'
        chunks_path = processed_dir / 'chunks' / 'chunks.json'

        # Load retriever
        retriever.load(
            str(index_path),
            str(chunks_path),
            cfg['embedding']['model']
        )

        # Initialize LLM
        logger.info("Initializing LLM...")
        llm = LLMClient(
            model=cfg['llm']['model'],
            base_url=cfg['llm']['base_url']
        )

        # Initialize prompt builder
        prompt_builder = PromptBuilder()

        logger.info("=" * 60)
        logger.info("✅ INITIALIZATION SUCCESSFUL!")
        logger.info("=" * 60)
        return f"✅ System ready! (CUDA: {'Enabled' if torch.cuda.is_available() else 'Disabled'})"

    except Exception as e:
        logger.error(f"Initialization error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"❌ Initialization failed: {str(e)}"


def run_search(query, top_k, generate_answer, ground_truth):
    """Run RAG search with evaluation."""
    import time
    try:
        if retriever is None:
            return "❌ Please initialize the system first!", "", ""

        logger.info(f"Starting RAG search for query: '{query}'")
        total_start = time.time()

        # Retrieve
        logger.info(f"Retrieving top {top_k} results...")
        t1 = time.time()
        results = retriever.retrieve(query, top_k=top_k)
        logger.info(f"✓ Retrieval completed in {time.time()-t1:.2f}s")

        # Format search results
        results_text = f"### 🔍 Search Results for: '{query}'\n\n"
        results_text += f"Found {len(results)} results:\n\n"

        for i, result in enumerate(results, 1):
            results_text += f"**[{i}]** Score: `{result['score']:.4f}`\n"
            content = result.get('text', '')[:300]
            results_text += f"{content}...\n\n"

        # Generate answer if requested
        answer_text = ""
        if generate_answer:
            logger.info("Generating answer with LLM...")
            t2 = time.time()
            context = retriever.get_context(results, max_tokens=2000)
            prompt = prompt_builder.build_health_prompt(query, context)
            answer = llm.generate(
                prompt,
                temperature=config['generation']['temperature'],
                max_tokens=config['generation']['max_tokens']
            )
            logger.info(f"✓ LLM answer generated in {time.time()-t2:.2f}s")
            answer_text = answer
        else:
            answer_text = "Answer generation skipped."

        # Evaluate
        logger.info("Evaluating results...")
        t3 = time.time()
        evaluator = SearchEvaluator()
        metrics = evaluator.evaluate(
            query=query,
            answer=answer_text if generate_answer else "No answer generated",
            search_results=results,
            ground_truth=ground_truth if ground_truth else None
        )
        logger.info(f"✓ Evaluation completed in {time.time()-t3:.2f}s")

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
        logger.error(f"Search error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return f"❌ Error: {str(e)}", "", ""


# Create Gradio Interface
with gr.Blocks(title="Traditional RAG System") as app:
    gr.Markdown(
        """
        # 🔍 Traditional RAG System
        *Vector-based Retrieval Augmented Generation*
        """
    )

    # System status section
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

    # Main interface
    gr.Markdown("**Traditional vector-based search**: Retrieves relevant chunks using semantic similarity")

    # Input section
    with gr.Row():
        query_input = gr.Textbox(
            label="Query",
            placeholder="e.g., What are the recommended daily iron intake levels?",
            lines=2,
            scale=4
        )
        with gr.Column(scale=1):
            top_k_slider = gr.Slider(
                minimum=5,
                maximum=50,
                value=15,
                step=1,
                label="Top-k Results"
            )
            generate_checkbox = gr.Checkbox(
                label="Generate Answer",
                value=True
            )

    with gr.Row():
        ground_truth_input = gr.Textbox(
            label="Ground Truth (optional - for evaluation)",
            placeholder="Expected answer for better evaluation...",
            lines=2
        )

    search_btn = gr.Button("🔍 Search", variant="primary", size="lg")

    gr.Markdown("---")

    # Results section
    with gr.Column():
        gr.Markdown("## 📝 Results")
        results_output = gr.Markdown("*Search results will appear here...*")

        gr.Markdown("---")
        gr.Markdown("## 💬 Generated Answer")
        answer_output = gr.Markdown("*Generated answer will appear here...*")

        gr.Markdown("---")
        gr.Markdown("## 📊 Evaluation Metrics")
        metrics_output = gr.Markdown("*Evaluation metrics will appear here...*")

    search_btn.click(
        fn=run_search,
        inputs=[query_input, top_k_slider, generate_checkbox, ground_truth_input],
        outputs=[results_output, answer_output, metrics_output]
    )

    # Footer
    with gr.Accordion("📖 Help & Metrics Guide", open=False):
        gr.Markdown(
            """
            ### 🔍 Search Method:
            - **Vector Search**: Uses FAISS to find semantically similar text chunks
            - **Cosine Similarity**: Measures similarity between query and document embeddings
            - **No Graph**: Unlike GraphRAG, this doesn't use knowledge graph structure

            ### 📊 Evaluation Metrics:
            - **Relevance Score**: How well search results match the query
            - **Coverage Score**: Diversity and comprehensiveness of information
            - **Answer Quality**: Completeness, coherence, and informativeness
            - **Faithfulness**: How well the answer is grounded in retrieved context

            ### ⚡ Performance:
            - Typically faster than GraphRAG due to simpler architecture
            - Best for finding similar text passages
            - May miss connections that require understanding relationships
            """
        )


if __name__ == "__main__":
    logger.info("Starting Traditional RAG Web Interface...")

    # Initialize before launching
    logger.info("Initializing components before starting UI...")
    init_result = initialize_components()
    logger.info(f"Initialization result: {init_result}")

    app.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from GraphRAG (7860)
        share=True,
        show_error=True,
        inbrowser=True
    )
