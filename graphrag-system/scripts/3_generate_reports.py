#!/usr/bin/env python3
"""Step 3: Generate community summaries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing import GraphBuilder, CommunityDetector, CommunitySummarizer
from src.utils import setup_logger, Config


def main():
    logger = setup_logger('generate_reports')
    config = Config()
    cfg = config.load()

    try:
        # Load graph
        logger.info("Loading knowledge graph...")
        graph_builder = GraphBuilder(cfg['graph']['output_dir'])
        graph_builder.load()
        graph = graph_builder.graph

        # Load communities
        logger.info("Loading communities...")
        detector = CommunityDetector(cfg['community']['output_dir'])
        communities = detector.load()

        logger.info(f"Loaded {len(communities)} communities")

        # Generate summaries
        logger.info("Generating community summaries...")
        summarizer = CommunitySummarizer()
        summarizer.model = cfg['llm']['model']

        # Load prompt template
        with open('configs/prompts/community_summary.txt') as f:
            summary_prompt = f.read()

        reports = summarizer.summarize_all(communities, graph, summary_prompt)

        # Save reports
        summarizer.save(reports)

        logger.info(f"Generated {len(reports)} community reports")
        logger.info("Step 3 complete! Run 4_create_embeddings.py next.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
