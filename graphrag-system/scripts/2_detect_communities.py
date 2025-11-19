#!/usr/bin/env python3
"""Step 2: Detect communities in the knowledge graph."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing import GraphBuilder, CommunityDetector
from src.utils import setup_logger, Config


def main():
    logger = setup_logger('detect_communities')
    config = Config()
    cfg = config.load()

    try:
        # Load graph
        logger.info("Loading knowledge graph...")
        graph_builder = GraphBuilder(cfg['graph']['output_dir'])
        graph_builder.load()
        graph = graph_builder.graph

        logger.info(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Detect communities
        logger.info("Detecting communities using Leiden algorithm...")
        detector = CommunityDetector(cfg['community']['output_dir'])
        communities = detector.detect(
            graph,
            resolution=cfg['community']['resolution']
        )

        logger.info(f"Detected {len(communities)} communities")

        # Print community stats
        sizes = [len(members) for members in communities.values()]
        logger.info(f"Community sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")

        # Save communities
        detector.save()

        logger.info("Step 2 complete! Run 3_generate_reports.py next.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
