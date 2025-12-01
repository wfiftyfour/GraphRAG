#!/usr/bin/env python3
"""Compare Traditional RAG vs GraphRAG performance."""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils import setup_logger


def load_results(file_path: str) -> Dict[str, Any]:
    """Load evaluation results from JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)


def format_metric(value: float, width: int = 8) -> str:
    """Format metric value."""
    return f"{value:.4f}".rjust(width)


def calculate_improvement(baseline: float, comparison: float) -> str:
    """Calculate percentage improvement."""
    if baseline == 0:
        return "N/A"

    improvement = ((comparison - baseline) / baseline) * 100
    sign = "+" if improvement > 0 else ""
    return f"{sign}{improvement:.2f}%"


def main():
    parser = argparse.ArgumentParser(description='Compare RAG vs GraphRAG')
    parser.add_argument('--rag-results', required=True, help='Traditional RAG evaluation results')
    parser.add_argument('--graphrag-local', help='GraphRAG local search results')
    parser.add_argument('--graphrag-global', help='GraphRAG global search results')
    parser.add_argument('--output', help='Output markdown file')
    args = parser.parse_args()

    logger = setup_logger('compare')

    # Load results
    logger.info("Loading evaluation results...")
    rag_results = load_results(args.rag_results)

    graphrag_local = None
    graphrag_global = None

    if args.graphrag_local:
        graphrag_local = load_results(args.graphrag_local)

    if args.graphrag_global:
        graphrag_global = load_results(args.graphrag_global)

    # Build comparison report
    report_lines = []
    report_lines.append("# RAG vs GraphRAG Performance Comparison")
    report_lines.append("")
    report_lines.append("## Summary Statistics")
    report_lines.append("")

    # Table header
    report_lines.append("| Metric | Traditional RAG | GraphRAG Local | GraphRAG Global | Best |")
    report_lines.append("|--------|----------------|----------------|-----------------|------|")

    # Get summaries
    rag_summary = rag_results['summary']
    local_summary = graphrag_local['summary'].get('local', {}) if graphrag_local else {}
    global_summary = graphrag_global['summary'].get('global', {}) if graphrag_global else {}

    # Metrics to compare
    metrics = [
        ('avg_relevance_score', 'Relevance Score'),
        ('avg_coverage_score', 'Coverage Score'),
        ('avg_answer_quality', 'Answer Quality'),
        ('avg_faithfulness', 'Faithfulness'),
        ('overall_score', 'Overall Score'),
        ('avg_time', 'Avg Time (s)')
    ]

    for key, label in metrics:
        rag_val = rag_summary.get(key, 0)
        local_val = local_summary.get(key, 0)
        global_val = global_summary.get(key, 0)

        # Find best (for time, lower is better)
        if key == 'avg_time':
            values = [(rag_val, 'RAG'), (local_val, 'Local'), (global_val, 'Global')]
            values = [(v, n) for v, n in values if v > 0]
            best = min(values, key=lambda x: x[0])[1] if values else 'N/A'
        else:
            values = [(rag_val, 'RAG'), (local_val, 'Local'), (global_val, 'Global')]
            best = max(values, key=lambda x: x[0])[1]

        report_lines.append(
            f"| {label} | "
            f"{format_metric(rag_val)} | "
            f"{format_metric(local_val) if local_val else 'N/A'} | "
            f"{format_metric(global_val) if global_val else 'N/A'} | "
            f"**{best}** |"
        )

    report_lines.append("")
    report_lines.append("## Detailed Comparison")
    report_lines.append("")

    # RAG vs GraphRAG Local
    if graphrag_local:
        report_lines.append("### Traditional RAG vs GraphRAG Local Search")
        report_lines.append("")
        report_lines.append("| Metric | RAG | GraphRAG Local | Improvement |")
        report_lines.append("|--------|-----|----------------|-------------|")

        for key, label in metrics[:5]:  # Exclude time
            rag_val = rag_summary.get(key, 0)
            local_val = local_summary.get(key, 0)
            improvement = calculate_improvement(rag_val, local_val)

            report_lines.append(
                f"| {label} | {format_metric(rag_val)} | "
                f"{format_metric(local_val)} | {improvement} |"
            )

        report_lines.append("")

    # RAG vs GraphRAG Global
    if graphrag_global:
        report_lines.append("### Traditional RAG vs GraphRAG Global Search")
        report_lines.append("")
        report_lines.append("| Metric | RAG | GraphRAG Global | Improvement |")
        report_lines.append("|--------|-----|-----------------|-------------|")

        for key, label in metrics[:5]:  # Exclude time
            rag_val = rag_summary.get(key, 0)
            global_val = global_summary.get(key, 0)
            improvement = calculate_improvement(rag_val, global_val)

            report_lines.append(
                f"| {label} | {format_metric(rag_val)} | "
                f"{format_metric(global_val)} | {improvement} |"
            )

        report_lines.append("")

    # Analysis
    report_lines.append("## Analysis")
    report_lines.append("")

    # Determine winner for each metric
    winners = {}
    for key, label in metrics[:5]:
        rag_val = rag_summary.get(key, 0)
        local_val = local_summary.get(key, 0) if local_summary else 0
        global_val = global_summary.get(key, 0) if global_summary else 0

        max_val = max(rag_val, local_val, global_val)
        if rag_val == max_val:
            winners[label] = 'Traditional RAG'
        elif local_val == max_val:
            winners[label] = 'GraphRAG Local'
        else:
            winners[label] = 'GraphRAG Global'

    report_lines.append("### Best Performing System per Metric:")
    report_lines.append("")
    for label, winner in winners.items():
        report_lines.append(f"- **{label}**: {winner}")

    report_lines.append("")
    report_lines.append("### Key Observations:")
    report_lines.append("")
    report_lines.append("1. **Retrieval Quality**: Compare how different approaches retrieve relevant information")
    report_lines.append("2. **Coverage**: Analyze which system provides more comprehensive answers")
    report_lines.append("3. **Speed**: Traditional RAG is typically faster due to simpler architecture")
    report_lines.append("4. **Context Understanding**: GraphRAG may excel in understanding relationships and connections")

    report_lines.append("")
    report_lines.append("## Configuration")
    report_lines.append("")
    report_lines.append(f"- **Total Queries**: {rag_summary['total_queries']}")
    report_lines.append(f"- **Answers Generated**: {rag_summary['answers_generated']}")
    report_lines.append(f"- **Evaluation Metrics**: Relevance, Coverage, Answer Quality, Faithfulness")

    # Output report
    report_text = '\n'.join(report_lines)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"Comparison report saved to: {args.output}")
    else:
        print("\n" + report_text)

    logger.info("\nComparison complete!")


if __name__ == '__main__':
    main()
