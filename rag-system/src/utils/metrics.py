"""Metrics collection and evaluation."""

import time
from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class Metrics:
    """Collect and compute metrics."""

    query_times: List[float] = field(default_factory=list)
    retrieval_times: List[float] = field(default_factory=list)
    generation_times: List[float] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)

    def add_query_time(self, duration: float):
        """Add query processing time."""
        self.query_times.append(duration)

    def add_retrieval_time(self, duration: float):
        """Add retrieval time."""
        self.retrieval_times.append(duration)

    def add_generation_time(self, duration: float):
        """Add generation time."""
        self.generation_times.append(duration)

    def add_relevance_score(self, score: float):
        """Add relevance score."""
        self.relevance_scores.append(score)

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        return {
            'avg_query_time': avg(self.query_times),
            'avg_retrieval_time': avg(self.retrieval_times),
            'avg_generation_time': avg(self.generation_times),
            'avg_relevance_score': avg(self.relevance_scores),
            'total_queries': len(self.query_times)
        }

    def reset(self):
        """Reset all metrics."""
        self.query_times = []
        self.retrieval_times = []
        self.generation_times = []
        self.relevance_scores = []


class Timer:
    """Context manager for timing operations."""

    def __init__(self):
        self.start_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.duration = time.time() - self.start_time
