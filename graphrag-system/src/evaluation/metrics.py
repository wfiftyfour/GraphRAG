"""Evaluation metrics for search quality assessment."""

from typing import List, Dict, Any
import numpy as np
from collections import Counter
import re


class SearchEvaluator:
    """Evaluates search results and generated answers using multiple metrics."""

    def __init__(self):
        self.metrics = {}

    def evaluate(self,
                 query: str,
                 answer: str,
                 search_results: List[Dict[str, Any]],
                 ground_truth: str = None) -> Dict[str, float]:
        """
        Evaluate search results and generated answer.

        Returns 4 key metrics:
        1. Relevance Score: Measures relevance of retrieved results
        2. Coverage Score: How well results cover different aspects
        3. Answer Quality: Quality metrics of generated answer
        4. Faithfulness: How faithful the answer is to retrieved context

        Args:
            query: The search query
            answer: Generated answer
            search_results: List of search results
            ground_truth: Optional ground truth answer for comparison

        Returns:
            Dictionary with 4 evaluation metrics
        """
        metrics = {}

        # Metric 1: Relevance Score
        metrics['relevance_score'] = self._calculate_relevance(query, search_results)

        # Metric 2: Coverage Score
        metrics['coverage_score'] = self._calculate_coverage(search_results)

        # Metric 3: Answer Quality Score
        metrics['answer_quality'] = self._calculate_answer_quality(
            answer, query, ground_truth
        )

        # Metric 4: Faithfulness Score
        metrics['faithfulness'] = self._calculate_faithfulness(answer, search_results)

        self.metrics = metrics
        return metrics

    def _calculate_relevance(self, query: str, results: List[Dict[str, Any]]) -> float:
        """
        Metric 1: Relevance Score
        Measures average relevance of top-k retrieved results.
        Based on similarity scores and query-result overlap.
        """
        if not results:
            return 0.0

        # Use existing similarity scores from search
        similarity_scores = [r.get('score', 0.0) for r in results]

        # Calculate query-result token overlap
        query_tokens = set(self._tokenize(query.lower()))
        overlap_scores = []

        for result in results:
            content = result.get('content', '') or result.get('summary', '')
            result_tokens = set(self._tokenize(content.lower()))

            if query_tokens:
                overlap = len(query_tokens & result_tokens) / len(query_tokens)
                overlap_scores.append(overlap)
            else:
                overlap_scores.append(0.0)

        # Combine similarity and overlap with weights
        relevance_scores = [
            0.7 * sim + 0.3 * overlap
            for sim, overlap in zip(similarity_scores, overlap_scores)
        ]

        # Weight top results more (DCG-like)
        weighted_sum = sum(
            score / np.log2(i + 2) for i, score in enumerate(relevance_scores)
        )
        ideal_sum = sum(1.0 / np.log2(i + 2) for i in range(len(relevance_scores)))

        return weighted_sum / ideal_sum if ideal_sum > 0 else 0.0

    def _calculate_coverage(self, results: List[Dict[str, Any]]) -> float:
        """
        Metric 2: Coverage Score
        Measures diversity and comprehensiveness of retrieved information.
        High coverage means results cover different aspects/entities.
        """
        if not results:
            return 0.0

        # Extract all content
        all_content = []
        entities = set()
        types = set()

        for result in results:
            content = result.get('content', '') or result.get('summary', '')
            all_content.append(content)

            # Track result types
            types.add(result.get('type', 'unknown'))

            # Extract entities from metadata
            metadata = result.get('metadata', {})
            if 'name' in metadata:
                entities.add(metadata['name'])

            # Extract entities from graph context
            graph_ctx = result.get('graph_context', {})
            entities.update(graph_ctx.get('neighbors', []))

        # Calculate diversity metrics

        # 1. Unique entities coverage (0-1)
        entity_diversity = min(len(entities) / (len(results) * 2), 1.0)

        # 2. Type diversity (0-1)
        type_diversity = min(len(types) / 3, 1.0)  # Expect up to 3 types

        # 3. Content diversity using token overlap (OPTIMIZED)
        # Only compare first 5 results to avoid O(n^2) complexity
        if len(all_content) > 1:
            sample_size = min(5, len(all_content))
            overlaps = []
            for i in range(sample_size):
                for j in range(i + 1, sample_size):
                    tokens_i = set(self._tokenize(all_content[i].lower()[:500]))  # Limit token extraction
                    tokens_j = set(self._tokenize(all_content[j].lower()[:500]))

                    if tokens_i and tokens_j:
                        overlap = len(tokens_i & tokens_j) / max(len(tokens_i), len(tokens_j))
                        overlaps.append(overlap)

            # Lower overlap = higher diversity
            content_diversity = 1.0 - (sum(overlaps) / len(overlaps)) if overlaps else 0.5
        else:
            content_diversity = 0.0

        # Combine diversity metrics
        coverage = (
            0.4 * entity_diversity +
            0.3 * content_diversity +
            0.3 * type_diversity
        )

        return coverage

    def _calculate_answer_quality(self,
                                   answer: str,
                                   query: str,
                                   ground_truth: str = None) -> float:
        """
        Metric 3: Answer Quality Score
        Measures quality of generated answer including:
        - Completeness (addresses the query)
        - Coherence (well-structured)
        - Informativeness (contains details)
        - Similarity to ground truth (if available)
        """
        if not answer:
            return 0.0

        scores = []

        # Limit answer length for performance
        answer_sample = answer[:2000] if len(answer) > 2000 else answer

        # 1. Completeness: Does answer address query terms?
        query_tokens = set(self._tokenize(query.lower()))
        answer_tokens = set(self._tokenize(answer_sample.lower()))

        if query_tokens:
            query_coverage = len(query_tokens & answer_tokens) / len(query_tokens)
            scores.append(query_coverage)

        # 2. Informativeness: Length and detail
        # Normalize by expected length (100-500 words is good)
        word_count = len(answer_sample.split())
        if word_count < 50:
            length_score = word_count / 50
        elif word_count > 500:
            length_score = 1.0 - min((word_count - 500) / 500, 0.5)
        else:
            length_score = 1.0
        scores.append(length_score)

        # 3. Coherence: Sentence structure (SIMPLIFIED)
        sentences = [s.strip() for s in re.split(r'[.!?]+', answer_sample) if s.strip()]
        if sentences:
            # Check for varied sentence lengths (good writing) - sample first 10 sentences
            sample_sentences = sentences[:10]
            lengths = [len(s.split()) for s in sample_sentences]
            avg_length = np.mean(lengths)
            std_length = np.std(lengths) if len(lengths) > 1 else 5

            # Good if avg 10-25 words, with some variance
            length_quality = 1.0 - min(abs(avg_length - 15) / 15, 1.0)
            variance_quality = min(std_length / 5, 1.0)

            coherence = 0.6 * length_quality + 0.4 * variance_quality
            scores.append(coherence)

        # 4. Ground truth similarity (if available) - SIMPLIFIED
        if ground_truth:
            gt_sample = ground_truth[:2000] if len(ground_truth) > 2000 else ground_truth
            gt_tokens = set(self._tokenize(gt_sample.lower()))

            if gt_tokens and answer_tokens:
                precision = len(gt_tokens & answer_tokens) / len(answer_tokens)
                recall = len(gt_tokens & answer_tokens) / len(gt_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                scores.append(f1)

        return np.mean(scores) if scores else 0.0

    def _calculate_faithfulness(self,
                               answer: str,
                               results: List[Dict[str, Any]]) -> float:
        """
        Metric 4: Faithfulness Score
        Measures how faithful/grounded the answer is to retrieved context.
        High score means answer uses information from retrieved results.
        """
        if not answer or not results:
            return 0.0

        # Limit answer length for performance
        answer_sample = answer[:2000] if len(answer) > 2000 else answer
        answer_tokens = set(self._tokenize(answer_sample.lower()))

        # Collect all context tokens (limit to top 5 results for performance)
        context_tokens = set()
        for result in results[:5]:
            content = result.get('content', '') or result.get('summary', '')
            # Limit content length
            content_sample = content[:1000] if len(content) > 1000 else content
            context_tokens.update(self._tokenize(content_sample.lower()))

        if not answer_tokens:
            return 0.0

        # Calculate what portion of answer comes from context
        grounded_tokens = answer_tokens & context_tokens
        faithfulness = len(grounded_tokens) / len(answer_tokens)

        # Check for hallucination indicators (SIMPLIFIED)
        # Named entities in answer should appear in context
        answer_sentences = [s.strip() for s in re.split(r'[.!?]+', answer_sample) if s.strip()]

        # Extract capitalized terms (potential entities) - limit to first 10 sentences
        answer_entities = set()
        for sentence in answer_sentences[:10]:
            words = sentence.split()
            for word in words:
                if word and word[0].isupper() and len(word) > 1:
                    answer_entities.add(word.lower())

        # Check if entities appear in context
        context_text = ' '.join([
            (r.get('content', '') or r.get('summary', ''))[:1000] for r in results[:5]
        ]).lower()

        if answer_entities:
            entity_grounding = sum(
                1 for entity in answer_entities if entity in context_text
            ) / len(answer_entities)

            # Combine token and entity faithfulness
            faithfulness = 0.7 * faithfulness + 0.3 * entity_grounding

        return faithfulness

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = [t for t in text.split() if len(t) > 2]  # Filter short tokens
        return tokens

    def get_summary(self) -> str:
        """Get formatted summary of metrics."""
        if not self.metrics:
            return "No metrics calculated yet."

        summary = "\n" + "="*60 + "\n"
        summary += "EVALUATION METRICS\n"
        summary += "="*60 + "\n\n"

        metric_descriptions = {
            'relevance_score': 'Relevance Score    (Query-Result Match)',
            'coverage_score': 'Coverage Score     (Information Diversity)',
            'answer_quality': 'Answer Quality     (Completeness & Coherence)',
            'faithfulness': 'Faithfulness       (Grounding to Context)'
        }

        for key, desc in metric_descriptions.items():
            score = self.metrics.get(key, 0.0)
            bar = self._get_progress_bar(score)
            summary += f"{desc}: {score:.4f} {bar}\n"

        # Overall score
        overall = np.mean(list(self.metrics.values()))
        summary += f"\n{'='*60}\n"
        summary += f"Overall Score: {overall:.4f} {self._get_progress_bar(overall)}\n"
        summary += f"{'='*60}\n"

        return summary

    def _get_progress_bar(self, score: float, length: int = 20) -> str:
        """Generate visual progress bar."""
        filled = int(score * length)
        bar = '█' * filled + '░' * (length - filled)
        return f"[{bar}]"

    def get_metrics_dict(self) -> Dict[str, float]:
        """Get metrics as dictionary."""
        return self.metrics.copy()
