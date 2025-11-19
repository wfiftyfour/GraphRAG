"""LLM utility functions."""

import time
from typing import List, Callable, Any
from functools import wraps


class LLMUtils:
    """Utility functions for LLM operations."""

    @staticmethod
    def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
        """Decorator for retrying LLM calls with exponential backoff."""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None

                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            print(f"Attempt {attempt + 1} failed, retrying in {delay}s...")
                            time.sleep(delay)

                raise last_exception

            return wrapper
        return decorator

    @staticmethod
    def batch_process(items: List[Any], batch_size: int, process_func: Callable) -> List[Any]:
        """Process items in batches."""
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = process_func(batch)
            results.extend(batch_results)

            # Small delay between batches
            if i + batch_size < len(items):
                time.sleep(0.1)

        return results

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough estimate of token count."""
        # Approximate: 4 characters per token for English
        return len(text) // 4

    @staticmethod
    def truncate_to_tokens(text: str, max_tokens: int) -> str:
        """Truncate text to approximate token limit."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        # Try to truncate at sentence boundary
        truncated = text[:max_chars]
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:
            return truncated[:last_period + 1]

        return truncated + "..."

    @staticmethod
    def clean_llm_json(response: str) -> str:
        """Extract JSON from LLM response."""
        # Find JSON array or object
        start_array = response.find('[')
        start_obj = response.find('{')

        if start_array == -1 and start_obj == -1:
            return "[]"

        if start_array != -1 and (start_obj == -1 or start_array < start_obj):
            end = response.rfind(']') + 1
            return response[start_array:end] if end > start_array else "[]"
        else:
            end = response.rfind('}') + 1
            return response[start_obj:end] if end > start_obj else "{}"
