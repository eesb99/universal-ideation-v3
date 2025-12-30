"""
Search module for Universal Ideation v3.

Provides external knowledge retrieval capabilities.
"""

from .perplexity_search import (
    PerplexitySearch,
    SearchResult,
    SearchType,
    create_perplexity_search,
    search_trends,
    search_gaps
)

__all__ = [
    "PerplexitySearch",
    "SearchResult",
    "SearchType",
    "create_perplexity_search",
    "search_trends",
    "search_gaps"
]
