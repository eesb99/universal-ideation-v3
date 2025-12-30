"""
Perplexity Search Integration for Universal Ideation v3

Provides real-time web search capabilities to enhance idea generation
with current trends, market data, and cross-domain insights.

Usage:
    from search.perplexity_search import PerplexitySearch

    searcher = PerplexitySearch()
    insights = searcher.search_domain_trends("protein beverages")
    cross_domain = searcher.search_cross_domain("protein beverages", "gaming")
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path.home() / ".env")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class SearchType(Enum):
    """Types of searches for ideation."""
    DOMAIN_TRENDS = "domain_trends"
    MARKET_GAPS = "market_gaps"
    CROSS_DOMAIN = "cross_domain"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    EMERGING_TECH = "emerging_tech"
    CONSUMER_INSIGHTS = "consumer_insights"


@dataclass
class SearchResult:
    """Result from Perplexity search."""
    query: str
    search_type: SearchType
    content: str
    citations: List[str]
    raw_response: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class PerplexitySearch:
    """
    Perplexity API integration for real-time knowledge retrieval.

    Enhances ideation with:
    - Current market trends and data
    - Competitor landscape analysis
    - Cross-domain innovation examples
    - Emerging technology insights
    - Consumer behavior patterns
    """

    API_URL = "https://api.perplexity.ai/chat/completions"
    DEFAULT_MODEL = "llama-3.1-sonar-large-128k-online"  # Online model with web search

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize Perplexity search client.

        Args:
            api_key: Perplexity API key (or set PERPLEXITY_API_KEY env var)
            model: Model to use (default: llama-3.1-sonar-large-128k-online)
        """
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY")
        self.model = model or self.DEFAULT_MODEL

        if not self.api_key:
            print("Warning: PERPLEXITY_API_KEY not set. Add it to ~/.env")

        if not REQUESTS_AVAILABLE:
            print("Warning: 'requests' library not installed. Run: pip install requests")

    def _make_request(self, messages: List[Dict], temperature: float = 0.2) -> Dict:
        """Make API request to Perplexity."""
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not configured")

        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required: pip install requests")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "return_citations": True
        }

        response = requests.post(
            self.API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        return response.json()

    def search(
        self,
        query: str,
        search_type: SearchType = SearchType.DOMAIN_TRENDS,
        context: Optional[str] = None
    ) -> SearchResult:
        """
        Perform a search query.

        Args:
            query: The search query
            search_type: Type of search to perform
            context: Additional context for the search

        Returns:
            SearchResult with content and citations
        """
        system_prompts = {
            SearchType.DOMAIN_TRENDS: """You are a market research analyst.
Provide current trends, market size, growth rates, and key players.
Focus on recent developments (last 1-2 years). Be specific with data.""",

            SearchType.MARKET_GAPS: """You are a business strategist.
Identify underserved markets, unmet customer needs, and white space opportunities.
Focus on actionable gaps that could be addressed with new products/services.""",

            SearchType.CROSS_DOMAIN: """You are an innovation consultant.
Find examples of successful cross-domain innovations and analogies.
Highlight how concepts from one industry have been applied to another.""",

            SearchType.COMPETITOR_ANALYSIS: """You are a competitive intelligence analyst.
Provide overview of key competitors, their strategies, strengths, and weaknesses.
Include recent moves, funding, and market positioning.""",

            SearchType.EMERGING_TECH: """You are a technology analyst.
Identify emerging technologies relevant to the query.
Focus on maturity level, adoption timeline, and potential applications.""",

            SearchType.CONSUMER_INSIGHTS: """You are a consumer research expert.
Provide insights on consumer behavior, preferences, and pain points.
Include demographic trends and shifting attitudes."""
        }

        system_message = system_prompts.get(
            search_type,
            system_prompts[SearchType.DOMAIN_TRENDS]
        )

        user_message = query
        if context:
            user_message = f"{query}\n\nAdditional context: {context}"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

        try:
            response = self._make_request(messages)

            content = response["choices"][0]["message"]["content"]
            citations = response.get("citations", [])

            return SearchResult(
                query=query,
                search_type=search_type,
                content=content,
                citations=citations,
                raw_response=response,
                success=True
            )

        except Exception as e:
            return SearchResult(
                query=query,
                search_type=search_type,
                content="",
                citations=[],
                raw_response={},
                success=False,
                error=str(e)
            )

    def search_domain_trends(self, domain: str) -> SearchResult:
        """
        Search for current trends in a domain.

        Args:
            domain: The domain/industry to research

        Returns:
            SearchResult with trend information
        """
        query = f"""What are the latest trends, innovations, and market developments
in the {domain} industry? Include:
1. Recent product launches and innovations
2. Market size and growth projections
3. Key consumer trends
4. Emerging technologies being adopted
5. Regulatory changes if relevant"""

        return self.search(query, SearchType.DOMAIN_TRENDS)

    def search_market_gaps(self, domain: str) -> SearchResult:
        """
        Search for market gaps and opportunities.

        Args:
            domain: The domain to analyze

        Returns:
            SearchResult with gap analysis
        """
        query = f"""What are the biggest unmet needs and market gaps in the {domain} industry?
Include:
1. Underserved customer segments
2. Common customer complaints and pain points
3. Features/products customers want but don't exist
4. Price point gaps in the market
5. Geographic or demographic white spaces"""

        return self.search(query, SearchType.MARKET_GAPS)

    def search_cross_domain(
        self,
        target_domain: str,
        source_domain: str
    ) -> SearchResult:
        """
        Search for cross-domain innovation opportunities.

        Args:
            target_domain: The domain you're innovating in
            source_domain: The domain to draw inspiration from

        Returns:
            SearchResult with cross-domain insights
        """
        query = f"""Find examples of successful innovations that applied concepts
from {source_domain} to {target_domain} or similar industries.
Include:
1. Specific product/service examples
2. What concept was transferred
3. How it was adapted
4. Results/success metrics
5. Why it worked"""

        return self.search(query, SearchType.CROSS_DOMAIN)

    def search_competitors(self, domain: str, focus: Optional[str] = None) -> SearchResult:
        """
        Search for competitor landscape.

        Args:
            domain: The domain to analyze
            focus: Optional specific focus area

        Returns:
            SearchResult with competitor analysis
        """
        query = f"""Who are the key players and competitors in the {domain} market?
Include:
1. Market leaders and their market share
2. Innovative startups and disruptors
3. Recent funding and acquisitions
4. Key differentiators of top players
5. Strategic moves and partnerships"""

        if focus:
            query += f"\n\nFocus particularly on: {focus}"

        return self.search(query, SearchType.COMPETITOR_ANALYSIS)

    def search_emerging_tech(self, domain: str) -> SearchResult:
        """
        Search for emerging technologies relevant to domain.

        Args:
            domain: The domain to analyze

        Returns:
            SearchResult with technology insights
        """
        query = f"""What emerging technologies are most relevant to the {domain} industry?
Include:
1. Technologies currently being adopted
2. Technologies on the horizon (1-3 years)
3. Potential game-changers (3-5 years)
4. Examples of early adopters
5. Implementation challenges and requirements"""

        return self.search(query, SearchType.EMERGING_TECH)

    def search_consumer_insights(self, domain: str, segment: Optional[str] = None) -> SearchResult:
        """
        Search for consumer insights and behavior.

        Args:
            domain: The domain to analyze
            segment: Optional specific consumer segment

        Returns:
            SearchResult with consumer insights
        """
        query = f"""What are the key consumer insights and behaviors in the {domain} market?
Include:
1. Primary purchase drivers
2. Key pain points and frustrations
3. Emerging preferences and values
4. Decision-making process
5. Brand loyalty factors"""

        if segment:
            query += f"\n\nFocus on this segment: {segment}"

        return self.search(query, SearchType.CONSUMER_INSIGHTS)

    def get_ideation_context(self, domain: str) -> Dict[str, SearchResult]:
        """
        Get comprehensive context for ideation in a domain.

        Performs multiple searches to build rich context.

        Args:
            domain: The domain to research

        Returns:
            Dict with results for trends, gaps, and tech
        """
        return {
            "trends": self.search_domain_trends(domain),
            "gaps": self.search_market_gaps(domain),
            "emerging_tech": self.search_emerging_tech(domain),
            "consumer_insights": self.search_consumer_insights(domain)
        }

    def format_for_prompt(self, results: Dict[str, SearchResult]) -> str:
        """
        Format search results for inclusion in LLM prompt.

        Args:
            results: Dict of SearchResults from get_ideation_context

        Returns:
            Formatted string for prompt augmentation
        """
        sections = []

        for key, result in results.items():
            if result.success and result.content:
                title = key.replace("_", " ").title()
                sections.append(f"## {title}\n{result.content}")

        if not sections:
            return ""

        return "\n\n".join([
            "# Current Market Intelligence",
            *sections
        ])


def create_perplexity_search(api_key: Optional[str] = None) -> Optional[PerplexitySearch]:
    """
    Factory function to create PerplexitySearch instance.

    Args:
        api_key: Optional API key (uses env var if not provided)

    Returns:
        PerplexitySearch instance or None if not configured
    """
    key = api_key or os.environ.get("PERPLEXITY_API_KEY")
    if not key:
        return None
    return PerplexitySearch(api_key=key)


# Convenience functions for quick searches
def search_trends(domain: str, api_key: Optional[str] = None) -> Optional[str]:
    """Quick search for domain trends."""
    searcher = create_perplexity_search(api_key)
    if not searcher:
        return None
    result = searcher.search_domain_trends(domain)
    return result.content if result.success else None


def search_gaps(domain: str, api_key: Optional[str] = None) -> Optional[str]:
    """Quick search for market gaps."""
    searcher = create_perplexity_search(api_key)
    if not searcher:
        return None
    result = searcher.search_market_gaps(domain)
    return result.content if result.success else None


if __name__ == "__main__":
    # Demo usage
    import argparse

    parser = argparse.ArgumentParser(description="Perplexity Search for Ideation")
    parser.add_argument("domain", help="Domain to search")
    parser.add_argument("-t", "--type", choices=["trends", "gaps", "tech", "competitors", "consumers"],
                        default="trends", help="Type of search")
    parser.add_argument("-s", "--source", help="Source domain for cross-domain search")

    args = parser.parse_args()

    searcher = PerplexitySearch()

    if not searcher.api_key:
        print("Error: Set PERPLEXITY_API_KEY in ~/.env")
        exit(1)

    print(f"Searching {args.type} for: {args.domain}")
    print("-" * 60)

    if args.type == "trends":
        result = searcher.search_domain_trends(args.domain)
    elif args.type == "gaps":
        result = searcher.search_market_gaps(args.domain)
    elif args.type == "tech":
        result = searcher.search_emerging_tech(args.domain)
    elif args.type == "competitors":
        result = searcher.search_competitors(args.domain)
    elif args.type == "consumers":
        result = searcher.search_consumer_insights(args.domain)

    if result.success:
        print(result.content)
        if result.citations:
            print("\n" + "-" * 60)
            print("Sources:")
            for citation in result.citations[:5]:
                print(f"  - {citation}")
    else:
        print(f"Error: {result.error}")
