"""
Cross-Domain Bridge Score for Universal Ideation v3

Measures analogical distance between source and target domains.
Based on conceptual blending theory (Fauconnier & Turner).

Breakthrough ideas often come from connecting distant domains:
- "Netflix for X" - streaming model applied elsewhere
- "Uber for Y" - on-demand model applied elsewhere
- "The iPhone of Z" - integration model applied elsewhere

Optimal range: 3-6 semantic hops (too close = incremental, too far = nonsense)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import re


class DomainCategory(Enum):
    """High-level domain categories for measuring semantic distance."""
    # Technology & Digital
    TECHNOLOGY = "technology"
    SOFTWARE = "software"
    HARDWARE = "hardware"
    AI_ML = "ai_ml"

    # Consumer & Lifestyle
    FOOD_BEVERAGE = "food_beverage"
    HEALTH_WELLNESS = "health_wellness"
    FASHION = "fashion"
    ENTERTAINMENT = "entertainment"
    SPORTS = "sports"

    # Business & Services
    FINANCE = "finance"
    RETAIL = "retail"
    LOGISTICS = "logistics"
    HOSPITALITY = "hospitality"

    # Science & Industry
    BIOTECH = "biotech"
    MANUFACTURING = "manufacturing"
    ENERGY = "energy"
    AGRICULTURE = "agriculture"

    # Other
    EDUCATION = "education"
    GOVERNMENT = "government"
    ARTS = "arts"
    UNKNOWN = "unknown"


@dataclass
class Analogy:
    """A detected analogy in an idea."""
    source_domain: str
    source_concept: str
    target_domain: str
    bridge_phrase: str  # The phrase that creates the bridge
    semantic_hops: int
    quality_score: float  # 0-1: How well the analogy works


@dataclass
class CrossDomainAnalysis:
    """Complete cross-domain analysis result."""
    analogies: List[Analogy]
    primary_source_domain: Optional[str]
    average_hops: float
    bridge_quality: float  # 0-100
    final_score: float  # 0-100
    cross_domain_factors: List[str]


class DomainKnowledgeGraph:
    """
    Simplified knowledge graph for measuring domain distances.

    In production, this would use actual knowledge graph (Wikidata, ConceptNet).
    Here we use a simplified adjacency model.
    """

    # Domain adjacency (1 = very close, 2 = related, 3+ = distant)
    DOMAIN_DISTANCES = {
        # Technology cluster
        (DomainCategory.TECHNOLOGY, DomainCategory.SOFTWARE): 1,
        (DomainCategory.TECHNOLOGY, DomainCategory.HARDWARE): 1,
        (DomainCategory.TECHNOLOGY, DomainCategory.AI_ML): 1,
        (DomainCategory.SOFTWARE, DomainCategory.AI_ML): 1,

        # Consumer cluster
        (DomainCategory.FOOD_BEVERAGE, DomainCategory.HEALTH_WELLNESS): 1,
        (DomainCategory.HEALTH_WELLNESS, DomainCategory.SPORTS): 1,
        (DomainCategory.FASHION, DomainCategory.ENTERTAINMENT): 2,
        (DomainCategory.ENTERTAINMENT, DomainCategory.SPORTS): 2,

        # Business cluster
        (DomainCategory.FINANCE, DomainCategory.RETAIL): 2,
        (DomainCategory.RETAIL, DomainCategory.LOGISTICS): 1,
        (DomainCategory.HOSPITALITY, DomainCategory.ENTERTAINMENT): 2,

        # Science cluster
        (DomainCategory.BIOTECH, DomainCategory.HEALTH_WELLNESS): 2,
        (DomainCategory.BIOTECH, DomainCategory.AGRICULTURE): 2,
        (DomainCategory.MANUFACTURING, DomainCategory.LOGISTICS): 2,

        # Cross-cluster connections (higher distance)
        (DomainCategory.TECHNOLOGY, DomainCategory.FOOD_BEVERAGE): 3,
        (DomainCategory.SOFTWARE, DomainCategory.HEALTH_WELLNESS): 3,
        (DomainCategory.FINANCE, DomainCategory.SPORTS): 4,
        (DomainCategory.AI_ML, DomainCategory.FOOD_BEVERAGE): 4,
        (DomainCategory.ENTERTAINMENT, DomainCategory.BIOTECH): 5,
        (DomainCategory.FASHION, DomainCategory.MANUFACTURING): 4,
        (DomainCategory.ARTS, DomainCategory.TECHNOLOGY): 4,
        (DomainCategory.EDUCATION, DomainCategory.ENTERTAINMENT): 3,
    }

    # Default distance for unspecified pairs
    DEFAULT_DISTANCE = 5

    @classmethod
    def get_distance(cls, domain1: DomainCategory, domain2: DomainCategory) -> int:
        """Get semantic distance between two domains."""
        if domain1 == domain2:
            return 0

        # Check both orderings
        key1 = (domain1, domain2)
        key2 = (domain2, domain1)

        if key1 in cls.DOMAIN_DISTANCES:
            return cls.DOMAIN_DISTANCES[key1]
        elif key2 in cls.DOMAIN_DISTANCES:
            return cls.DOMAIN_DISTANCES[key2]
        else:
            return cls.DEFAULT_DISTANCE


class AnalogyDetector:
    """
    Detects analogies and cross-domain references in idea text.

    Looks for patterns like:
    - "X for Y" (Uber for dogs)
    - "like X but for Y" (like Netflix but for learning)
    - "the X of Y" (the iPhone of fitness)
    - "applies X to Y" (applies gaming to nutrition)
    """

    # Patterns that indicate analogies
    ANALOGY_PATTERNS = [
        # "X for Y" pattern
        r"(?:the\s+)?(\w+(?:\s+\w+)?)\s+for\s+(\w+(?:\s+\w+)?)",
        # "like X but for Y" pattern
        r"like\s+(\w+(?:\s+\w+)?)\s+(?:but\s+)?for\s+(\w+(?:\s+\w+)?)",
        # "the X of Y" pattern
        r"the\s+(\w+(?:\s+\w+)?)\s+of\s+(\w+(?:\s+\w+)?)",
        # "applies X to Y" pattern
        r"appl(?:y|ies|ied)\s+(\w+(?:\s+\w+)?)\s+to\s+(\w+(?:\s+\w+)?)",
        # "X meets Y" pattern
        r"(\w+(?:\s+\w+)?)\s+meets\s+(\w+(?:\s+\w+)?)",
        # "combining X and Y" pattern
        r"combin(?:e|es|ing)\s+(\w+(?:\s+\w+)?)\s+(?:and|with)\s+(\w+(?:\s+\w+)?)",
    ]

    # Known cross-domain source concepts
    KNOWN_SOURCES = {
        # Tech/Platform analogies
        "netflix": DomainCategory.ENTERTAINMENT,
        "uber": DomainCategory.LOGISTICS,
        "airbnb": DomainCategory.HOSPITALITY,
        "amazon": DomainCategory.RETAIL,
        "spotify": DomainCategory.ENTERTAINMENT,
        "tinder": DomainCategory.SOFTWARE,
        "instagram": DomainCategory.ENTERTAINMENT,
        "apple": DomainCategory.TECHNOLOGY,
        "iphone": DomainCategory.TECHNOLOGY,
        "tesla": DomainCategory.MANUFACTURING,

        # Business model analogies
        "subscription": DomainCategory.SOFTWARE,
        "marketplace": DomainCategory.RETAIL,
        "platform": DomainCategory.TECHNOLOGY,
        "saas": DomainCategory.SOFTWARE,
        "freemium": DomainCategory.SOFTWARE,

        # Other domains
        "gaming": DomainCategory.ENTERTAINMENT,
        "gamification": DomainCategory.ENTERTAINMENT,
        "streaming": DomainCategory.ENTERTAINMENT,
        "social media": DomainCategory.SOFTWARE,
        "ai": DomainCategory.AI_ML,
        "machine learning": DomainCategory.AI_ML,
        "blockchain": DomainCategory.TECHNOLOGY,
        "fintech": DomainCategory.FINANCE,
    }

    def detect_analogies(
        self,
        idea: Dict,
        target_domain: DomainCategory = DomainCategory.FOOD_BEVERAGE
    ) -> List[Analogy]:
        """
        Detect analogies in idea text.

        Args:
            idea: Idea dictionary
            target_domain: The domain the idea is targeting

        Returns:
            List of detected analogies
        """
        text = self._extract_text(idea).lower()
        analogies = []

        # Check for known source concepts
        for source_name, source_domain in self.KNOWN_SOURCES.items():
            if source_name in text:
                hops = DomainKnowledgeGraph.get_distance(source_domain, target_domain)
                analogies.append(Analogy(
                    source_domain=source_domain.value,
                    source_concept=source_name,
                    target_domain=target_domain.value,
                    bridge_phrase=f"{source_name} applied to {target_domain.value}",
                    semantic_hops=hops,
                    quality_score=self._assess_bridge_quality(source_name, text, hops)
                ))

        # Apply regex patterns to find structured analogies
        for pattern in self.ANALOGY_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) >= 2:
                    source, target = match[0], match[1]
                    if source.lower() in self.KNOWN_SOURCES:
                        source_domain = self.KNOWN_SOURCES[source.lower()]
                        hops = DomainKnowledgeGraph.get_distance(source_domain, target_domain)
                        analogies.append(Analogy(
                            source_domain=source_domain.value,
                            source_concept=source,
                            target_domain=target_domain.value,
                            bridge_phrase=f"{source} for {target}",
                            semantic_hops=hops,
                            quality_score=self._assess_bridge_quality(source, text, hops)
                        ))

        # Remove duplicates
        seen = set()
        unique_analogies = []
        for a in analogies:
            key = (a.source_concept, a.target_domain)
            if key not in seen:
                seen.add(key)
                unique_analogies.append(a)

        return unique_analogies

    def _extract_text(self, idea: Dict) -> str:
        """Extract searchable text from idea."""
        parts = []
        for key in ['title', 'description', 'differentiators', 'positioning',
                    'semantic_distance_strategy', 'improvements_from_prior']:
            if key in idea:
                value = idea[key]
                if isinstance(value, list):
                    parts.extend(str(v) for v in value)
                else:
                    parts.append(str(value))
        return ' '.join(parts)

    def _assess_bridge_quality(self, source: str, text: str, hops: int) -> float:
        """
        Assess how well the analogy bridge works.

        Quality factors:
        - Is the source clearly mentioned?
        - Is there explanation of how it applies?
        - Are there specific elements borrowed?
        """
        quality = 0.5  # Base quality

        # Source clearly present
        if source.lower() in text.lower():
            quality += 0.2

        # Explanation keywords present
        explanation_words = ['model', 'approach', 'strategy', 'method', 'way', 'like', 'similar']
        if any(word in text.lower() for word in explanation_words):
            quality += 0.15

        # Specific elements mentioned
        specific_words = ['subscription', 'platform', 'on-demand', 'personalized', 'algorithm']
        if any(word in text.lower() for word in specific_words):
            quality += 0.15

        return min(1.0, quality)


class CrossDomainBridge:
    """
    Calculates cross-domain bridge score.

    Formula:
    Score = f(average_hops, num_analogies, bridge_quality)

    Optimal: 3-6 hops
    - <2 hops: Too close, incremental (score: 40-60)
    - 3-6 hops: Sweet spot (score: 70-100)
    - >6 hops: Too far, potentially nonsense (diminishing returns)
    """

    OPTIMAL_HOPS_MIN = 3
    OPTIMAL_HOPS_MAX = 6

    def __init__(self, target_domain: str = "food_beverage"):
        self.target_domain = self._parse_domain(target_domain)
        self.detector = AnalogyDetector()
        self.analysis_history: List[CrossDomainAnalysis] = []

    def _parse_domain(self, domain_str: str) -> DomainCategory:
        """Parse domain string to DomainCategory."""
        domain_str = domain_str.lower().replace(' ', '_').replace('-', '_')

        # Map common terms
        mappings = {
            'protein': DomainCategory.FOOD_BEVERAGE,
            'beverage': DomainCategory.FOOD_BEVERAGE,
            'nutrition': DomainCategory.FOOD_BEVERAGE,
            'food': DomainCategory.FOOD_BEVERAGE,
            'health': DomainCategory.HEALTH_WELLNESS,
            'fitness': DomainCategory.SPORTS,
            'tech': DomainCategory.TECHNOLOGY,
            'ai': DomainCategory.AI_ML,
            'software': DomainCategory.SOFTWARE,
        }

        for key, value in mappings.items():
            if key in domain_str:
                return value

        return DomainCategory.UNKNOWN

    def analyze(self, idea: Dict) -> CrossDomainAnalysis:
        """
        Perform complete cross-domain analysis.

        Args:
            idea: The idea to analyze

        Returns:
            CrossDomainAnalysis with scores and details
        """
        # Detect analogies
        analogies = self.detector.detect_analogies(idea, self.target_domain)

        if not analogies:
            # No cross-domain thinking detected
            analysis = CrossDomainAnalysis(
                analogies=[],
                primary_source_domain=None,
                average_hops=0,
                bridge_quality=0,
                final_score=30.0,  # Low score for no cross-domain
                cross_domain_factors=["No cross-domain analogies detected"]
            )
            self.analysis_history.append(analysis)
            return analysis

        # Calculate metrics
        average_hops = sum(a.semantic_hops for a in analogies) / len(analogies)
        average_quality = sum(a.quality_score for a in analogies) / len(analogies)

        # Find primary source domain
        primary = max(analogies, key=lambda a: a.quality_score)
        primary_source = primary.source_domain

        # Calculate hop-based score
        hop_score = self._calculate_hop_score(average_hops)

        # Calculate quality-based score
        quality_score = average_quality * 100

        # Number of analogies bonus (more = more cross-pollination)
        analogy_bonus = min(20, len(analogies) * 5)

        # Final score
        final_score = (hop_score * 0.5) + (quality_score * 0.3) + analogy_bonus

        # Extract cross-domain factors
        factors = []
        for a in analogies:
            factors.append(
                f"Bridges {a.source_domain} to {a.target_domain} "
                f"({a.semantic_hops} hops): {a.bridge_phrase}"
            )

        analysis = CrossDomainAnalysis(
            analogies=analogies,
            primary_source_domain=primary_source,
            average_hops=average_hops,
            bridge_quality=average_quality * 100,
            final_score=min(100, final_score),
            cross_domain_factors=factors
        )

        self.analysis_history.append(analysis)
        return analysis

    def _calculate_hop_score(self, avg_hops: float) -> float:
        """
        Calculate score based on semantic distance.

        Optimal range: 3-6 hops
        """
        if avg_hops < 2:
            # Too close - incremental, not cross-domain
            return 40 + (avg_hops * 10)  # 40-60
        elif avg_hops <= self.OPTIMAL_HOPS_MIN:
            # Approaching sweet spot
            return 60 + ((avg_hops - 2) * 20)  # 60-80
        elif avg_hops <= self.OPTIMAL_HOPS_MAX:
            # Sweet spot
            return 80 + ((avg_hops - 3) * 6.67)  # 80-100
        else:
            # Too far - diminishing returns
            return max(50, 100 - (avg_hops - 6) * 10)  # 100 down to 50

    def get_evaluation_prompt(self, idea: Dict, domain: str) -> str:
        """
        Generate prompt for agent-based cross-domain evaluation.
        """
        return f"""Evaluate the CROSS-DOMAIN thinking in this idea.

DOMAIN: {domain}

IDEA:
{idea}

CROSS-DOMAIN EVALUATION:
Score how well this idea bridges concepts from distant domains (0-100).

LOOK FOR:
1. Explicit analogies ("X for Y", "like X but for Y")
2. Business model borrowing (subscription, platform, marketplace)
3. Technology transfer (AI applied to traditional domain)
4. Conceptual blending (combining unrelated concepts)

OPTIMAL RANGE:
- 3-6 semantic hops is ideal (creative but coherent)
- <2 hops = too incremental (low score)
- >7 hops = potentially forced or nonsense (diminishing returns)

OUTPUT FORMAT (JSON):
{{
    "analogies_detected": [
        {{"source": "...", "target": "...", "hops": N}}
    ],
    "average_hops": N,
    "bridge_quality": <0-100>,
    "final_score": <0-100>,
    "cross_domain_factors": ["factor1", "factor2"],
    "reasoning": "2-3 sentences explaining the cross-domain connections"
}}
"""

    def get_statistics(self) -> Dict:
        """Get cross-domain analysis statistics."""
        if not self.analysis_history:
            return {"total_analyses": 0}

        scores = [a.final_score for a in self.analysis_history]
        hops = [a.average_hops for a in self.analysis_history if a.average_hops > 0]
        analogy_counts = [len(a.analogies) for a in self.analysis_history]

        return {
            "total_analyses": len(self.analysis_history),
            "average_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "average_hops": sum(hops) / len(hops) if hops else 0,
            "average_analogies": sum(analogy_counts) / len(analogy_counts),
            "ideas_with_analogies": sum(1 for a in self.analysis_history if a.analogies),
            "cross_domain_rate": sum(1 for a in self.analysis_history if a.analogies) / len(self.analysis_history)
        }


def calculate_cross_domain(
    idea: Dict,
    domain: str = "protein beverages"
) -> float:
    """
    Quick helper to calculate cross-domain score.

    Args:
        idea: Idea dictionary
        domain: Target domain context

    Returns:
        Cross-domain score 0-100
    """
    analyzer = CrossDomainBridge(domain)
    analysis = analyzer.analyze(idea)
    return analysis.final_score


# Example analogies for testing
EXAMPLE_ANALOGIES = {
    "netflix_for_protein": {
        "title": "ProteinFlix - Subscription Protein Delivery",
        "description": "The Netflix of protein supplements. Personalized monthly boxes based on your fitness goals and taste preferences. Algorithm learns your preferences over time.",
        "differentiators": ["subscription model", "AI personalization", "discovery feature"],
        "target_market": "Fitness enthusiasts who want variety"
    },
    "uber_for_nutrition": {
        "title": "NutriNow - On-Demand Nutrition Coaching",
        "description": "Like Uber but for nutritionists. Get on-demand video consultations with certified nutrition experts. Pay per session, no commitments.",
        "differentiators": ["on-demand access", "gig economy model", "no subscription"],
        "target_market": "Health-conscious consumers"
    },
    "gamification_protein": {
        "title": "ProteinQuest - Gamified Nutrition",
        "description": "Combining gaming mechanics with protein supplementation. Earn XP, level up, unlock achievements as you hit your protein goals.",
        "differentiators": ["gamification", "social leaderboards", "rewards system"],
        "target_market": "Young fitness gamers"
    },
    "no_analogy": {
        "title": "Standard Whey Protein",
        "description": "High quality whey protein powder for muscle building.",
        "differentiators": ["pure protein", "no additives"],
        "target_market": "Gym goers"
    }
}
