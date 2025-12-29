"""
Salience Weighting for Universal Ideation v3.2

Weights atomic claims by their importance for novelty assessment:
- Core claims (central to the idea) weighted higher
- Supporting claims (elaboration) weighted lower
- Considers claim type, position, specificity, and uniqueness
- Prevents inflated novelty from trivial claim novelty

Based on NovAScore: Salience-weighted novelty evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import re
import math

from .acu_decomposer import AtomicClaim, ClaimType, ClaimSource


class SalienceLevel(Enum):
    """Salience classification levels."""
    CRITICAL = "critical"       # Core differentiator, essential to idea
    IMPORTANT = "important"     # Key feature or benefit
    MODERATE = "moderate"       # Supporting detail
    LOW = "low"                 # Minor or generic claim


@dataclass
class SalienceScore:
    """Salience assessment for a single claim."""
    claim_id: str
    salience_level: SalienceLevel
    salience_weight: float  # 0-1, used for weighted novelty
    factors: Dict[str, float]  # Individual factor scores
    explanation: str


@dataclass
class SalienceResult:
    """Complete salience weighting result for an idea."""
    idea_id: str
    scores: List[SalienceScore]
    total_claims: int
    critical_count: int
    important_count: int
    moderate_count: int
    low_count: int
    average_salience: float
    weight_distribution: Dict[str, float]  # Claim type -> avg weight

    def get_weight(self, claim_id: str) -> float:
        """Get salience weight for a claim."""
        for score in self.scores:
            if score.claim_id == claim_id:
                return score.salience_weight
        return 0.5  # Default moderate weight

    def to_dict(self) -> Dict:
        return {
            "idea_id": self.idea_id,
            "total_claims": self.total_claims,
            "critical_count": self.critical_count,
            "important_count": self.important_count,
            "moderate_count": self.moderate_count,
            "low_count": self.low_count,
            "average_salience": self.average_salience,
            "weight_distribution": self.weight_distribution
        }


class SalienceWeighter:
    """
    Assigns salience weights to atomic claims.

    Factors considered:
    1. Claim type (novelty/comparison > feature > benefit > market)
    2. Source position (title > differentiator > description)
    3. Specificity (specific numbers/names > generic)
    4. Uniqueness within idea (unique keywords > repeated)
    5. Actionability (actionable claims > abstract)
    """

    # Base weights by claim type (importance for novelty)
    TYPE_WEIGHTS = {
        ClaimType.NOVELTY: 1.0,      # Explicit novelty claims most important
        ClaimType.COMPARISON: 0.95,   # Comparisons show differentiation
        ClaimType.FEATURE: 0.85,      # Core features
        ClaimType.MECHANISM: 0.80,    # How it works
        ClaimType.BENEFIT: 0.75,      # Value propositions
        ClaimType.MARKET: 0.65,       # Target market
        ClaimType.CONSTRAINT: 0.55,   # Limitations
        ClaimType.UNKNOWN: 0.50,      # Unknown type
    }

    # Source position weights (where claim came from)
    SOURCE_WEIGHTS = {
        ClaimSource.TITLE: 1.0,           # Title claims are core
        ClaimSource.DIFFERENTIATOR: 0.95, # Explicit differentiators
        ClaimSource.MECHANISM: 0.80,      # How it works
        ClaimSource.DESCRIPTION: 0.70,    # Description elaborates
        ClaimSource.TARGET_MARKET: 0.60,  # Market is supporting
        ClaimSource.CUSTOM: 0.50,         # LLM-extracted
    }

    # Specificity indicators (boost salience)
    SPECIFICITY_PATTERNS = [
        (r'\d+%', 0.15),           # Percentages
        (r'\$[\d,]+', 0.15),       # Dollar amounts
        (r'\d+\s*(x|times)', 0.12), # Multipliers
        (r'first|only|unique', 0.10),  # Uniqueness claims
        (r'\d+\s*(?:mg|g|ml|l|kg)', 0.08),  # Measurements
        (r'patent|trademark', 0.10),  # IP indicators
    ]

    # Generic/low-salience indicators (reduce salience)
    GENERIC_PATTERNS = [
        (r'^(?:the|a|an)\s+', -0.05),     # Article starters
        (r'good|great|best|better', -0.10),  # Vague superlatives
        (r'easy|simple|fast', -0.05),      # Generic benefits
        (r'quality|premium|professional', -0.08),  # Overused terms
    ]

    def __init__(
        self,
        type_weights: Optional[Dict[ClaimType, float]] = None,
        source_weights: Optional[Dict[ClaimSource, float]] = None,
        salience_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize salience weighter.

        Args:
            type_weights: Custom claim type weights
            source_weights: Custom source position weights
            salience_thresholds: Custom thresholds for salience levels
        """
        self.type_weights = type_weights or dict(self.TYPE_WEIGHTS)
        self.source_weights = source_weights or dict(self.SOURCE_WEIGHTS)

        self.salience_thresholds = salience_thresholds or {
            "critical": 0.85,
            "important": 0.70,
            "moderate": 0.50,
        }

    def weight_claims(
        self,
        claims: List[AtomicClaim],
        idea_id: str = "unknown"
    ) -> SalienceResult:
        """
        Assign salience weights to all claims.

        Args:
            claims: Claims to weight
            idea_id: ID of the source idea

        Returns:
            SalienceResult with all weights
        """
        # Build keyword frequency map for uniqueness scoring
        keyword_freq = self._build_keyword_frequency(claims)

        scores = []
        for claim in claims:
            score = self._score_claim(claim, keyword_freq, len(claims))
            scores.append(score)

        # Calculate statistics
        critical = sum(1 for s in scores if s.salience_level == SalienceLevel.CRITICAL)
        important = sum(1 for s in scores if s.salience_level == SalienceLevel.IMPORTANT)
        moderate = sum(1 for s in scores if s.salience_level == SalienceLevel.MODERATE)
        low = sum(1 for s in scores if s.salience_level == SalienceLevel.LOW)

        avg_salience = (
            sum(s.salience_weight for s in scores) / len(scores)
            if scores else 0
        )

        # Calculate weight distribution by claim type
        type_weights: Dict[str, List[float]] = {}
        for i, score in enumerate(scores):
            claim_type = claims[i].claim_type.value
            if claim_type not in type_weights:
                type_weights[claim_type] = []
            type_weights[claim_type].append(score.salience_weight)

        weight_distribution = {
            t: sum(ws) / len(ws) if ws else 0
            for t, ws in type_weights.items()
        }

        return SalienceResult(
            idea_id=idea_id,
            scores=scores,
            total_claims=len(claims),
            critical_count=critical,
            important_count=important,
            moderate_count=moderate,
            low_count=low,
            average_salience=avg_salience,
            weight_distribution=weight_distribution
        )

    def _score_claim(
        self,
        claim: AtomicClaim,
        keyword_freq: Dict[str, int],
        total_claims: int
    ) -> SalienceScore:
        """Score a single claim's salience."""
        factors = {}

        # Factor 1: Claim type
        type_score = self.type_weights.get(claim.claim_type, 0.5)
        factors["claim_type"] = type_score

        # Factor 2: Source position
        source_score = self.source_weights.get(claim.source, 0.5)
        factors["source_position"] = source_score

        # Factor 3: Specificity
        specificity_score = self._calculate_specificity(claim.text)
        factors["specificity"] = specificity_score

        # Factor 4: Uniqueness (keywords not repeated much)
        uniqueness_score = self._calculate_uniqueness(claim.keywords, keyword_freq)
        factors["uniqueness"] = uniqueness_score

        # Factor 5: Actionability
        actionability_score = self._calculate_actionability(claim.text)
        factors["actionability"] = actionability_score

        # Factor 6: Confidence (from extraction)
        factors["extraction_confidence"] = claim.confidence

        # Calculate weighted combination
        # Type and source are most important, others are modifiers
        base_weight = (type_score * 0.35) + (source_score * 0.25)
        modifier = (
            specificity_score * 0.15 +
            uniqueness_score * 0.10 +
            actionability_score * 0.10 +
            claim.confidence * 0.05
        )

        salience_weight = base_weight + modifier

        # Clamp to 0-1
        salience_weight = max(0.1, min(1.0, salience_weight))

        # Classify level
        salience_level = self._classify_salience(salience_weight)

        # Generate explanation
        explanation = self._generate_explanation(factors, salience_level)

        return SalienceScore(
            claim_id=claim.id,
            salience_level=salience_level,
            salience_weight=round(salience_weight, 3),
            factors=factors,
            explanation=explanation
        )

    def _build_keyword_frequency(self, claims: List[AtomicClaim]) -> Dict[str, int]:
        """Build frequency map of keywords across all claims."""
        freq = {}
        for claim in claims:
            for kw in claim.keywords:
                kw_lower = kw.lower()
                freq[kw_lower] = freq.get(kw_lower, 0) + 1
        return freq

    def _calculate_specificity(self, text: str) -> float:
        """Calculate specificity score based on concrete details."""
        score = 0.5  # Base score

        text_lower = text.lower()

        # Check for specificity patterns (boost)
        for pattern, boost in self.SPECIFICITY_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += boost

        # Check for generic patterns (reduce)
        for pattern, penalty in self.GENERIC_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                score += penalty  # penalty is negative

        return max(0.1, min(1.0, score))

    def _calculate_uniqueness(
        self,
        keywords: List[str],
        keyword_freq: Dict[str, int]
    ) -> float:
        """Calculate uniqueness score (less frequent = more unique)."""
        if not keywords:
            return 0.5

        # Average inverse frequency
        scores = []
        for kw in keywords:
            freq = keyword_freq.get(kw.lower(), 1)
            # Inverse frequency normalized
            inv_freq = 1.0 / freq
            scores.append(inv_freq)

        avg_uniqueness = sum(scores) / len(scores)

        # Scale to 0-1 range
        return min(1.0, avg_uniqueness)

    def _calculate_actionability(self, text: str) -> float:
        """Calculate actionability score (concrete vs abstract)."""
        score = 0.5

        text_lower = text.lower()

        # Actionable indicators
        actionable_patterns = [
            r'(?:enables?|allows?|provides?|delivers?|creates?)',
            r'(?:can|will|should)\s+\w+',
            r'(?:step|process|method|approach)',
            r'(?:use|apply|implement|deploy)',
        ]

        for pattern in actionable_patterns:
            if re.search(pattern, text_lower):
                score += 0.1

        # Abstract/vague indicators
        abstract_patterns = [
            r'(?:might|could|may|possibly)',
            r'(?:concept|idea|vision|philosophy)',
            r'(?:theoretical|hypothetical)',
        ]

        for pattern in abstract_patterns:
            if re.search(pattern, text_lower):
                score -= 0.1

        return max(0.1, min(1.0, score))

    def _classify_salience(self, weight: float) -> SalienceLevel:
        """Classify salience weight into level."""
        if weight >= self.salience_thresholds["critical"]:
            return SalienceLevel.CRITICAL
        elif weight >= self.salience_thresholds["important"]:
            return SalienceLevel.IMPORTANT
        elif weight >= self.salience_thresholds["moderate"]:
            return SalienceLevel.MODERATE
        else:
            return SalienceLevel.LOW

    def _generate_explanation(
        self,
        factors: Dict[str, float],
        level: SalienceLevel
    ) -> str:
        """Generate human-readable explanation of salience."""
        # Find top contributing factors
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        top_factors = sorted_factors[:2]

        factor_desc = {
            "claim_type": "claim type importance",
            "source_position": "source position",
            "specificity": "specific details",
            "uniqueness": "unique keywords",
            "actionability": "actionability",
            "extraction_confidence": "extraction confidence"
        }

        top_names = [factor_desc.get(f[0], f[0]) for f in top_factors]

        return f"{level.value.title()} salience due to {' and '.join(top_names)}"


def weight_claims(claims: List[AtomicClaim]) -> SalienceResult:
    """Convenience function to weight claims."""
    weighter = SalienceWeighter()
    return weighter.weight_claims(claims)
