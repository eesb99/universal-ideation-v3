"""
Surprise Dimension for Universal Ideation v3

Measures schema violation and expectation deviation.
Novelty != Surprise. Novelty is statistical rarity; Surprise is schema violation.

Based on schema violation studies in creativity science.
An idea can be novel (statistically rare) but not surprising (fits expected patterns).
A truly surprising idea breaks mental models and expectations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import re


class ConventionCategory(Enum):
    """Categories of domain conventions that can be violated."""
    TARGET_AUDIENCE = "target_audience"
    DISTRIBUTION = "distribution"
    PRICE_POINT = "price_point"
    FORMAT = "format"
    POSITIONING = "positioning"
    USAGE_OCCASION = "usage_occasion"
    INGREDIENT_BASE = "ingredient_base"
    BENEFIT_CLAIM = "benefit_claim"


@dataclass
class DomainConvention:
    """A convention within a domain."""
    category: ConventionCategory
    expected_value: str
    alternatives: List[str]  # Non-conventional alternatives
    violation_weight: float = 1.0  # Some violations are more surprising


@dataclass
class SchemaViolation:
    """A detected schema violation."""
    category: ConventionCategory
    expected: str
    actual: str
    violation_score: float  # 0-1, how severe the violation
    explanation: str


@dataclass
class SurpriseAnalysis:
    """Complete surprise analysis result."""
    schema_score: float  # 0-100: How many conventions violated
    expectation_score: float  # 0-100: How unexpected vs predictions
    final_score: float  # Weighted combination
    violations: List[SchemaViolation]
    violated_categories: List[str]
    surprise_factors: List[str]  # Human-readable surprise elements


class DomainConventions:
    """
    Defines expected conventions for a domain.

    These represent the "schema" that most ideas in the domain follow.
    Surprising ideas violate these expectations.
    """

    # Default conventions for protein/nutrition domain
    PROTEIN_BEVERAGE_CONVENTIONS = {
        ConventionCategory.TARGET_AUDIENCE: DomainConvention(
            category=ConventionCategory.TARGET_AUDIENCE,
            expected_value="fitness enthusiasts, gym-goers",
            alternatives=["elderly", "children", "office workers", "gamers",
                         "pregnant women", "shift workers", "religious fasters"],
            violation_weight=1.2
        ),
        ConventionCategory.DISTRIBUTION: DomainConvention(
            category=ConventionCategory.DISTRIBUTION,
            expected_value="specialty stores, gyms, online",
            alternatives=["convenience stores", "vending machines", "hospitals",
                         "schools", "gas stations", "subscription boxes"],
            violation_weight=1.0
        ),
        ConventionCategory.PRICE_POINT: DomainConvention(
            category=ConventionCategory.PRICE_POINT,
            expected_value="premium ($3-5 per serving)",
            alternatives=["ultra-budget (<$1)", "ultra-premium (>$10)",
                         "freemium model", "pay-what-you-want"],
            violation_weight=0.8
        ),
        ConventionCategory.FORMAT: DomainConvention(
            category=ConventionCategory.FORMAT,
            expected_value="powder, ready-to-drink bottle",
            alternatives=["gummy", "bar format", "frozen", "spray",
                         "patch", "capsule", "infused food"],
            violation_weight=1.3
        ),
        ConventionCategory.POSITIONING: DomainConvention(
            category=ConventionCategory.POSITIONING,
            expected_value="performance, muscle building",
            alternatives=["relaxation", "beauty", "cognitive",
                         "social experience", "ritual", "medical"],
            violation_weight=1.1
        ),
        ConventionCategory.USAGE_OCCASION: DomainConvention(
            category=ConventionCategory.USAGE_OCCASION,
            expected_value="post-workout, morning",
            alternatives=["before bed", "during meetings", "social occasions",
                         "religious observance", "travel", "emergency"],
            violation_weight=1.0
        ),
        ConventionCategory.INGREDIENT_BASE: DomainConvention(
            category=ConventionCategory.INGREDIENT_BASE,
            expected_value="whey, pea, soy protein",
            alternatives=["insect", "algae", "mycoprotein", "lab-grown",
                         "ancient grains", "fermented"],
            violation_weight=1.4
        ),
        ConventionCategory.BENEFIT_CLAIM: DomainConvention(
            category=ConventionCategory.BENEFIT_CLAIM,
            expected_value="muscle recovery, protein intake",
            alternatives=["mood enhancement", "sleep quality", "skin health",
                         "longevity", "environmental impact", "community"],
            violation_weight=1.0
        )
    }

    # Generic conventions for any domain
    GENERIC_CONVENTIONS = {
        ConventionCategory.TARGET_AUDIENCE: DomainConvention(
            category=ConventionCategory.TARGET_AUDIENCE,
            expected_value="mainstream consumers",
            alternatives=["niche segments", "B2B", "underserved populations"],
            violation_weight=1.0
        ),
        ConventionCategory.DISTRIBUTION: DomainConvention(
            category=ConventionCategory.DISTRIBUTION,
            expected_value="traditional retail",
            alternatives=["direct-to-consumer", "platform", "embedded"],
            violation_weight=1.0
        ),
        ConventionCategory.PRICE_POINT: DomainConvention(
            category=ConventionCategory.PRICE_POINT,
            expected_value="market average",
            alternatives=["freemium", "premium", "dynamic pricing"],
            violation_weight=0.8
        ),
        ConventionCategory.FORMAT: DomainConvention(
            category=ConventionCategory.FORMAT,
            expected_value="standard format",
            alternatives=["novel format", "hybrid", "service"],
            violation_weight=1.2
        ),
        ConventionCategory.POSITIONING: DomainConvention(
            category=ConventionCategory.POSITIONING,
            expected_value="functional benefit",
            alternatives=["emotional", "identity", "experience"],
            violation_weight=1.0
        )
    }

    @classmethod
    def get_conventions(cls, domain: str) -> Dict[ConventionCategory, DomainConvention]:
        """Get conventions for a domain."""
        domain_lower = domain.lower()

        if any(term in domain_lower for term in ['protein', 'beverage', 'nutrition', 'supplement']):
            return cls.PROTEIN_BEVERAGE_CONVENTIONS
        else:
            return cls.GENERIC_CONVENTIONS


class SurpriseDimension:
    """
    Calculates surprise score based on schema violations.

    Formula:
    Surprise = (Schema Violations × 0.6) + (Expectation Deviation × 0.4)

    Schema Violations: How many domain conventions are broken
    Expectation Deviation: How different from "predicted next idea"
    """

    SCHEMA_WEIGHT = 0.6
    EXPECTATION_WEIGHT = 0.4

    def __init__(self, domain: str = "protein beverages"):
        self.domain = domain
        self.conventions = DomainConventions.get_conventions(domain)
        self.prediction_history: List[str] = []  # What evaluators predicted
        self.analysis_history: List[SurpriseAnalysis] = []

    def detect_violations(self, idea: Dict) -> List[SchemaViolation]:
        """
        Detect which conventions the idea violates.

        Args:
            idea: Dictionary with idea details (title, description, target_market, etc.)

        Returns:
            List of detected violations
        """
        violations = []
        idea_text = self._extract_idea_text(idea)

        for category, convention in self.conventions.items():
            violation = self._check_violation(idea_text, idea, convention)
            if violation:
                violations.append(violation)

        return violations

    def _extract_idea_text(self, idea: Dict) -> str:
        """Extract searchable text from idea."""
        parts = []
        for key in ['title', 'description', 'target_market', 'differentiators',
                    'format', 'positioning', 'distribution']:
            if key in idea:
                value = idea[key]
                if isinstance(value, list):
                    parts.extend(value)
                else:
                    parts.append(str(value))
        return ' '.join(parts).lower()

    def _check_violation(
        self,
        idea_text: str,
        idea: Dict,
        convention: DomainConvention
    ) -> Optional[SchemaViolation]:
        """Check if idea violates a specific convention."""

        # Check if any alternative (non-conventional) element is present
        for alternative in convention.alternatives:
            if alternative.lower() in idea_text:
                # Found a violation - idea uses non-conventional element
                return SchemaViolation(
                    category=convention.category,
                    expected=convention.expected_value,
                    actual=alternative,
                    violation_score=convention.violation_weight,
                    explanation=f"Uses '{alternative}' instead of expected '{convention.expected_value}'"
                )

        # Check specific fields for violations
        if convention.category == ConventionCategory.TARGET_AUDIENCE:
            target = idea.get('target_market', '').lower()
            if target and not any(exp in target for exp in ['fitness', 'gym', 'athlete', 'sport']):
                # Non-standard target audience
                return SchemaViolation(
                    category=convention.category,
                    expected=convention.expected_value,
                    actual=target,
                    violation_score=convention.violation_weight * 0.8,
                    explanation=f"Targets '{target}' instead of fitness enthusiasts"
                )

        return None

    def calculate_schema_score(self, violations: List[SchemaViolation]) -> float:
        """
        Calculate schema violation score.

        More violations = higher surprise (up to a point).
        """
        if not violations:
            return 30.0  # No violations = not surprising

        # Sum weighted violation scores
        total_weight = sum(v.violation_score for v in violations)

        # Normalize: 1 violation ≈ 50, 3+ violations ≈ 90-100
        # Using logarithmic scaling for diminishing returns
        num_violations = len(violations)

        if num_violations == 1:
            base_score = 50
        elif num_violations == 2:
            base_score = 70
        elif num_violations == 3:
            base_score = 85
        else:
            base_score = min(100, 85 + (num_violations - 3) * 5)

        # Adjust by violation weights
        weight_modifier = total_weight / num_violations if num_violations > 0 else 1.0

        return min(100, base_score * weight_modifier)

    def calculate_expectation_score(
        self,
        idea: Dict,
        prior_ideas: List[Dict],
        evaluator_predictions: Optional[List[str]] = None
    ) -> float:
        """
        Calculate how unexpected this idea is.

        Compares idea to:
        1. Trajectory of prior ideas (what would logically come next?)
        2. Evaluator predictions (what did they expect?)
        """
        if not prior_ideas:
            return 50.0  # No baseline = neutral expectation

        idea_text = self._extract_idea_text(idea)

        # Check similarity to "logical next step" from prior ideas
        # If very similar to prior ideas, it was expected
        prior_texts = [self._extract_idea_text(p) for p in prior_ideas[-5:]]

        # Simple word overlap as proxy for similarity
        idea_words = set(idea_text.split())
        prior_words = set()
        for pt in prior_texts:
            prior_words.update(pt.split())

        overlap = len(idea_words & prior_words) / max(len(idea_words), 1)

        # High overlap = expected = low score
        # Low overlap = unexpected = high score
        trajectory_score = (1 - overlap) * 100

        # If evaluator predictions provided, check against those
        prediction_score = 50.0
        if evaluator_predictions:
            pred_text = ' '.join(evaluator_predictions).lower()
            pred_overlap = len(idea_words & set(pred_text.split())) / max(len(idea_words), 1)
            prediction_score = (1 - pred_overlap) * 100

        # Combine: trajectory 60%, predictions 40%
        return trajectory_score * 0.6 + prediction_score * 0.4

    def analyze(
        self,
        idea: Dict,
        prior_ideas: Optional[List[Dict]] = None,
        evaluator_predictions: Optional[List[str]] = None
    ) -> SurpriseAnalysis:
        """
        Perform complete surprise analysis.

        Args:
            idea: The idea to analyze
            prior_ideas: Previous ideas for trajectory comparison
            evaluator_predictions: What evaluators predicted would come next

        Returns:
            SurpriseAnalysis with scores and details
        """
        # Detect schema violations
        violations = self.detect_violations(idea)

        # Calculate component scores
        schema_score = self.calculate_schema_score(violations)
        expectation_score = self.calculate_expectation_score(
            idea,
            prior_ideas or [],
            evaluator_predictions
        )

        # Final weighted score
        final_score = (
            schema_score * self.SCHEMA_WEIGHT +
            expectation_score * self.EXPECTATION_WEIGHT
        )

        # Extract surprise factors
        surprise_factors = []
        for v in violations:
            surprise_factors.append(
                f"Breaks {v.category.value} convention: {v.explanation}"
            )

        if expectation_score > 70:
            surprise_factors.append("Deviates significantly from expected trajectory")

        analysis = SurpriseAnalysis(
            schema_score=schema_score,
            expectation_score=expectation_score,
            final_score=final_score,
            violations=violations,
            violated_categories=[v.category.value for v in violations],
            surprise_factors=surprise_factors
        )

        self.analysis_history.append(analysis)
        return analysis

    def get_evaluation_prompt(self, idea: Dict, domain: str) -> str:
        """
        Generate prompt for agent-based surprise evaluation.

        Used when programmatic detection isn't sufficient.
        """
        conventions_text = "\n".join([
            f"- {cat.value}: Typically '{conv.expected_value}'"
            for cat, conv in self.conventions.items()
        ])

        return f"""Evaluate the SURPRISE level of this idea.

DOMAIN: {domain}

DOMAIN CONVENTIONS (what's typically expected):
{conventions_text}

IDEA:
{idea}

SURPRISE EVALUATION:
Score how surprising this idea is (0-100) based on:

1. SCHEMA VIOLATIONS (60% weight):
   - How many domain conventions does it break?
   - Are the violations significant or superficial?
   - Does it challenge fundamental assumptions?

2. EXPECTATION DEVIATION (40% weight):
   - Would experts have predicted this approach?
   - Does it come from an unexpected direction?
   - Is it a logical extension or a paradigm shift?

OUTPUT FORMAT (JSON):
{{
    "schema_score": <0-100>,
    "expectation_score": <0-100>,
    "final_score": <0-100>,
    "conventions_violated": ["category1", "category2"],
    "surprise_factors": ["factor1", "factor2"],
    "reasoning": "2-3 sentences explaining the surprise level"
}}
"""

    def get_statistics(self) -> Dict:
        """Get surprise analysis statistics."""
        if not self.analysis_history:
            return {"total_analyses": 0}

        scores = [a.final_score for a in self.analysis_history]
        violation_counts = [len(a.violations) for a in self.analysis_history]

        return {
            "total_analyses": len(self.analysis_history),
            "average_surprise": sum(scores) / len(scores),
            "max_surprise": max(scores),
            "min_surprise": min(scores),
            "average_violations": sum(violation_counts) / len(violation_counts),
            "high_surprise_count": sum(1 for s in scores if s > 75),
            "low_surprise_count": sum(1 for s in scores if s < 40)
        }


def calculate_surprise(
    idea: Dict,
    domain: str = "protein beverages",
    prior_ideas: Optional[List[Dict]] = None
) -> float:
    """
    Quick helper to calculate surprise score.

    Args:
        idea: Idea dictionary
        domain: Domain context
        prior_ideas: Previous ideas for comparison

    Returns:
        Surprise score 0-100
    """
    analyzer = SurpriseDimension(domain)
    analysis = analyzer.analyze(idea, prior_ideas)
    return analysis.final_score
