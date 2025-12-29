"""
Reflection Generator for Universal Ideation v3.2

Implements ReflectEvo-style batch reflection analysis:
- Analyzes batches of ideas to extract success/failure patterns
- Identifies dimension correlations with high scores
- Generates actionable learning statements
- Supports cross-session learning through persistence

Based on ReflectEvo 2024: Self-improvement through reflection-based learning.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
import json


class ReflectionType(Enum):
    """Types of reflections generated."""
    SUCCESS_PATTERN = "success_pattern"      # What worked well
    FAILURE_PATTERN = "failure_pattern"      # What didn't work
    DIMENSION_INSIGHT = "dimension_insight"  # Dimension-specific learning
    CORRELATION = "correlation"              # Cross-dimension relationships
    ANTI_PATTERN = "anti_pattern"            # Patterns to avoid


class ReflectionConfidence(Enum):
    """Confidence level of reflection."""
    HIGH = "high"        # Strong evidence (5+ ideas)
    MEDIUM = "medium"    # Moderate evidence (3-4 ideas)
    LOW = "low"          # Weak evidence (1-2 ideas)


@dataclass
class Reflection:
    """A single learning reflection extracted from idea analysis."""
    id: str
    type: ReflectionType
    pattern: str                    # The observed pattern
    observation: str                # What was observed
    evidence_count: int             # Number of ideas supporting this
    confidence: ReflectionConfidence
    dimension_impacts: Dict[str, float]  # Suggested weight adjustments
    created_at: str
    domain: str
    idea_ids: List[str] = field(default_factory=list)  # Supporting idea IDs

    def to_dict(self) -> Dict:
        """Convert to dictionary for persistence."""
        return {
            "id": self.id,
            "type": self.type.value,
            "pattern": self.pattern,
            "observation": self.observation,
            "evidence_count": self.evidence_count,
            "confidence": self.confidence.value,
            "dimension_impacts": self.dimension_impacts,
            "created_at": self.created_at,
            "domain": self.domain,
            "idea_ids": self.idea_ids
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Reflection":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            type=ReflectionType(data["type"]),
            pattern=data["pattern"],
            observation=data["observation"],
            evidence_count=data["evidence_count"],
            confidence=ReflectionConfidence(data["confidence"]),
            dimension_impacts=data["dimension_impacts"],
            created_at=data["created_at"],
            domain=data["domain"],
            idea_ids=data.get("idea_ids", [])
        )


@dataclass
class ReflectionBatch:
    """A batch of reflections from a single analysis session."""
    session_id: str
    domain: str
    reflections: List[Reflection]
    ideas_analyzed: int
    high_score_threshold: float
    low_score_threshold: float
    created_at: str

    def get_success_patterns(self) -> List[Reflection]:
        """Get all success pattern reflections."""
        return [r for r in self.reflections
                if r.type == ReflectionType.SUCCESS_PATTERN]

    def get_failure_patterns(self) -> List[Reflection]:
        """Get all failure pattern reflections."""
        return [r for r in self.reflections
                if r.type == ReflectionType.FAILURE_PATTERN]

    def get_high_confidence(self) -> List[Reflection]:
        """Get high-confidence reflections only."""
        return [r for r in self.reflections
                if r.confidence == ReflectionConfidence.HIGH]


@dataclass
class IdeaForReflection:
    """Simplified idea representation for reflection analysis."""
    id: str
    title: str
    dimension_scores: Dict[str, float]
    final_score: float
    generator_mode: str
    accepted: bool
    domain: str


class ReflectionGenerator:
    """
    Generates learning reflections from idea batches.

    Key capabilities:
    - Batch analysis of 5-10 ideas
    - Success/failure pattern extraction
    - Dimension correlation detection
    - Weight adjustment suggestions
    """

    # Dimension names for analysis
    DIMENSIONS = [
        "novelty", "feasibility", "market", "complexity",
        "scenario", "contrarian", "surprise", "cross_domain"
    ]

    # Thresholds for pattern detection
    HIGH_SCORE_THRESHOLD = 75.0
    LOW_SCORE_THRESHOLD = 55.0
    PATTERN_MIN_EVIDENCE = 3  # Minimum ideas to establish pattern

    def __init__(
        self,
        domain: str = "general",
        high_score_threshold: float = 75.0,
        low_score_threshold: float = 55.0,
        min_batch_size: int = 5
    ):
        """
        Initialize reflection generator.

        Args:
            domain: Domain context for reflections
            high_score_threshold: Score above which ideas are "successful"
            low_score_threshold: Score below which ideas are "failures"
            min_batch_size: Minimum ideas needed for analysis
        """
        self.domain = domain
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.min_batch_size = min_batch_size

        # Statistics
        self.total_reflections_generated = 0
        self.batches_analyzed = 0

    def analyze_batch(
        self,
        ideas: List[IdeaForReflection],
        session_id: Optional[str] = None
    ) -> ReflectionBatch:
        """
        Analyze a batch of ideas and generate reflections.

        Args:
            ideas: List of ideas to analyze
            session_id: Optional session identifier

        Returns:
            ReflectionBatch with all generated reflections
        """
        if len(ideas) < self.min_batch_size:
            # Still generate what we can with limited data
            pass

        session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        reflections = []

        # Separate ideas by performance
        high_performers = [i for i in ideas if i.final_score >= self.high_score_threshold]
        low_performers = [i for i in ideas if i.final_score <= self.low_score_threshold]
        all_accepted = [i for i in ideas if i.accepted]

        # 1. Generate success patterns
        if len(high_performers) >= 2:
            success_reflections = self._analyze_success_patterns(high_performers)
            reflections.extend(success_reflections)

        # 2. Generate failure patterns
        if len(low_performers) >= 2:
            failure_reflections = self._analyze_failure_patterns(low_performers)
            reflections.extend(failure_reflections)

        # 3. Generate dimension insights
        dimension_reflections = self._analyze_dimensions(ideas)
        reflections.extend(dimension_reflections)

        # 4. Generate correlations
        if len(ideas) >= 5:
            correlation_reflections = self._analyze_correlations(ideas)
            reflections.extend(correlation_reflections)

        # 5. Generate mode-specific insights
        mode_reflections = self._analyze_generator_modes(ideas)
        reflections.extend(mode_reflections)

        self.batches_analyzed += 1
        self.total_reflections_generated += len(reflections)

        return ReflectionBatch(
            session_id=session_id,
            domain=self.domain,
            reflections=reflections,
            ideas_analyzed=len(ideas),
            high_score_threshold=self.high_score_threshold,
            low_score_threshold=self.low_score_threshold,
            created_at=datetime.now().isoformat()
        )

    def _analyze_success_patterns(
        self,
        high_performers: List[IdeaForReflection]
    ) -> List[Reflection]:
        """Extract patterns from successful ideas."""
        reflections = []

        # Find common high dimensions
        dim_averages = self._calculate_dimension_averages(high_performers)
        top_dims = sorted(dim_averages.items(), key=lambda x: x[1], reverse=True)[:3]

        for dim, avg_score in top_dims:
            if avg_score >= 70:  # Significant contributor
                reflection = Reflection(
                    id=self._generate_id("success"),
                    type=ReflectionType.SUCCESS_PATTERN,
                    pattern=f"High {dim} scores correlate with success",
                    observation=f"Top-performing ideas averaged {avg_score:.1f} in {dim} dimension",
                    evidence_count=len(high_performers),
                    confidence=self._determine_confidence(len(high_performers)),
                    dimension_impacts={dim: 0.02},  # Suggest +2% weight
                    created_at=datetime.now().isoformat(),
                    domain=self.domain,
                    idea_ids=[i.id for i in high_performers]
                )
                reflections.append(reflection)

        # Analyze generator mode success
        mode_counts = {}
        for idea in high_performers:
            mode_counts[idea.generator_mode] = mode_counts.get(idea.generator_mode, 0) + 1

        if mode_counts:
            top_mode = max(mode_counts, key=mode_counts.get)
            if mode_counts[top_mode] >= 2:
                reflection = Reflection(
                    id=self._generate_id("success_mode"),
                    type=ReflectionType.SUCCESS_PATTERN,
                    pattern=f"{top_mode} mode produces high-quality ideas",
                    observation=f"{mode_counts[top_mode]}/{len(high_performers)} top ideas came from {top_mode} mode",
                    evidence_count=mode_counts[top_mode],
                    confidence=self._determine_confidence(mode_counts[top_mode]),
                    dimension_impacts={},  # Mode adjustment handled separately
                    created_at=datetime.now().isoformat(),
                    domain=self.domain,
                    idea_ids=[i.id for i in high_performers if i.generator_mode == top_mode]
                )
                reflections.append(reflection)

        return reflections

    def _analyze_failure_patterns(
        self,
        low_performers: List[IdeaForReflection]
    ) -> List[Reflection]:
        """Extract patterns from failed ideas."""
        reflections = []

        # Find common low dimensions
        dim_averages = self._calculate_dimension_averages(low_performers)
        bottom_dims = sorted(dim_averages.items(), key=lambda x: x[1])[:3]

        for dim, avg_score in bottom_dims:
            if avg_score <= 50:  # Significant weakness
                reflection = Reflection(
                    id=self._generate_id("failure"),
                    type=ReflectionType.FAILURE_PATTERN,
                    pattern=f"Low {dim} scores indicate quality issues",
                    observation=f"Failed ideas averaged only {avg_score:.1f} in {dim} dimension",
                    evidence_count=len(low_performers),
                    confidence=self._determine_confidence(len(low_performers)),
                    dimension_impacts={dim: 0.01},  # Suggest +1% weight (prioritize fixing weakness)
                    created_at=datetime.now().isoformat(),
                    domain=self.domain,
                    idea_ids=[i.id for i in low_performers]
                )
                reflections.append(reflection)

        return reflections

    def _analyze_dimensions(
        self,
        ideas: List[IdeaForReflection]
    ) -> List[Reflection]:
        """Generate dimension-specific insights."""
        reflections = []

        # Calculate correlation with final score
        for dim in self.DIMENSIONS:
            dim_scores = [i.dimension_scores.get(dim, 0) for i in ideas]
            final_scores = [i.final_score for i in ideas]

            if len(dim_scores) >= 3:
                correlation = self._calculate_correlation(dim_scores, final_scores)

                if abs(correlation) >= 0.5:  # Significant correlation
                    direction = "positive" if correlation > 0 else "negative"
                    impact = 0.02 if correlation > 0.7 else 0.01

                    reflection = Reflection(
                        id=self._generate_id("dimension"),
                        type=ReflectionType.DIMENSION_INSIGHT,
                        pattern=f"{dim} has {direction} correlation with success",
                        observation=f"Correlation coefficient: {correlation:.2f} across {len(ideas)} ideas",
                        evidence_count=len(ideas),
                        confidence=self._determine_confidence(len(ideas)),
                        dimension_impacts={dim: impact if correlation > 0 else -impact},
                        created_at=datetime.now().isoformat(),
                        domain=self.domain,
                        idea_ids=[i.id for i in ideas]
                    )
                    reflections.append(reflection)

        return reflections

    def _analyze_correlations(
        self,
        ideas: List[IdeaForReflection]
    ) -> List[Reflection]:
        """Find correlations between dimensions."""
        reflections = []

        # Check pairs of dimensions
        for i, dim1 in enumerate(self.DIMENSIONS):
            for dim2 in self.DIMENSIONS[i+1:]:
                scores1 = [idea.dimension_scores.get(dim1, 0) for idea in ideas]
                scores2 = [idea.dimension_scores.get(dim2, 0) for idea in ideas]

                correlation = self._calculate_correlation(scores1, scores2)

                if correlation >= 0.7:  # Strong positive correlation
                    reflection = Reflection(
                        id=self._generate_id("correlation"),
                        type=ReflectionType.CORRELATION,
                        pattern=f"{dim1} and {dim2} are positively linked",
                        observation=f"Improving {dim1} tends to improve {dim2} (r={correlation:.2f})",
                        evidence_count=len(ideas),
                        confidence=self._determine_confidence(len(ideas)),
                        dimension_impacts={},  # Informational only
                        created_at=datetime.now().isoformat(),
                        domain=self.domain,
                        idea_ids=[i.id for i in ideas]
                    )
                    reflections.append(reflection)
                elif correlation <= -0.5:  # Negative correlation (trade-off)
                    reflection = Reflection(
                        id=self._generate_id("tradeoff"),
                        type=ReflectionType.CORRELATION,
                        pattern=f"{dim1} and {dim2} show trade-off",
                        observation=f"Increasing {dim1} may decrease {dim2} (r={correlation:.2f})",
                        evidence_count=len(ideas),
                        confidence=self._determine_confidence(len(ideas)),
                        dimension_impacts={},  # Informational only
                        created_at=datetime.now().isoformat(),
                        domain=self.domain,
                        idea_ids=[i.id for i in ideas]
                    )
                    reflections.append(reflection)

        return reflections

    def _analyze_generator_modes(
        self,
        ideas: List[IdeaForReflection]
    ) -> List[Reflection]:
        """Analyze performance by generator mode."""
        reflections = []

        mode_scores = {}
        mode_counts = {}

        for idea in ideas:
            mode = idea.generator_mode
            if mode not in mode_scores:
                mode_scores[mode] = []
                mode_counts[mode] = 0
            mode_scores[mode].append(idea.final_score)
            mode_counts[mode] += 1

        # Compare mode performance
        mode_averages = {m: sum(s)/len(s) for m, s in mode_scores.items() if s}

        if len(mode_averages) >= 2:
            best_mode = max(mode_averages, key=mode_averages.get)
            worst_mode = min(mode_averages, key=mode_averages.get)

            if mode_averages[best_mode] - mode_averages[worst_mode] >= 5:
                reflection = Reflection(
                    id=self._generate_id("mode_comparison"),
                    type=ReflectionType.DIMENSION_INSIGHT,
                    pattern=f"{best_mode} mode outperforms {worst_mode} mode",
                    observation=f"{best_mode}: avg {mode_averages[best_mode]:.1f} vs {worst_mode}: avg {mode_averages[worst_mode]:.1f}",
                    evidence_count=mode_counts[best_mode] + mode_counts[worst_mode],
                    confidence=self._determine_confidence(mode_counts[best_mode]),
                    dimension_impacts={},  # Mode adjustments handled separately
                    created_at=datetime.now().isoformat(),
                    domain=self.domain,
                    idea_ids=[i.id for i in ideas]
                )
                reflections.append(reflection)

        return reflections

    def _calculate_dimension_averages(
        self,
        ideas: List[IdeaForReflection]
    ) -> Dict[str, float]:
        """Calculate average score for each dimension."""
        totals = {dim: 0.0 for dim in self.DIMENSIONS}
        counts = {dim: 0 for dim in self.DIMENSIONS}

        for idea in ideas:
            for dim in self.DIMENSIONS:
                if dim in idea.dimension_scores:
                    totals[dim] += idea.dimension_scores[dim]
                    counts[dim] += 1

        return {
            dim: totals[dim] / counts[dim] if counts[dim] > 0 else 0
            for dim in self.DIMENSIONS
        }

    def _calculate_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denominator = (var_x * var_y) ** 0.5

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def _determine_confidence(self, evidence_count: int) -> ReflectionConfidence:
        """Determine confidence level based on evidence count."""
        if evidence_count >= 5:
            return ReflectionConfidence.HIGH
        elif evidence_count >= 3:
            return ReflectionConfidence.MEDIUM
        else:
            return ReflectionConfidence.LOW

    def _generate_id(self, prefix: str) -> str:
        """Generate unique reflection ID."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        return f"{prefix}_{timestamp}"

    def get_statistics(self) -> Dict:
        """Get generator statistics."""
        return {
            "total_reflections_generated": self.total_reflections_generated,
            "batches_analyzed": self.batches_analyzed,
            "domain": self.domain,
            "thresholds": {
                "high_score": self.high_score_threshold,
                "low_score": self.low_score_threshold
            }
        }


def create_test_ideas(n: int = 10, domain: str = "protein beverages") -> List[IdeaForReflection]:
    """Create test ideas for reflection analysis."""
    import random

    ideas = []
    modes = ["explorer", "refiner", "contrarian"]

    for i in range(n):
        scores = {
            "novelty": random.uniform(40, 90),
            "feasibility": random.uniform(50, 85),
            "market": random.uniform(45, 88),
            "complexity": random.uniform(40, 80),
            "scenario": random.uniform(50, 85),
            "contrarian": random.uniform(35, 80),
            "surprise": random.uniform(30, 75),
            "cross_domain": random.uniform(25, 70)
        }
        final_score = sum(scores.values()) / len(scores)

        ideas.append(IdeaForReflection(
            id=f"idea_{i+1}",
            title=f"Test Idea {i+1}",
            dimension_scores=scores,
            final_score=final_score,
            generator_mode=random.choice(modes),
            accepted=final_score >= 65,
            domain=domain
        ))

    return ideas
