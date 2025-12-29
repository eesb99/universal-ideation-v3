"""
Atomic Novelty Scorer (NovAScore) for Universal Ideation v3.2

Implements comprehensive atomic novelty evaluation:
- ACU decomposition into atomic claims
- NLI-based novelty detection for each claim
- Salience weighting for importance-adjusted scoring
- Hybrid scoring combining multiple signals

Achieves 0.94 accuracy vs 0.83 for cosine similarity (per research).

Based on NovAScore: Atomic Content Unit decomposition with NLI detection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from datetime import datetime
from enum import Enum
import json

from .acu_decomposer import (
    ACUDecomposer, AtomicClaim, DecompositionResult, ClaimType
)
from .nli_detector import (
    NLINoveltyDetector, NoveltyDetectionResult, ClaimNoveltyResult, NoveltyLevel
)
from .salience_weighter import (
    SalienceWeighter, SalienceResult, SalienceLevel
)


class NoveltyTier(Enum):
    """Overall novelty classification tiers."""
    BREAKTHROUGH = "breakthrough"     # Score >= 90, fundamentally new
    HIGHLY_NOVEL = "highly_novel"     # Score 75-89, significant innovation
    NOVEL = "novel"                   # Score 60-74, meaningful novelty
    INCREMENTAL = "incremental"       # Score 40-59, small improvement
    DERIVATIVE = "derivative"         # Score < 40, mostly restates prior art


@dataclass
class NoveltyBreakdown:
    """Detailed novelty score breakdown."""
    # Component scores (0-100)
    atomic_novelty_score: float  # Raw NLI-based novelty
    salience_weighted_score: float  # Weighted by importance
    structural_novelty_score: float  # Novel claim combinations
    semantic_distance_score: float  # Distance from prior idea space

    # Claim-level statistics
    total_claims: int
    novel_claims: int  # Moderately novel or higher
    highly_novel_claims: int
    contradictory_claims: int

    # Top contributing claims
    top_novel_claims: List[Dict]  # Most novel claims with details


@dataclass
class AtomicNoveltyResult:
    """Complete atomic novelty assessment result."""
    idea_id: str
    idea: Dict

    # Overall scores
    final_score: float  # 0-100, weighted combination
    novelty_tier: NoveltyTier
    confidence: float  # 0-1, confidence in assessment

    # Breakdown
    breakdown: NoveltyBreakdown

    # Component results
    decomposition: DecompositionResult
    novelty_detection: NoveltyDetectionResult
    salience_weighting: SalienceResult

    # Metadata
    processing_time_ms: float
    use_llm: bool

    def get_summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"NovAScore: {self.final_score:.1f}/100 ({self.novelty_tier.value})\n"
            f"Claims: {self.breakdown.total_claims} total, "
            f"{self.breakdown.novel_claims} novel, "
            f"{self.breakdown.highly_novel_claims} highly novel\n"
            f"Confidence: {self.confidence:.0%}"
        )

    def to_dict(self) -> Dict:
        return {
            "idea_id": self.idea_id,
            "final_score": self.final_score,
            "novelty_tier": self.novelty_tier.value,
            "confidence": self.confidence,
            "breakdown": {
                "atomic_novelty_score": self.breakdown.atomic_novelty_score,
                "salience_weighted_score": self.breakdown.salience_weighted_score,
                "structural_novelty_score": self.breakdown.structural_novelty_score,
                "semantic_distance_score": self.breakdown.semantic_distance_score,
                "total_claims": self.breakdown.total_claims,
                "novel_claims": self.breakdown.novel_claims,
                "highly_novel_claims": self.breakdown.highly_novel_claims,
                "top_novel_claims": self.breakdown.top_novel_claims
            },
            "processing_time_ms": self.processing_time_ms
        }


class AtomicNoveltyScorer:
    """
    NovAScore: Atomic novelty evaluation system.

    Combines:
    1. ACU Decomposition - Break idea into atomic claims
    2. NLI Detection - Assess novelty of each claim vs prior corpus
    3. Salience Weighting - Weight claims by importance
    4. Hybrid Scoring - Combine signals for final score
    """

    # Score component weights
    DEFAULT_WEIGHTS = {
        "atomic_novelty": 0.40,      # Raw NLI-based novelty
        "salience_weighted": 0.30,   # Importance-adjusted novelty
        "structural": 0.15,          # Novel claim combinations
        "semantic_distance": 0.15,   # Distance from prior space
    }

    # Tier thresholds
    TIER_THRESHOLDS = {
        NoveltyTier.BREAKTHROUGH: 90,
        NoveltyTier.HIGHLY_NOVEL: 75,
        NoveltyTier.NOVEL: 60,
        NoveltyTier.INCREMENTAL: 40,
    }

    def __init__(
        self,
        prior_ideas: Optional[List[Dict]] = None,
        llm_callback: Optional[Callable[[str], Dict]] = None,
        use_llm: bool = False,
        component_weights: Optional[Dict[str, float]] = None,
        min_claims_for_confidence: int = 3
    ):
        """
        Initialize NovAScore system.

        Args:
            prior_ideas: Prior ideas to compare against
            llm_callback: Optional LLM for enhanced detection
            use_llm: Whether to use LLM for decomposition/detection
            component_weights: Custom weights for score components
            min_claims_for_confidence: Minimum claims for high confidence
        """
        self.llm_callback = llm_callback
        self.use_llm = use_llm and llm_callback is not None
        self.component_weights = component_weights or dict(self.DEFAULT_WEIGHTS)
        self.min_claims_for_confidence = min_claims_for_confidence

        # Initialize components
        self.decomposer = ACUDecomposer(
            llm_callback=llm_callback,
            use_llm=use_llm
        )

        self.salience_weighter = SalienceWeighter()

        # Initialize detector with prior claims
        self.detector = NLINoveltyDetector(
            llm_callback=llm_callback,
            use_llm=use_llm
        )

        # Process prior ideas into claims
        if prior_ideas:
            self._add_prior_ideas(prior_ideas)

        # Track processed ideas
        self.processed_count = 0

    def _add_prior_ideas(self, ideas: List[Dict]):
        """Decompose and add prior ideas to corpus."""
        for idea in ideas:
            decomposition = self.decomposer.decompose(idea)
            self.detector.add_prior_claims(decomposition.claims)

    def add_prior_idea(self, idea: Dict):
        """Add a single prior idea to corpus."""
        decomposition = self.decomposer.decompose(idea)
        self.detector.add_prior_claims(decomposition.claims)

    def score_novelty(
        self,
        idea: Dict,
        idea_id: Optional[str] = None,
        semantic_distance: Optional[float] = None
    ) -> AtomicNoveltyResult:
        """
        Score the atomic novelty of an idea.

        Args:
            idea: Idea dictionary to assess
            idea_id: Optional ID for the idea
            semantic_distance: Optional pre-computed semantic distance

        Returns:
            AtomicNoveltyResult with complete assessment
        """
        start_time = datetime.now()

        if idea_id is None:
            idea_id = f"idea_{self.processed_count}"
            self.processed_count += 1

        # Step 1: Decompose into atomic claims
        decomposition = self.decomposer.decompose(idea, idea_id)

        # Step 2: Detect novelty for each claim
        novelty_detection = self.detector.detect_novelty(
            decomposition.claims, idea_id
        )

        # Step 3: Weight claims by salience
        salience = self.salience_weighter.weight_claims(
            decomposition.claims, idea_id
        )

        # Step 4: Calculate component scores
        atomic_score = self._calculate_atomic_novelty(novelty_detection)
        weighted_score = self._calculate_weighted_novelty(
            novelty_detection, salience
        )
        structural_score = self._calculate_structural_novelty(
            decomposition, novelty_detection
        )
        semantic_score = self._calculate_semantic_score(semantic_distance)

        # Step 5: Combine into final score
        final_score = (
            atomic_score * self.component_weights["atomic_novelty"] +
            weighted_score * self.component_weights["salience_weighted"] +
            structural_score * self.component_weights["structural"] +
            semantic_score * self.component_weights["semantic_distance"]
        )

        # Step 6: Classify tier
        novelty_tier = self._classify_tier(final_score)

        # Step 7: Calculate confidence
        confidence = self._calculate_confidence(
            decomposition, novelty_detection, salience
        )

        # Step 8: Build breakdown
        breakdown = self._build_breakdown(
            atomic_score, weighted_score, structural_score, semantic_score,
            decomposition, novelty_detection
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return AtomicNoveltyResult(
            idea_id=idea_id,
            idea=idea,
            final_score=round(final_score, 1),
            novelty_tier=novelty_tier,
            confidence=round(confidence, 2),
            breakdown=breakdown,
            decomposition=decomposition,
            novelty_detection=novelty_detection,
            salience_weighting=salience,
            processing_time_ms=processing_time,
            use_llm=self.use_llm
        )

    def _calculate_atomic_novelty(
        self,
        detection: NoveltyDetectionResult
    ) -> float:
        """Calculate raw atomic novelty score (unweighted average)."""
        return detection.average_novelty_score

    def _calculate_weighted_novelty(
        self,
        detection: NoveltyDetectionResult,
        salience: SalienceResult
    ) -> float:
        """Calculate salience-weighted novelty score."""
        if not detection.claim_results:
            return 50.0

        weighted_sum = 0.0
        weight_sum = 0.0

        for claim_result in detection.claim_results:
            weight = salience.get_weight(claim_result.claim.id)
            weighted_sum += claim_result.novelty_score * weight
            weight_sum += weight

        if weight_sum == 0:
            return 50.0

        return weighted_sum / weight_sum

    def _calculate_structural_novelty(
        self,
        decomposition: DecompositionResult,
        detection: NoveltyDetectionResult
    ) -> float:
        """
        Calculate structural novelty (novel claim combinations).

        Higher score if novel claims span multiple types/sources.
        """
        if not detection.claim_results:
            return 50.0

        # Get novel claims
        novel_claims = detection.get_novel_claims(NoveltyLevel.MODERATELY_NOVEL)

        if not novel_claims:
            return 30.0  # No novel claims = low structural novelty

        # Check diversity of novel claim types
        novel_types = set()
        novel_sources = set()

        for cr in novel_claims:
            novel_types.add(cr.claim.claim_type)
            novel_sources.add(cr.claim.source)

        # Score based on type diversity (max 5 types)
        type_diversity = min(len(novel_types) / 5, 1.0)

        # Score based on source diversity (max 4 sources)
        source_diversity = min(len(novel_sources) / 4, 1.0)

        # Score based on novel claim ratio
        novel_ratio = len(novel_claims) / len(detection.claim_results)

        # Combine
        structural_score = (
            type_diversity * 35 +
            source_diversity * 25 +
            novel_ratio * 40
        )

        return min(100, structural_score)

    def _calculate_semantic_score(
        self,
        semantic_distance: Optional[float]
    ) -> float:
        """Convert semantic distance to novelty score."""
        if semantic_distance is None:
            return 50.0  # Neutral if not provided

        # Distance 0-1, higher = more novel
        # Convert to 0-100 score with non-linear scaling
        # Low distances (similar) should penalize more
        if semantic_distance < 0.3:
            return semantic_distance * 100  # 0-30
        elif semantic_distance < 0.5:
            return 30 + (semantic_distance - 0.3) * 150  # 30-60
        else:
            return 60 + (semantic_distance - 0.5) * 80  # 60-100

    def _classify_tier(self, score: float) -> NoveltyTier:
        """Classify score into novelty tier."""
        if score >= self.TIER_THRESHOLDS[NoveltyTier.BREAKTHROUGH]:
            return NoveltyTier.BREAKTHROUGH
        elif score >= self.TIER_THRESHOLDS[NoveltyTier.HIGHLY_NOVEL]:
            return NoveltyTier.HIGHLY_NOVEL
        elif score >= self.TIER_THRESHOLDS[NoveltyTier.NOVEL]:
            return NoveltyTier.NOVEL
        elif score >= self.TIER_THRESHOLDS[NoveltyTier.INCREMENTAL]:
            return NoveltyTier.INCREMENTAL
        else:
            return NoveltyTier.DERIVATIVE

    def _calculate_confidence(
        self,
        decomposition: DecompositionResult,
        detection: NoveltyDetectionResult,
        salience: SalienceResult
    ) -> float:
        """Calculate confidence in the assessment."""
        confidence = 1.0

        # Reduce confidence if few claims
        if decomposition.total_claims < self.min_claims_for_confidence:
            confidence *= 0.7

        # Reduce confidence if many low-confidence extractions
        low_confidence_claims = sum(
            1 for c in decomposition.claims if c.confidence < 0.7
        )
        if low_confidence_claims > decomposition.total_claims * 0.3:
            confidence *= 0.8

        # Reduce confidence if no prior claims to compare against
        if len(self.detector.prior_claims) < 5:
            confidence *= 0.8

        # Reduce confidence if mostly unknown claim types
        unknown_count = sum(
            1 for c in decomposition.claims
            if c.claim_type == ClaimType.UNKNOWN
        )
        if unknown_count > decomposition.total_claims * 0.5:
            confidence *= 0.7

        return max(0.3, confidence)

    def _build_breakdown(
        self,
        atomic_score: float,
        weighted_score: float,
        structural_score: float,
        semantic_score: float,
        decomposition: DecompositionResult,
        detection: NoveltyDetectionResult
    ) -> NoveltyBreakdown:
        """Build detailed breakdown."""
        # Count novel claims
        novel_claims = detection.get_novel_claims(NoveltyLevel.MODERATELY_NOVEL)
        highly_novel = [
            cr for cr in novel_claims
            if cr.novelty_level == NoveltyLevel.HIGHLY_NOVEL
        ]

        # Get top novel claims for display
        sorted_claims = sorted(
            detection.claim_results,
            key=lambda x: x.novelty_score,
            reverse=True
        )
        top_novel = [
            {
                "claim": cr.claim.text[:100],
                "novelty_score": cr.novelty_score,
                "level": cr.novelty_level.value,
                "type": cr.claim.claim_type.value
            }
            for cr in sorted_claims[:5]
        ]

        return NoveltyBreakdown(
            atomic_novelty_score=round(atomic_score, 1),
            salience_weighted_score=round(weighted_score, 1),
            structural_novelty_score=round(structural_score, 1),
            semantic_distance_score=round(semantic_score, 1),
            total_claims=decomposition.total_claims,
            novel_claims=len(novel_claims),
            highly_novel_claims=len(highly_novel),
            contradictory_claims=detection.contradictions_found,
            top_novel_claims=top_novel
        )

    def get_statistics(self) -> Dict:
        """Get scorer statistics."""
        return {
            "prior_ideas_processed": len(self.detector.prior_claims),
            "ideas_scored": self.processed_count,
            "use_llm": self.use_llm,
            "component_weights": self.component_weights,
            "detector_stats": self.detector.get_statistics()
        }

    def export_prior_claims(self) -> List[Dict]:
        """Export prior claims for persistence."""
        return [c.to_dict() for c in self.detector.prior_claims]


def score_idea_novelty(
    idea: Dict,
    prior_ideas: Optional[List[Dict]] = None,
    semantic_distance: Optional[float] = None
) -> AtomicNoveltyResult:
    """Convenience function to score a single idea."""
    scorer = AtomicNoveltyScorer(prior_ideas=prior_ideas)
    return scorer.score_novelty(idea, semantic_distance=semantic_distance)


def create_test_scorer() -> AtomicNoveltyScorer:
    """Create scorer with sample prior ideas."""
    prior_ideas = [
        {
            "title": "Personalized Protein Recommendations",
            "description": "AI-based system that recommends protein products based on user profile",
            "differentiators": ["Machine learning recommendations", "User profile analysis"]
        },
        {
            "title": "Fitness Tracker Integration Platform",
            "description": "Connect fitness wearables to nutrition apps",
            "differentiators": ["Wearable sync", "Activity-based nutrition"]
        },
    ]

    return AtomicNoveltyScorer(prior_ideas=prior_ideas)
