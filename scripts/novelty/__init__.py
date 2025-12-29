"""
Novelty module for Universal Ideation v3.2

Provides NovAScore atomic novelty evaluation:
- ACUDecomposer: Break ideas into atomic content units
- NLINoveltyDetector: NLI-based novelty detection (0.94 accuracy)
- SalienceWeighter: Importance-weighted claim scoring
- AtomicNoveltyScorer: Combined NovAScore system
"""

from .acu_decomposer import (
    ACUDecomposer,
    AtomicClaim,
    DecompositionResult,
    ClaimType,
    ClaimSource,
    decompose_idea,
    create_test_idea
)

from .nli_detector import (
    NLINoveltyDetector,
    NLIRelation,
    NLIResult,
    NoveltyLevel,
    ClaimNoveltyResult,
    NoveltyDetectionResult,
    create_test_detector
)

from .salience_weighter import (
    SalienceWeighter,
    SalienceLevel,
    SalienceScore,
    SalienceResult,
    weight_claims
)

from .atomic_novelty import (
    AtomicNoveltyScorer,
    AtomicNoveltyResult,
    NoveltyBreakdown,
    NoveltyTier,
    score_idea_novelty,
    create_test_scorer
)

__all__ = [
    # ACU Decomposer
    "ACUDecomposer",
    "AtomicClaim",
    "DecompositionResult",
    "ClaimType",
    "ClaimSource",
    "decompose_idea",
    "create_test_idea",
    # NLI Detector
    "NLINoveltyDetector",
    "NLIRelation",
    "NLIResult",
    "NoveltyLevel",
    "ClaimNoveltyResult",
    "NoveltyDetectionResult",
    "create_test_detector",
    # Salience Weighter
    "SalienceWeighter",
    "SalienceLevel",
    "SalienceScore",
    "SalienceResult",
    "weight_claims",
    # Atomic Novelty Scorer
    "AtomicNoveltyScorer",
    "AtomicNoveltyResult",
    "NoveltyBreakdown",
    "NoveltyTier",
    "score_idea_novelty",
    "create_test_scorer"
]
