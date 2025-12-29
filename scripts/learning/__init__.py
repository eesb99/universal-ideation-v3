"""
Learning module for Universal Ideation v3.2

Provides DARLING reward calculation and ReflectEvo self-improvement:
- DARLINGReward: Quality + Diversity + Exploration rewards (v3.0)
- ReflectionGenerator: Batch analysis and pattern extraction (v3.2)
- WeightAdjuster: Adaptive dimension weight adjustment (v3.2)
- LearningPersistence: Cross-session SQLite storage (v3.2)
"""

from .darling_reward import (
    DARLINGReward,
    DimensionScores,
    RewardBreakdown,
    SemanticRegion,
    GeneratorMode,
    calculate_darling_reward,
    create_sample_scores
)

from .reflection_generator import (
    ReflectionGenerator,
    Reflection,
    ReflectionBatch,
    ReflectionType,
    ReflectionConfidence,
    IdeaForReflection,
    create_test_ideas
)

from .weight_adjuster import (
    WeightAdjuster,
    ModeWeightAdjuster,
    WeightAdjustment,
    WeightState,
    AdjustmentReason
)

from .learning_persistence import (
    LearningPersistence,
    SessionRecord
)

__all__ = [
    # DARLING Reward (v3.0)
    "DARLINGReward",
    "DimensionScores",
    "RewardBreakdown",
    "SemanticRegion",
    "GeneratorMode",
    "calculate_darling_reward",
    "create_sample_scores",
    # Reflection Generator (v3.2)
    "ReflectionGenerator",
    "Reflection",
    "ReflectionBatch",
    "ReflectionType",
    "ReflectionConfidence",
    "IdeaForReflection",
    "create_test_ideas",
    # Weight Adjuster (v3.2)
    "WeightAdjuster",
    "ModeWeightAdjuster",
    "WeightAdjustment",
    "WeightState",
    "AdjustmentReason",
    # Learning Persistence (v3.2)
    "LearningPersistence",
    "SessionRecord"
]
