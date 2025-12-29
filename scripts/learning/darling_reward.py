"""
DARLING Reward Calculation for Universal Ideation v3

Diversity-Aware Reinforcement Learning reward function.
Balances quality, diversity, and exploration to prevent mode collapse.

Based on DARLING 2024 paper: Quality + Diversity + Exploration rewards.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum


class GeneratorMode(Enum):
    EXPLORER = "explorer"
    REFINER = "refiner"
    CONTRARIAN = "contrarian"


@dataclass
class DimensionScores:
    """8-dimension scoring from v3 architecture."""
    novelty: float = 0.0        # 12% weight
    feasibility: float = 0.0    # 18% weight
    market: float = 0.0         # 18% weight
    complexity: float = 0.0     # 12% weight - network effects
    scenario: float = 0.0       # 12% weight - future robustness
    contrarian: float = 0.0     # 10% weight - assumption challenging
    surprise: float = 0.0       # 10% weight - schema violation (NEW)
    cross_domain: float = 0.0   # 8% weight - analogical distance (NEW)


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward calculation."""
    quality_score: float
    diversity_bonus: float
    exploration_bonus: float
    final_reward: float
    dimension_contributions: Dict[str, float]
    region_visited: str
    is_new_region: bool
    generator_mode: str


class SemanticRegion:
    """Tracks explored semantic regions for exploration bonus."""

    def __init__(self, n_regions: int = 16):
        """
        Initialize region tracker.

        Args:
            n_regions: Number of regions to discretize embedding space into
        """
        self.n_regions = n_regions
        self.visited_regions: Set[str] = set()
        self.region_visit_counts: Dict[str, int] = {}

    def get_region_id(self, embedding: np.ndarray) -> str:
        """
        Map embedding to a semantic region ID.

        Uses simple binning approach on first few principal components.
        For production: use k-means clustering or learned regions.

        Args:
            embedding: Normalized embedding vector

        Returns:
            Region identifier string
        """
        # Use first 4 dimensions as proxy for semantic structure
        # Each dimension is binned into 2 buckets (high/low)
        # This gives 2^4 = 16 regions

        bins = []
        for i in range(min(4, len(embedding))):
            # Bin based on sign (assumes normalized embeddings)
            bins.append('H' if embedding[i] > 0 else 'L')

        return ''.join(bins)

    def visit_region(self, embedding: np.ndarray) -> Tuple[str, bool]:
        """
        Record visit to a region.

        Args:
            embedding: Normalized embedding vector

        Returns:
            Tuple of (region_id, is_first_visit)
        """
        region_id = self.get_region_id(embedding)
        is_new = region_id not in self.visited_regions

        self.visited_regions.add(region_id)
        self.region_visit_counts[region_id] = self.region_visit_counts.get(region_id, 0) + 1

        return region_id, is_new

    def get_region_novelty(self, embedding: np.ndarray) -> float:
        """
        Calculate how novel visiting this region would be.

        Args:
            embedding: Normalized embedding vector

        Returns:
            Novelty score 0-1 (1 = never visited, 0 = heavily visited)
        """
        region_id = self.get_region_id(embedding)

        if region_id not in self.region_visit_counts:
            return 1.0

        visit_count = self.region_visit_counts[region_id]
        # Decay novelty with visits: 1/(1 + visits)
        return 1.0 / (1.0 + visit_count)

    def get_exploration_coverage(self) -> float:
        """
        Calculate what fraction of regions have been explored.

        Returns:
            Coverage ratio 0-1
        """
        return len(self.visited_regions) / self.n_regions

    def get_statistics(self) -> Dict:
        """Get exploration statistics."""
        return {
            "total_regions": self.n_regions,
            "visited_regions": len(self.visited_regions),
            "coverage": self.get_exploration_coverage(),
            "visit_distribution": dict(self.region_visit_counts)
        }


class DARLINGReward:
    """
    Diversity-Aware Reinforcement Learning reward function.

    Key insight: Standard quality-only optimization leads to mode collapse.
    DARLING adds diversity and exploration bonuses to maintain search breadth.
    """

    # v3 Dimension weights (total = 100%)
    DIMENSION_WEIGHTS = {
        'novelty': 0.12,
        'feasibility': 0.18,
        'market': 0.18,
        'complexity': 0.12,
        'scenario': 0.12,
        'contrarian': 0.10,
        'surprise': 0.10,
        'cross_domain': 0.08
    }

    # Reward component weights
    QUALITY_WEIGHT = 0.50
    DIVERSITY_WEIGHT = 0.30
    EXPLORATION_WEIGHT = 0.20

    def __init__(
        self,
        quality_weight: float = 0.50,
        diversity_weight: float = 0.30,
        exploration_weight: float = 0.20,
        n_regions: int = 16
    ):
        """
        Initialize DARLING reward calculator.

        Args:
            quality_weight: Weight for quality score (default 0.50)
            diversity_weight: Weight for diversity bonus (default 0.30)
            exploration_weight: Weight for exploration bonus (default 0.20)
            n_regions: Number of semantic regions to track
        """
        self.quality_weight = quality_weight
        self.diversity_weight = diversity_weight
        self.exploration_weight = exploration_weight

        # Validate weights sum to 1
        total = quality_weight + diversity_weight + exploration_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        # Initialize region tracker
        self.region_tracker = SemanticRegion(n_regions)

        # Statistics
        self.reward_history: List[float] = []
        self.quality_history: List[float] = []
        self.diversity_history: List[float] = []
        self.exploration_history: List[float] = []

    def calculate_quality_score(self, scores: DimensionScores) -> Tuple[float, Dict[str, float]]:
        """
        Calculate weighted quality score from 8 dimensions.

        Args:
            scores: DimensionScores dataclass

        Returns:
            Tuple of (weighted_score, dimension_contributions)
        """
        contributions = {}
        total = 0.0

        dimension_values = {
            'novelty': scores.novelty,
            'feasibility': scores.feasibility,
            'market': scores.market,
            'complexity': scores.complexity,
            'scenario': scores.scenario,
            'contrarian': scores.contrarian,
            'surprise': scores.surprise,
            'cross_domain': scores.cross_domain
        }

        for dimension, value in dimension_values.items():
            weight = self.DIMENSION_WEIGHTS[dimension]
            contribution = value * weight
            contributions[dimension] = contribution
            total += contribution

        return total, contributions

    def calculate_diversity_bonus(
        self,
        centroid_distance: float,
        nearest_idea_distance: float = 1.0
    ) -> float:
        """
        Calculate diversity bonus based on distance from existing ideas.

        Args:
            centroid_distance: Cosine distance from centroid (0-1)
            nearest_idea_distance: Distance from nearest existing idea

        Returns:
            Diversity bonus (0-20 points scale)
        """
        # Base bonus from centroid distance
        # Scale: 0 distance = 0 bonus, 1.0 distance = 15 bonus
        centroid_bonus = centroid_distance * 15

        # Additional bonus for being far from any single idea
        nearest_bonus = nearest_idea_distance * 5

        return min(20, centroid_bonus + nearest_bonus)

    def calculate_exploration_bonus(
        self,
        embedding: np.ndarray,
        generator_mode: GeneratorMode
    ) -> Tuple[float, str, bool]:
        """
        Calculate exploration bonus for visiting new semantic regions.

        Args:
            embedding: Normalized embedding vector
            generator_mode: Current generator mode

        Returns:
            Tuple of (bonus_points, region_id, is_new_region)
        """
        # Visit region and check novelty
        region_id, is_new = self.region_tracker.visit_region(embedding)
        region_novelty = self.region_tracker.get_region_novelty(embedding)

        # Base bonus for region novelty
        base_bonus = region_novelty * 10

        # First visit to region gets extra bonus
        if is_new:
            base_bonus += 5

        # Generator mode bonus
        mode_bonus = 0
        if generator_mode == GeneratorMode.EXPLORER:
            mode_bonus = 3  # Encourage exploration mode
        elif generator_mode == GeneratorMode.CONTRARIAN:
            mode_bonus = 2  # Encourage challenging assumptions

        return base_bonus + mode_bonus, region_id, is_new

    def calculate_reward(
        self,
        scores: DimensionScores,
        embedding: np.ndarray,
        centroid_distance: float,
        generator_mode: GeneratorMode,
        nearest_idea_distance: float = 1.0
    ) -> RewardBreakdown:
        """
        Calculate complete DARLING reward.

        Final Reward = Quality × w1 + Diversity × w2 + Exploration × w3

        Args:
            scores: 8-dimension evaluation scores
            embedding: Normalized embedding of idea
            centroid_distance: Distance from centroid (from SemanticDistanceGate)
            generator_mode: Current generator mode
            nearest_idea_distance: Distance from nearest existing idea

        Returns:
            RewardBreakdown with full calculation details
        """
        # 1. Quality Score (weighted 8-dimension average)
        quality_score, contributions = self.calculate_quality_score(scores)

        # 2. Diversity Bonus
        diversity_bonus = self.calculate_diversity_bonus(
            centroid_distance,
            nearest_idea_distance
        )

        # 3. Exploration Bonus
        exploration_bonus, region_id, is_new = self.calculate_exploration_bonus(
            embedding,
            generator_mode
        )

        # 4. Final Reward (weighted sum, normalized to 0-100 scale)
        # Quality is already 0-100, bonuses are 0-20 scale
        # Normalize bonuses to 0-100 scale for consistent weighting
        normalized_diversity = diversity_bonus * 5  # 0-20 → 0-100
        normalized_exploration = exploration_bonus * (100/18)  # ~0-18 → 0-100

        final_reward = (
            quality_score * self.quality_weight +
            normalized_diversity * self.diversity_weight +
            normalized_exploration * self.exploration_weight
        )

        # Track history
        self.reward_history.append(final_reward)
        self.quality_history.append(quality_score)
        self.diversity_history.append(diversity_bonus)
        self.exploration_history.append(exploration_bonus)

        return RewardBreakdown(
            quality_score=quality_score,
            diversity_bonus=diversity_bonus,
            exploration_bonus=exploration_bonus,
            final_reward=final_reward,
            dimension_contributions=contributions,
            region_visited=region_id,
            is_new_region=is_new,
            generator_mode=generator_mode.value
        )

    def get_reward_trend(self, window: int = 10) -> Dict[str, float]:
        """
        Analyze reward trends for plateau detection.

        Args:
            window: Number of recent rewards to analyze

        Returns:
            Trend statistics
        """
        if len(self.reward_history) < window * 2:
            return {
                "recent_avg": np.mean(self.reward_history) if self.reward_history else 0,
                "previous_avg": 0,
                "trend": 0,
                "is_plateau": False
            }

        recent = self.reward_history[-window:]
        previous = self.reward_history[-window * 2:-window]

        recent_avg = np.mean(recent)
        previous_avg = np.mean(previous)
        trend = recent_avg - previous_avg

        return {
            "recent_avg": recent_avg,
            "previous_avg": previous_avg,
            "trend": trend,
            "is_plateau": abs(trend) < 0.5,
            "variance": np.var(recent)
        }

    def get_component_balance(self) -> Dict[str, float]:
        """
        Analyze balance between quality, diversity, and exploration.

        Returns:
            Average contribution from each component
        """
        if not self.reward_history:
            return {
                "quality_avg": 0,
                "diversity_avg": 0,
                "exploration_avg": 0
            }

        return {
            "quality_avg": np.mean(self.quality_history),
            "diversity_avg": np.mean(self.diversity_history),
            "exploration_avg": np.mean(self.exploration_history),
            "quality_contribution": np.mean(self.quality_history) * self.quality_weight,
            "diversity_contribution": np.mean(self.diversity_history) * 5 * self.diversity_weight,
            "exploration_contribution": np.mean(self.exploration_history) * (100/18) * self.exploration_weight
        }

    def get_statistics(self) -> Dict:
        """Get comprehensive reward statistics."""
        return {
            "total_rewards_calculated": len(self.reward_history),
            "average_reward": np.mean(self.reward_history) if self.reward_history else 0,
            "max_reward": max(self.reward_history) if self.reward_history else 0,
            "min_reward": min(self.reward_history) if self.reward_history else 0,
            "reward_std": np.std(self.reward_history) if len(self.reward_history) > 1 else 0,
            "component_balance": self.get_component_balance(),
            "exploration_stats": self.region_tracker.get_statistics(),
            "trend": self.get_reward_trend()
        }


def create_sample_scores(
    novelty: float = 70,
    feasibility: float = 75,
    market: float = 72,
    complexity: float = 68,
    scenario: float = 70,
    contrarian: float = 65,
    surprise: float = 60,
    cross_domain: float = 55
) -> DimensionScores:
    """Create sample dimension scores for testing."""
    return DimensionScores(
        novelty=novelty,
        feasibility=feasibility,
        market=market,
        complexity=complexity,
        scenario=scenario,
        contrarian=contrarian,
        surprise=surprise,
        cross_domain=cross_domain
    )


# Convenience function for quick reward calculation
def calculate_darling_reward(
    scores: Dict[str, float],
    centroid_distance: float,
    generator_mode: str = "explorer",
    embedding: Optional[np.ndarray] = None
) -> float:
    """
    Quick helper to calculate DARLING reward.

    Args:
        scores: Dictionary with dimension scores
        centroid_distance: Distance from centroid (0-1)
        generator_mode: One of "explorer", "refiner", "contrarian"
        embedding: Optional embedding for exploration tracking

    Returns:
        Final reward score
    """
    calculator = DARLINGReward()

    dimension_scores = DimensionScores(
        novelty=scores.get('novelty', 70),
        feasibility=scores.get('feasibility', 70),
        market=scores.get('market', 70),
        complexity=scores.get('complexity', 70),
        scenario=scores.get('scenario', 70),
        contrarian=scores.get('contrarian', 70),
        surprise=scores.get('surprise', 60),
        cross_domain=scores.get('cross_domain', 50)
    )

    mode = GeneratorMode(generator_mode)

    if embedding is None:
        embedding = np.random.randn(384)
        embedding = embedding / np.linalg.norm(embedding)

    result = calculator.calculate_reward(
        scores=dimension_scores,
        embedding=embedding,
        centroid_distance=centroid_distance,
        generator_mode=mode
    )

    return result.final_reward
