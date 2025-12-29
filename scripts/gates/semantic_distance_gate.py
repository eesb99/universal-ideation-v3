"""
Semantic Distance Gate for Universal Ideation v3

Enforces minimum conceptual distance to prevent idea clustering.
Ideas too similar to existing centroid are rejected, forcing exploration.

Based on contrastive generation and semantic steering literature.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum


class GateDecision(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    BORDERLINE = "borderline"


@dataclass
class GateResult:
    """Result of semantic distance check."""
    decision: GateDecision
    distance: float
    threshold: float
    centroid_similarity: float
    nearest_idea_similarity: float
    reason: str


@dataclass
class IdeaEmbedding:
    """Container for idea with its embedding."""
    idea_id: str
    title: str
    embedding: np.ndarray
    score: float


class SemanticDistanceGate:
    """
    Reject ideas too similar to existing centroid.

    Key innovation: GATE enforcement, not just penalty.
    v2 penalized similarity but still accepted ideas.
    v3 REJECTS ideas that don't meet distance threshold.
    """

    def __init__(
        self,
        min_distance: float = 0.4,
        rejection_similarity: float = 0.6,
        adaptive_relaxation: bool = True
    ):
        """
        Initialize the semantic distance gate.

        Args:
            min_distance: Minimum cosine distance from centroid (default 0.4)
            rejection_similarity: Reject if similarity > this value (default 0.6)
            adaptive_relaxation: Whether to relax threshold over time
        """
        self.min_distance = min_distance
        self.rejection_similarity = rejection_similarity
        self.adaptive_relaxation = adaptive_relaxation

        # State tracking
        self.embeddings: List[IdeaEmbedding] = []
        self.centroid: Optional[np.ndarray] = None
        self.current_threshold = min_distance

        # Statistics
        self.total_checked = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.borderline_count = 0

    def update_centroid(self) -> None:
        """Recalculate centroid from all accepted ideas."""
        if not self.embeddings:
            self.centroid = None
            return

        # Stack all embeddings and compute mean
        all_embeddings = np.array([e.embedding for e in self.embeddings])
        self.centroid = np.mean(all_embeddings, axis=0)

        # Normalize centroid for consistent comparison
        norm = np.linalg.norm(self.centroid)
        if norm > 0:
            self.centroid = self.centroid / norm

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def cosine_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine distance (1 - similarity)."""
        return 1.0 - self.cosine_similarity(vec1, vec2)

    def adaptive_threshold(self, iteration: int, max_iterations: int = 30) -> float:
        """
        Calculate adaptive threshold that relaxes over time.

        Rationale: As session progresses, finding distant ideas becomes harder.
        Gradually relaxing threshold maintains idea flow while still enforcing novelty.

        Args:
            iteration: Current iteration number
            max_iterations: Maximum iterations expected

        Returns:
            Adjusted threshold value
        """
        if not self.adaptive_relaxation:
            return self.min_distance

        # Progress through session (0.0 to 1.0)
        progress = min(iteration / max_iterations, 1.0)

        # Relaxation curve: Start strict (0.4), gradually relax to 0.25
        # Using smooth quadratic curve for gradual transition
        relaxation = 0.15 * (progress ** 1.5)  # Slower initial, faster later

        self.current_threshold = max(0.25, self.min_distance - relaxation)
        return self.current_threshold

    def check_distance(
        self,
        new_embedding: np.ndarray,
        iteration: int = 0,
        max_iterations: int = 30
    ) -> GateResult:
        """
        Check if new idea meets distance requirements.

        Args:
            new_embedding: Embedding vector of new idea
            iteration: Current iteration number
            max_iterations: Maximum iterations for threshold calculation

        Returns:
            GateResult with decision and metrics
        """
        self.total_checked += 1

        # First idea always passes
        if self.centroid is None:
            self.accepted_count += 1
            return GateResult(
                decision=GateDecision.ACCEPT,
                distance=1.0,
                threshold=self.min_distance,
                centroid_similarity=0.0,
                nearest_idea_similarity=0.0,
                reason="First idea - automatically accepted"
            )

        # Calculate adaptive threshold
        threshold = self.adaptive_threshold(iteration, max_iterations)

        # Calculate distance from centroid
        centroid_distance = self.cosine_distance(new_embedding, self.centroid)
        centroid_similarity = 1.0 - centroid_distance

        # Find nearest existing idea
        nearest_similarity = 0.0
        if self.embeddings:
            similarities = [
                self.cosine_similarity(new_embedding, e.embedding)
                for e in self.embeddings
            ]
            nearest_similarity = max(similarities)

        # Decision logic
        if centroid_similarity > self.rejection_similarity:
            # Too similar to centroid - reject
            self.rejected_count += 1
            return GateResult(
                decision=GateDecision.REJECT,
                distance=centroid_distance,
                threshold=threshold,
                centroid_similarity=centroid_similarity,
                nearest_idea_similarity=nearest_similarity,
                reason=f"Too similar to centroid ({centroid_similarity:.2f} > {self.rejection_similarity})"
            )
        elif centroid_distance < threshold:
            # Below distance threshold - reject
            self.rejected_count += 1
            return GateResult(
                decision=GateDecision.REJECT,
                distance=centroid_distance,
                threshold=threshold,
                centroid_similarity=centroid_similarity,
                nearest_idea_similarity=nearest_similarity,
                reason=f"Distance {centroid_distance:.2f} below threshold {threshold:.2f}"
            )
        elif nearest_similarity > 0.85:
            # Too similar to a specific existing idea
            self.borderline_count += 1
            return GateResult(
                decision=GateDecision.BORDERLINE,
                distance=centroid_distance,
                threshold=threshold,
                centroid_similarity=centroid_similarity,
                nearest_idea_similarity=nearest_similarity,
                reason=f"Very similar to existing idea ({nearest_similarity:.2f})"
            )
        else:
            # Passes all checks
            self.accepted_count += 1
            return GateResult(
                decision=GateDecision.ACCEPT,
                distance=centroid_distance,
                threshold=threshold,
                centroid_similarity=centroid_similarity,
                nearest_idea_similarity=nearest_similarity,
                reason=f"Distance {centroid_distance:.2f} meets threshold {threshold:.2f}"
            )

    def add_idea(self, idea: IdeaEmbedding) -> None:
        """
        Add accepted idea to the collection and update centroid.

        Args:
            idea: IdeaEmbedding to add
        """
        self.embeddings.append(idea)
        self.update_centroid()

    def get_diversity_score(self) -> float:
        """
        Calculate overall diversity of accepted ideas.

        Returns:
            Score from 0 (all identical) to 1 (maximally diverse)
        """
        if len(self.embeddings) < 2:
            return 1.0  # Can't calculate diversity with < 2 ideas

        # Calculate pairwise distances
        n = len(self.embeddings)
        total_distance = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                distance = self.cosine_distance(
                    self.embeddings[i].embedding,
                    self.embeddings[j].embedding
                )
                total_distance += distance
                count += 1

        # Average pairwise distance
        avg_distance = total_distance / count if count > 0 else 0.0

        # Normalize to 0-1 range (cosine distance is already 0-2, but typically 0-1)
        return min(avg_distance, 1.0)

    def get_cluster_count(self, threshold: float = 0.3) -> int:
        """
        Estimate number of semantic clusters in accepted ideas.

        Uses simple agglomerative approach without external dependencies.

        Args:
            threshold: Distance threshold for clustering

        Returns:
            Estimated number of clusters
        """
        if len(self.embeddings) < 2:
            return len(self.embeddings)

        # Simple single-linkage clustering
        n = len(self.embeddings)
        cluster_ids = list(range(n))  # Each idea starts in its own cluster

        for i in range(n):
            for j in range(i + 1, n):
                distance = self.cosine_distance(
                    self.embeddings[i].embedding,
                    self.embeddings[j].embedding
                )
                if distance < threshold:
                    # Merge clusters
                    old_cluster = cluster_ids[j]
                    new_cluster = cluster_ids[i]
                    cluster_ids = [
                        new_cluster if c == old_cluster else c
                        for c in cluster_ids
                    ]

        return len(set(cluster_ids))

    def get_statistics(self) -> dict:
        """Get gate statistics."""
        return {
            "total_checked": self.total_checked,
            "accepted": self.accepted_count,
            "rejected": self.rejected_count,
            "borderline": self.borderline_count,
            "acceptance_rate": self.accepted_count / max(self.total_checked, 1),
            "current_threshold": self.current_threshold,
            "diversity_score": self.get_diversity_score(),
            "estimated_clusters": self.get_cluster_count(),
            "ideas_stored": len(self.embeddings)
        }

    def get_rejection_feedback(self, result: GateResult) -> str:
        """
        Generate feedback for rejected idea to guide regeneration.

        Args:
            result: GateResult from check_distance

        Returns:
            Feedback string for idea generator
        """
        if result.decision == GateDecision.ACCEPT:
            return ""

        feedback_parts = []

        if result.centroid_similarity > 0.5:
            feedback_parts.append(
                f"Idea is {result.centroid_similarity:.0%} similar to the semantic center. "
                "Try a completely different approach, market segment, or technology."
            )

        if result.nearest_idea_similarity > 0.7:
            feedback_parts.append(
                f"Idea is {result.nearest_idea_similarity:.0%} similar to an existing idea. "
                "Explore a different category or business model."
            )

        if result.distance < 0.3:
            feedback_parts.append(
                "Idea clusters with existing concepts. "
                "Consider: different target market, pricing model, distribution channel, or technology stack."
            )

        return " ".join(feedback_parts) if feedback_parts else result.reason


def create_mock_embedding(dimension: int = 384) -> np.ndarray:
    """Create a random mock embedding for testing."""
    vec = np.random.randn(dimension)
    return vec / np.linalg.norm(vec)


# Convenience function for quick distance check
def check_semantic_distance(
    new_embedding: np.ndarray,
    existing_embeddings: List[np.ndarray],
    min_distance: float = 0.4
) -> Tuple[bool, float]:
    """
    Quick helper to check if embedding is distant enough.

    Args:
        new_embedding: New idea embedding
        existing_embeddings: List of existing embeddings
        min_distance: Minimum required distance

    Returns:
        Tuple of (passes_check, distance_from_centroid)
    """
    if not existing_embeddings:
        return (True, 1.0)

    # Calculate centroid
    centroid = np.mean(existing_embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    # Calculate distance
    new_norm = new_embedding / np.linalg.norm(new_embedding)
    similarity = float(np.dot(new_norm, centroid))
    distance = 1.0 - similarity

    return (distance >= min_distance, distance)
