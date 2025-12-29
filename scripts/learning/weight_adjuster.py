"""
Weight Adjuster for Universal Ideation v3.2

Implements adaptive dimension weight adjustment with stability bounds:
- +/-5% maximum adjustment per dimension per session
- Prevents oscillation through momentum and dampening
- Tracks weight history for analysis
- Supports weight reset and manual overrides

Based on ReflectEvo: Self-improving systems with bounded adaptation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import json


class AdjustmentReason(Enum):
    """Reason for weight adjustment."""
    SUCCESS_CORRELATION = "success_correlation"
    FAILURE_CORRELATION = "failure_correlation"
    DIMENSION_INSIGHT = "dimension_insight"
    MANUAL_OVERRIDE = "manual_override"
    RESET = "reset"
    DECAY = "decay"  # Gradual return to baseline


@dataclass
class WeightAdjustment:
    """Record of a single weight adjustment."""
    dimension: str
    old_weight: float
    new_weight: float
    delta: float
    reason: AdjustmentReason
    reflection_id: Optional[str]
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "dimension": self.dimension,
            "old_weight": self.old_weight,
            "new_weight": self.new_weight,
            "delta": self.delta,
            "reason": self.reason.value,
            "reflection_id": self.reflection_id,
            "timestamp": self.timestamp
        }


@dataclass
class WeightState:
    """Current state of dimension weights."""
    weights: Dict[str, float]
    session_deltas: Dict[str, float]  # Cumulative changes this session
    total_adjustments: int
    last_adjustment: Optional[str]

    def get_remaining_budget(self, dimension: str, max_delta: float = 0.05) -> Tuple[float, float]:
        """Get remaining adjustment budget (negative, positive)."""
        current_delta = self.session_deltas.get(dimension, 0.0)
        return (-max_delta - current_delta, max_delta - current_delta)


class WeightAdjuster:
    """
    Manages adaptive dimension weight adjustments.

    Key constraints:
    - Maximum +/-5% adjustment per dimension per session
    - Weights must sum to 1.0
    - Momentum prevents rapid oscillation
    - Supports gradual decay back to baseline
    """

    # Default v3 weights
    DEFAULT_WEIGHTS = {
        'novelty': 0.12,
        'feasibility': 0.18,
        'market': 0.18,
        'complexity': 0.12,
        'scenario': 0.12,
        'contrarian': 0.10,
        'surprise': 0.10,
        'cross_domain': 0.08
    }

    # Bounds
    MAX_SESSION_DELTA = 0.05  # +/-5% per session
    MAX_TOTAL_DELTA = 0.10   # +/-10% from default ever
    MIN_WEIGHT = 0.02        # No dimension below 2%
    MAX_WEIGHT = 0.30        # No dimension above 30%

    def __init__(
        self,
        initial_weights: Optional[Dict[str, float]] = None,
        max_session_delta: float = 0.05,
        momentum: float = 0.3
    ):
        """
        Initialize weight adjuster.

        Args:
            initial_weights: Starting weights (defaults to v3 weights)
            max_session_delta: Maximum change per dimension per session
            momentum: Weight for previous adjustment direction (0-1)
        """
        self.default_weights = dict(self.DEFAULT_WEIGHTS)
        self.current_weights = dict(initial_weights or self.DEFAULT_WEIGHTS)
        self.max_session_delta = max_session_delta
        self.momentum = momentum

        # Session tracking
        self.session_deltas: Dict[str, float] = {dim: 0.0 for dim in self.current_weights}
        self.previous_deltas: Dict[str, float] = {dim: 0.0 for dim in self.current_weights}

        # History
        self.adjustment_history: List[WeightAdjustment] = []
        self.session_count = 0

        # Validate initial weights
        self._validate_weights()

    def _validate_weights(self) -> None:
        """Ensure weights are valid and sum to 1.0."""
        total = sum(self.current_weights.values())
        if abs(total - 1.0) > 0.001:
            # Normalize
            for dim in self.current_weights:
                self.current_weights[dim] /= total

    def apply_reflection(
        self,
        dimension: str,
        suggested_delta: float,
        reason: AdjustmentReason,
        reflection_id: Optional[str] = None
    ) -> Optional[WeightAdjustment]:
        """
        Apply a weight adjustment from a reflection.

        Args:
            dimension: Dimension to adjust
            suggested_delta: Suggested change (positive or negative)
            reason: Reason for adjustment
            reflection_id: ID of source reflection

        Returns:
            WeightAdjustment if applied, None if blocked
        """
        if dimension not in self.current_weights:
            return None

        # Check session budget
        current_session_delta = self.session_deltas.get(dimension, 0.0)
        remaining_budget = self.max_session_delta - abs(current_session_delta)

        if remaining_budget <= 0:
            return None  # Session limit reached for this dimension

        # Apply momentum (blend with previous direction)
        prev_delta = self.previous_deltas.get(dimension, 0.0)
        adjusted_delta = (
            suggested_delta * (1 - self.momentum) +
            prev_delta * self.momentum
        )

        # Clamp to remaining budget
        if adjusted_delta > 0:
            adjusted_delta = min(adjusted_delta, remaining_budget)
        else:
            adjusted_delta = max(adjusted_delta, -remaining_budget)

        # Check total deviation from default
        current_deviation = self.current_weights[dimension] - self.default_weights[dimension]
        if abs(current_deviation + adjusted_delta) > self.MAX_TOTAL_DELTA:
            # Would exceed total allowed deviation
            if adjusted_delta > 0:
                adjusted_delta = max(0, self.MAX_TOTAL_DELTA - current_deviation)
            else:
                adjusted_delta = min(0, -self.MAX_TOTAL_DELTA - current_deviation)

        if abs(adjusted_delta) < 0.001:
            return None  # Too small to apply

        # Calculate new weight
        old_weight = self.current_weights[dimension]
        new_weight = old_weight + adjusted_delta

        # Enforce bounds
        new_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))
        actual_delta = new_weight - old_weight

        if abs(actual_delta) < 0.001:
            return None

        # Apply adjustment
        self.current_weights[dimension] = new_weight
        self.session_deltas[dimension] = current_session_delta + actual_delta
        self.previous_deltas[dimension] = actual_delta

        # Normalize weights to sum to 1.0
        self._rebalance_weights(dimension)

        # Record adjustment
        adjustment = WeightAdjustment(
            dimension=dimension,
            old_weight=old_weight,
            new_weight=self.current_weights[dimension],
            delta=actual_delta,
            reason=reason,
            reflection_id=reflection_id,
            timestamp=datetime.now().isoformat()
        )
        self.adjustment_history.append(adjustment)

        return adjustment

    def _rebalance_weights(self, changed_dimension: str) -> None:
        """
        Rebalance weights after adjustment to sum to 1.0.

        Distributes the difference proportionally across other dimensions.
        """
        total = sum(self.current_weights.values())
        if abs(total - 1.0) < 0.001:
            return

        # Calculate how much to redistribute
        excess = total - 1.0

        # Get other dimensions (excluding changed one)
        other_dims = [d for d in self.current_weights if d != changed_dimension]

        if not other_dims:
            return

        # Calculate total weight of other dimensions
        other_total = sum(self.current_weights[d] for d in other_dims)

        if other_total <= 0:
            return

        # Redistribute proportionally
        for dim in other_dims:
            proportion = self.current_weights[dim] / other_total
            adjustment = -excess * proportion
            new_weight = self.current_weights[dim] + adjustment

            # Enforce bounds
            new_weight = max(self.MIN_WEIGHT, min(self.MAX_WEIGHT, new_weight))
            self.current_weights[dim] = new_weight

        # Final normalization pass
        total = sum(self.current_weights.values())
        for dim in self.current_weights:
            self.current_weights[dim] /= total

    def apply_batch_reflections(
        self,
        dimension_impacts: Dict[str, float],
        reason: AdjustmentReason,
        reflection_ids: Optional[List[str]] = None
    ) -> List[WeightAdjustment]:
        """
        Apply multiple dimension adjustments from a reflection batch.

        Args:
            dimension_impacts: Dict of dimension -> suggested delta
            reason: Reason for adjustments
            reflection_ids: IDs of source reflections

        Returns:
            List of applied adjustments
        """
        adjustments = []
        reflection_id = reflection_ids[0] if reflection_ids else None

        for dimension, delta in dimension_impacts.items():
            adjustment = self.apply_reflection(
                dimension=dimension,
                suggested_delta=delta,
                reason=reason,
                reflection_id=reflection_id
            )
            if adjustment:
                adjustments.append(adjustment)

        return adjustments

    def reset_session(self) -> None:
        """Reset session deltas for new session."""
        self.session_deltas = {dim: 0.0 for dim in self.current_weights}
        self.session_count += 1

    def reset_to_default(self) -> List[WeightAdjustment]:
        """Reset all weights to default values."""
        adjustments = []

        for dim in self.current_weights:
            if abs(self.current_weights[dim] - self.default_weights[dim]) > 0.001:
                old_weight = self.current_weights[dim]
                self.current_weights[dim] = self.default_weights[dim]

                adjustment = WeightAdjustment(
                    dimension=dim,
                    old_weight=old_weight,
                    new_weight=self.default_weights[dim],
                    delta=self.default_weights[dim] - old_weight,
                    reason=AdjustmentReason.RESET,
                    reflection_id=None,
                    timestamp=datetime.now().isoformat()
                )
                adjustments.append(adjustment)

        self.session_deltas = {dim: 0.0 for dim in self.current_weights}
        self.previous_deltas = {dim: 0.0 for dim in self.current_weights}

        return adjustments

    def apply_decay(self, decay_rate: float = 0.1) -> List[WeightAdjustment]:
        """
        Apply gradual decay toward default weights.

        Args:
            decay_rate: Fraction to move toward default (0-1)

        Returns:
            List of applied adjustments
        """
        adjustments = []

        for dim in self.current_weights:
            current = self.current_weights[dim]
            default = self.default_weights[dim]
            diff = default - current

            if abs(diff) > 0.001:
                decay_amount = diff * decay_rate
                old_weight = current
                new_weight = current + decay_amount

                self.current_weights[dim] = new_weight

                adjustment = WeightAdjustment(
                    dimension=dim,
                    old_weight=old_weight,
                    new_weight=new_weight,
                    delta=decay_amount,
                    reason=AdjustmentReason.DECAY,
                    reflection_id=None,
                    timestamp=datetime.now().isoformat()
                )
                adjustments.append(adjustment)

        return adjustments

    def get_weight_state(self) -> WeightState:
        """Get current weight state."""
        return WeightState(
            weights=dict(self.current_weights),
            session_deltas=dict(self.session_deltas),
            total_adjustments=len(self.adjustment_history),
            last_adjustment=self.adjustment_history[-1].timestamp if self.adjustment_history else None
        )

    def get_deviation_report(self) -> Dict[str, Dict]:
        """Get report of deviations from default weights."""
        report = {}

        for dim in self.current_weights:
            current = self.current_weights[dim]
            default = self.default_weights[dim]
            deviation = current - default
            deviation_pct = (deviation / default) * 100 if default > 0 else 0

            report[dim] = {
                "current": current,
                "default": default,
                "deviation": deviation,
                "deviation_pct": deviation_pct,
                "session_delta": self.session_deltas.get(dim, 0),
                "at_limit": abs(self.session_deltas.get(dim, 0)) >= self.max_session_delta
            }

        return report

    def get_statistics(self) -> Dict:
        """Get adjustment statistics."""
        # Count adjustments by reason
        reason_counts = {}
        for adj in self.adjustment_history:
            reason = adj.reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Count adjustments by dimension
        dim_counts = {}
        for adj in self.adjustment_history:
            dim = adj.dimension
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        return {
            "total_adjustments": len(self.adjustment_history),
            "session_count": self.session_count,
            "by_reason": reason_counts,
            "by_dimension": dim_counts,
            "current_weights": self.current_weights,
            "deviation_from_default": self.get_deviation_report()
        }

    def export_history(self) -> List[Dict]:
        """Export adjustment history."""
        return [adj.to_dict() for adj in self.adjustment_history]

    def import_weights(self, weights: Dict[str, float]) -> None:
        """Import weights from external source."""
        for dim, weight in weights.items():
            if dim in self.current_weights:
                self.current_weights[dim] = weight

        self._validate_weights()


class ModeWeightAdjuster:
    """
    Adjusts generator mode weights based on performance.

    Separate from dimension weights - controls exploration vs exploitation balance.
    """

    DEFAULT_MODE_WEIGHTS = {
        'explorer': 0.40,
        'refiner': 0.40,
        'contrarian': 0.20
    }

    MAX_MODE_DELTA = 0.10  # +/-10% per session

    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        """Initialize mode weight adjuster."""
        self.current_weights = dict(initial_weights or self.DEFAULT_MODE_WEIGHTS)
        self.default_weights = dict(self.DEFAULT_MODE_WEIGHTS)
        self.session_deltas: Dict[str, float] = {m: 0.0 for m in self.current_weights}
        self.adjustment_history: List[Dict] = []

    def adjust_for_performance(
        self,
        mode_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adjust mode weights based on average performance.

        Args:
            mode_scores: Average score for each mode

        Returns:
            New mode weights
        """
        if not mode_scores:
            return self.current_weights

        # Calculate performance-based adjustments
        avg_score = sum(mode_scores.values()) / len(mode_scores)

        for mode, score in mode_scores.items():
            if mode not in self.current_weights:
                continue

            # Adjust based on relative performance
            relative_perf = (score - avg_score) / 100  # Normalize to +/- small value
            suggested_delta = relative_perf * 0.05  # Scale to reasonable adjustment

            # Check session budget
            current_delta = self.session_deltas.get(mode, 0)
            remaining = self.MAX_MODE_DELTA - abs(current_delta)

            if remaining > 0:
                actual_delta = max(-remaining, min(remaining, suggested_delta))
                self.current_weights[mode] += actual_delta
                self.session_deltas[mode] = current_delta + actual_delta

        # Normalize to sum to 1.0
        total = sum(self.current_weights.values())
        for mode in self.current_weights:
            self.current_weights[mode] /= total

        return self.current_weights

    def get_weights(self) -> Dict[str, float]:
        """Get current mode weights."""
        return dict(self.current_weights)

    def reset_session(self) -> None:
        """Reset session deltas."""
        self.session_deltas = {m: 0.0 for m in self.current_weights}
