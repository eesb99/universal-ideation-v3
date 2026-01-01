"""
Universal Ideation v3.3 - Full Orchestrator with LLM Integration

Integrates all v3 components:
- Triple Generator System (Explorer, Refiner, Contrarian)
- Semantic Distance Gate (reject too-similar ideas)
- Verification Gate (validate idea completeness and feasibility) [v3.2]
- Atomic Novelty (NovAScore: ACU + NLI + Salience weighting) [v3.2]
- Reflection Learning (ReflectEvo: adaptive weights + persistence) [v3.2]
- DARLING Reward Calculation (quality + diversity + exploration)
- Cognitive Diversity Evaluators (4 personas + debate)
- 8-Dimension Scoring (including Surprise + Cross-Domain)
- Plateau Escape Protocol (don't stop at local optimum)
- LLM-powered idea generation via Anthropic API [v3.3]

v3.1 Changes:
- TCRTE prompt structure
- Constraint-based creativity
- Dynamic cross-domain frequency

v3.2 Changes:
- Verification Gates (VeriMAP-style fail-fast)
- Reflection Learning (ReflectEvo self-improvement)
- Atomic Novelty (NovAScore 0.94 accuracy)

v3.3 Changes:
- LLM-powered idea generation (--use-llm flag)
- Anthropic API integration for real idea generation
- Falls back to stub generators if API not available

Usage:
    # With LLM (real ideas)
    python run_v3.py "domain" --use-llm --verbose

    # Test mode (stub generators)
    python run_v3.py "domain" --verbose
"""

# Path setup for distribution package
import sys
import os
from pathlib import Path
_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from datetime import datetime
import time
import json
import numpy as np
import hashlib
import re

# Import all v3 components
from generators.triple_generator import (
    TripleGenerator, GeneratorMode, ModeWeights,
    ConstraintTemplate, ConstraintConfig
)
from gates.semantic_distance_gate import (
    SemanticDistanceGate, GateDecision, GateResult
)
from gates.verification_gate import (
    VerificationGate, GateVerificationResult, RetryController, RetryContext
)
from gates.verification_diagnostics import (
    VerificationDiagnostics, SessionHealth
)
from learning.darling_reward import (
    DARLINGReward, RewardBreakdown, DimensionScores
)
from learning.reflection_generator import (
    ReflectionGenerator, IdeaForReflection, ReflectionBatch
)
from learning.weight_adjuster import (
    WeightAdjuster, ModeWeightAdjuster, AdjustmentReason
)
from learning.learning_persistence import (
    LearningPersistence
)
from evaluators.cognitive_diversity import (
    EvaluatorPanel, EvaluatorPersona
)
from evaluators.structured_debate import (
    StructuredDebate, DebateResult
)
from evaluators.surprise_dimension import (
    SurpriseDimension, SurpriseAnalysis
)
from evaluators.cross_domain_bridge import (
    CrossDomainBridge, CrossDomainAnalysis
)
from escape.plateau_escape import (
    PlateauEscapeProtocol, PlateauStatus, EscapeIdea, EscapeResult
)
from novelty.atomic_novelty import (
    AtomicNoveltyScorer, AtomicNoveltyResult, NoveltyTier
)


class SessionPhase(Enum):
    """Current phase of ideation session."""
    INITIALIZING = "initializing"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    LEARNING = "learning"
    ESCAPING = "escaping"
    FINALIZING = "finalizing"
    COMPLETE = "complete"


@dataclass
class IdeaResult:
    """Complete result for a single idea."""
    iteration: int
    idea: Dict
    generator_mode: GeneratorMode
    gate_decision: GateResult

    # Evaluation scores
    dimension_scores: Dict[str, float]
    evaluator_scores: Dict[str, float]
    debate_result: Optional[DebateResult]
    surprise_analysis: Optional[SurpriseAnalysis]
    cross_domain_analysis: Optional[CrossDomainAnalysis]

    # v3.2: Atomic novelty
    atomic_novelty_result: Optional[AtomicNoveltyResult] = None

    # DARLING reward
    reward_result: RewardBreakdown = None
    final_score: float = 0.0

    # Metadata
    accepted: bool = False
    rejection_reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SessionResult:
    """Complete session results."""
    domain: str
    total_iterations: int
    duration_seconds: float

    # Ideas
    accepted_ideas: List[IdeaResult]
    rejected_ideas: List[IdeaResult]

    # Escape attempts
    escape_attempts: int
    escape_successful: bool

    # Statistics
    average_score: float
    best_score: float
    best_idea: Optional[IdeaResult]

    # Learnings
    learnings: List[str]

    # Session metadata
    generator_mode_distribution: Dict[str, int]
    dimension_averages: Dict[str, float]
    plateau_info: Dict


@dataclass
class OrchestratorConfig:
    """Configuration for ideation orchestrator."""
    # Iteration limits
    max_iterations: int = 30
    max_minutes: int = 30

    # Scoring thresholds
    acceptance_threshold: float = 65.0

    # Gate settings
    min_semantic_distance: float = 0.4

    # Escape settings
    plateau_window: int = 10
    plateau_threshold: float = 0.5
    max_escape_attempts: int = 2

    # Checkpoint intervals
    contrarian_checkpoint_interval: int = 5
    first_principles_interval: int = 10
    cross_domain_injection_interval: int = 15

    # v3.1: Dynamic cross-domain settings
    dynamic_cross_domain: bool = True  # Enable dynamic frequency adjustment
    low_novelty_threshold: float = 60.0  # Trigger cross-domain if novelty below this
    novelty_check_window: int = 3  # Check last N ideas for low novelty

    # Debate settings
    debate_disagreement_threshold: float = 20.0

    # v3.1: Constraint configuration
    constraints: Optional[ConstraintConfig] = None

    # v3.2: Verification gate settings
    enable_verification: bool = True
    verification_max_retries: int = 3
    verification_strict_mode: bool = False
    enable_llm_verification: bool = False  # Disabled by default (expensive)

    # v3.2: Reflection Learning settings (ReflectEvo)
    enable_reflection: bool = True
    reflection_batch_size: int = 5  # Analyze after every N accepted ideas
    enable_weight_adaptation: bool = True  # Adapt dimension weights from reflections
    enable_persistence: bool = True  # Store learnings across sessions
    load_prior_learnings: bool = True  # Load relevant learnings at session start

    # v3.2: Atomic Novelty settings (NovAScore)
    enable_atomic_novelty: bool = True
    atomic_novelty_weight: float = 0.3  # Weight in final novelty dimension score
    use_llm_for_novelty: bool = False  # Use LLM for decomposition/NLI (slower but better)
    min_novelty_score: float = 40.0  # Minimum atomic novelty for acceptance

    # 8-Dimension weights
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "novelty": 0.12,
        "feasibility": 0.18,
        "market": 0.18,
        "complexity": 0.12,
        "scenario": 0.12,
        "contrarian": 0.10,
        "surprise": 0.10,
        "cross_domain": 0.08
    })


class IdeationOrchestrator:
    """
    Main orchestrator for Universal Ideation v3.

    Coordinates all components in the DARLING learning loop.
    """

    def __init__(
        self,
        domain: str = "protein beverages",
        config: Optional[OrchestratorConfig] = None
    ):
        self.domain = domain
        self.config = config or OrchestratorConfig()

        # Session state
        self.phase = SessionPhase.INITIALIZING
        self.iteration = 0
        self.start_time: Optional[float] = None

        # Results tracking (must be before _init_components)
        self.accepted_ideas: List[IdeaResult] = []
        self.rejected_ideas: List[IdeaResult] = []
        self.learnings: List[str] = []

        # Initialize components (after learnings initialized)
        self._init_components()
        self.mode_counts: Dict[str, int] = {
            "explorer": 0, "refiner": 0, "contrarian": 0
        }

    def _init_components(self):
        """Initialize all v3 components."""
        # Generator with constraints (v3.1)
        self.generator = TripleGenerator(constraints=self.config.constraints)

        # Semantic Distance Gate
        self.gate = SemanticDistanceGate(
            min_distance=self.config.min_semantic_distance
        )

        # v3.2: Verification Gate
        if self.config.enable_verification:
            self.verification_gate = VerificationGate(
                max_retries=self.config.verification_max_retries,
                enable_llm_verification=self.config.enable_llm_verification,
                strict_mode=self.config.verification_strict_mode
            )
            self.verification_diagnostics = VerificationDiagnostics()
            self.retry_controller = RetryController(self.verification_gate)
        else:
            self.verification_gate = None
            self.verification_diagnostics = None
            self.retry_controller = None

        # DARLING Reward
        self.reward_calculator = DARLINGReward()

        # Evaluators
        self.evaluator_panel = EvaluatorPanel()
        self.debate_system = StructuredDebate()

        # New dimensions
        self.surprise_evaluator = SurpriseDimension(self.domain)
        self.cross_domain_evaluator = CrossDomainBridge(self.domain)

        # Escape protocol
        self.escape_protocol = PlateauEscapeProtocol(
            window_size=self.config.plateau_window,
            threshold=self.config.plateau_threshold,
            max_escape_attempts=self.config.max_escape_attempts
        )

        # v3.2: Reflection Learning components
        if self.config.enable_reflection:
            self.reflection_generator = ReflectionGenerator()
            self.weight_adjuster = WeightAdjuster(
                initial_weights=dict(self.config.dimension_weights)
            )
            self.mode_weight_adjuster = ModeWeightAdjuster()

            if self.config.enable_persistence:
                self.learning_persistence = LearningPersistence()
                # Record session start
                import uuid
                self.session_id = str(uuid.uuid4())[:8]
                self.learning_persistence.start_session(self.session_id, self.domain)

                # Load prior learnings if available
                if self.config.load_prior_learnings:
                    self._load_prior_learnings()
            else:
                self.learning_persistence = None
                self.session_id = None
        else:
            self.reflection_generator = None
            self.weight_adjuster = None
            self.mode_weight_adjuster = None
            self.learning_persistence = None
            self.session_id = None

        # Track ideas for reflection batch
        self.pending_reflection_ideas: List[IdeaForReflection] = []

        # v3.2: Atomic Novelty Scorer
        if self.config.enable_atomic_novelty:
            self.atomic_novelty_scorer = AtomicNoveltyScorer(
                use_llm=self.config.use_llm_for_novelty
            )
        else:
            self.atomic_novelty_scorer = None

    def _load_prior_learnings(self):
        """Load relevant prior learnings for this domain."""
        if not self.learning_persistence:
            return

        # Get recommended weights from prior sessions
        prior_weights = self.learning_persistence.get_recommended_weights(self.domain)
        if prior_weights and self.weight_adjuster:
            self.weight_adjuster.import_weights(prior_weights)
            self.learnings.append(
                f"PRIOR LEARNING: Loaded optimized weights from {self.domain} history"
            )

        # Get high-confidence patterns
        prior_reflections = self.learning_persistence.get_high_confidence_patterns(
            self.domain, pattern_type="success"
        )
        if prior_reflections:
            for ref in prior_reflections[:5]:  # Top 5 patterns
                self.learnings.append(f"PRIOR PATTERN: {ref.pattern}")

        # Log learning trend
        trend = self.learning_persistence.get_learning_trend(self.domain)
        if trend.get("sessions_analyzed", 0) >= 2:
            self.learnings.append(
                f"TREND: {trend['trend']} (improvement: {trend.get('improvement', 0):.1f} points)"
            )

    def _analyze_reflection_batch(self):
        """Analyze accumulated ideas and apply learnings."""
        if not self.reflection_generator or not self.pending_reflection_ideas:
            return

        if len(self.pending_reflection_ideas) < self.config.reflection_batch_size:
            return

        # Generate reflections from batch
        batch = self.reflection_generator.analyze_batch(
            self.pending_reflection_ideas,
            session_id=self.session_id
        )

        # Apply weight adjustments from reflections
        if self.config.enable_weight_adaptation and self.weight_adjuster:
            for reflection in batch.reflections:
                if reflection.dimension_impacts:
                    adjustments = self.weight_adjuster.apply_batch_reflections(
                        reflection.dimension_impacts,
                        AdjustmentReason.DIMENSION_INSIGHT,
                        [reflection.id]
                    )
                    for adj in adjustments:
                        self.learnings.append(
                            f"WEIGHT ADJUSTED: {adj.dimension} {adj.old_weight:.2%} -> {adj.new_weight:.2%} "
                            f"(from reflection: {reflection.pattern[:40]})"
                        )

        # Persist to SQLite
        if self.learning_persistence:
            self.learning_persistence.save_reflection_batch(batch)

        # Add reflection insights to learnings
        for reflection in batch.reflections:
            self.learnings.append(
                f"REFLECTION [{reflection.type.value}]: {reflection.observation}"
            )

        # Clear pending ideas (keep recent ones for overlap)
        overlap = self.config.reflection_batch_size // 2
        self.pending_reflection_ideas = self.pending_reflection_ideas[-overlap:]

    def _finalize_session(self):
        """Finalize session and persist learnings."""
        if not self.learning_persistence:
            return

        # Analyze any remaining ideas
        if self.pending_reflection_ideas and self.reflection_generator:
            batch = self.reflection_generator.analyze_batch(
                self.pending_reflection_ideas,
                session_id=self.session_id
            )
            self.learning_persistence.save_reflection_batch(batch)

        # Record session end
        all_scores = [r.final_score for r in self.accepted_ideas]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        best_score = max(all_scores) if all_scores else 0

        self.learning_persistence.end_session(
            session_id=self.session_id,
            ideas_generated=len(self.accepted_ideas) + len(self.rejected_ideas),
            ideas_accepted=len(self.accepted_ideas),
            average_score=avg_score,
            best_score=best_score,
            final_weights=self.weight_adjuster.current_weights if self.weight_adjuster else {}
        )

        # Aggregate domain patterns for future sessions
        self.learning_persistence.aggregate_domain_patterns(self.domain)

    def run(
        self,
        max_iterations: Optional[int] = None,
        max_minutes: Optional[int] = None,
        idea_generator: Optional[Callable] = None,
        score_evaluator: Optional[Callable] = None,
        on_iteration: Optional[Callable] = None
    ) -> SessionResult:
        """
        Run complete ideation session.

        Args:
            max_iterations: Override config max iterations
            max_minutes: Override config max minutes
            idea_generator: Custom idea generation function
            score_evaluator: Custom scoring function
            on_iteration: Callback after each iteration

        Returns:
            SessionResult with all ideas and statistics
        """
        max_iter = max_iterations or self.config.max_iterations
        max_min = max_minutes or self.config.max_minutes

        self.start_time = time.time()
        self.phase = SessionPhase.GENERATING

        while self._should_continue(max_iter, max_min):
            self.iteration += 1

            # Check for plateau and attempt escape if needed
            if self._check_escape_needed():
                escape_result = self._attempt_escape(idea_generator, score_evaluator)
                if not escape_result.should_continue:
                    break

            # Run single iteration
            result = self._run_iteration(idea_generator, score_evaluator)

            # Track result
            if result.accepted:
                self.accepted_ideas.append(result)

                # v3.2: Add to reflection batch
                if self.reflection_generator:
                    self.pending_reflection_ideas.append(IdeaForReflection(
                        id=f"idea_{self.iteration}",
                        title=result.idea.get("title", "Untitled"),
                        dimension_scores=result.dimension_scores,
                        final_score=result.final_score,
                        generator_mode=result.generator_mode.value,
                        accepted=True,
                        domain=self.domain
                    ))
            else:
                self.rejected_ideas.append(result)

            # Update escape protocol with score
            self.escape_protocol.add_score(result.final_score)

            # v3.2: Periodic reflection analysis
            if self.config.enable_reflection:
                self._analyze_reflection_batch()

            # Callback
            if on_iteration:
                on_iteration(self.iteration, result)

            # Checkpoints
            self._run_checkpoints()

        # v3.2: Finalize session with persistence
        if self.config.enable_reflection:
            self._finalize_session()

        self.phase = SessionPhase.COMPLETE
        return self._compile_results()

    def _should_continue(self, max_iter: int, max_min: int) -> bool:
        """Check if session should continue."""
        if self.iteration >= max_iter:
            return False

        if self.start_time:
            elapsed = (time.time() - self.start_time) / 60
            if elapsed >= max_min:
                return False

        if self.escape_protocol.status == PlateauStatus.CONFIRMED_PLATEAU:
            return False

        return True

    def _run_iteration(
        self,
        idea_generator: Optional[Callable],
        score_evaluator: Optional[Callable]
    ) -> IdeaResult:
        """Run single ideation iteration."""

        # Step 1: Select mode
        recent_scores = [r.final_score for r in self.accepted_ideas[-10:]]
        is_plateau = self.escape_protocol.status == PlateauStatus.PLATEAU_DETECTED

        mode = self.generator.select_mode(
            iteration=self.iteration,
            max_iterations=self.config.max_iterations,
            recent_scores=recent_scores,
            is_plateau=is_plateau
        )
        self.mode_counts[mode.value] += 1

        # Step 2: Generate idea
        self.phase = SessionPhase.GENERATING

        if idea_generator:
            idea = idea_generator(self.domain, mode, self.learnings)
        else:
            idea = self._generate_idea_stub(mode)

        # Step 2.5 (v3.2): Verification Gate
        verification_result = None
        if self.verification_gate:
            verification_result = self.verification_gate.verify(
                idea=idea,
                domain=self.domain,
                prior_ideas=[r.idea for r in self.accepted_ideas]
            )

            # Record verification for diagnostics
            if self.verification_diagnostics:
                self.verification_diagnostics.record_verification(verification_result, idea)

            # If verification failed, reject idea
            if not verification_result.passed:
                return IdeaResult(
                    iteration=self.iteration,
                    idea=idea,
                    generator_mode=mode,
                    gate_decision=GateResult(
                        decision=GateDecision.REJECT,
                        distance=0,
                        threshold=0,
                        centroid_similarity=0,
                        nearest_idea_similarity=0,
                        reason=f"Verification failed: {verification_result.get_summary()}"
                    ),
                    dimension_scores={},
                    evaluator_scores={},
                    debate_result=None,
                    surprise_analysis=None,
                    cross_domain_analysis=None,
                    reward_result=RewardBreakdown(
                        quality_score=0, diversity_bonus=0,
                        exploration_bonus=0, final_reward=0,
                        dimension_contributions={},
                        region_visited="",
                        is_new_region=False,
                        generator_mode=""
                    ),
                    final_score=0,
                    accepted=False,
                    rejection_reason=f"Verification: {verification_result.retry_hints}"
                )

        # Step 3: Semantic distance gate (simplified for testing)
        # In production, would use actual embeddings
        prior_ideas = [r.idea for r in self.accepted_ideas]

        # Stub gate result - always accept in test mode
        gate_result = GateResult(
            decision=GateDecision.ACCEPT,
            distance=0.6,
            threshold=0.4,
            centroid_similarity=0.4,
            nearest_idea_similarity=0.3,
            reason="Accepted"
        )

        if gate_result.decision == GateDecision.REJECT:
            return IdeaResult(
                iteration=self.iteration,
                idea=idea,
                generator_mode=mode,
                gate_decision=gate_result,
                dimension_scores={},
                evaluator_scores={},
                debate_result=None,
                surprise_analysis=None,
                cross_domain_analysis=None,
                reward_result=RewardBreakdown(
                    quality_score=0, diversity_bonus=0,
                    exploration_bonus=0, final_reward=0,
                    dimension_contributions={},
                    region_visited="",
                    is_new_region=False,
                    generator_mode=""
                ),
                final_score=0,
                accepted=False,
                rejection_reason=f"Gate rejected: {gate_result.reason}"
            )

        # Step 4: Evaluate with cognitive diversity panel
        self.phase = SessionPhase.EVALUATING

        if score_evaluator:
            dimension_scores = score_evaluator(idea, self.domain)
        else:
            dimension_scores = self._evaluate_stub(idea)

        # Step 5: Cognitive diversity evaluation
        evaluator_scores = {}
        for persona, evaluator in self.evaluator_panel.evaluators.items():
            score = evaluator.calculate_weighted_score(dimension_scores)
            evaluator_scores[persona.value] = score

        # Step 6: Structured debate if high disagreement
        debate_result = None
        # Check disagreement between evaluators
        if evaluator_scores:
            scores = list(evaluator_scores.values())
            if len(scores) >= 2:
                disagreement = max(scores) - min(scores)
                if disagreement > self.config.debate_disagreement_threshold:
                    # Would trigger debate in full implementation
                    pass

        # Step 7: New dimension scores
        surprise_analysis = self.surprise_evaluator.analyze(idea, prior_ideas)
        cross_domain_analysis = self.cross_domain_evaluator.analyze(idea)

        dimension_scores["surprise"] = surprise_analysis.final_score
        dimension_scores["cross_domain"] = cross_domain_analysis.final_score

        # Step 7.5 (v3.2): Atomic Novelty Assessment
        atomic_novelty_result = None
        if self.atomic_novelty_scorer:
            atomic_novelty_result = self.atomic_novelty_scorer.score_novelty(
                idea=idea,
                semantic_distance=gate_result.distance
            )

            # Blend atomic novelty with existing novelty score
            existing_novelty = dimension_scores.get("novelty", 50)
            atomic_weight = self.config.atomic_novelty_weight
            dimension_scores["novelty"] = (
                existing_novelty * (1 - atomic_weight) +
                atomic_novelty_result.final_score * atomic_weight
            )

            # Add idea to prior corpus for future comparisons
            self.atomic_novelty_scorer.add_prior_idea(idea)

            # Check minimum novelty threshold
            if atomic_novelty_result.final_score < self.config.min_novelty_score:
                self.learnings.append(
                    f"LOW NOVELTY: Idea scored {atomic_novelty_result.final_score:.0f}/100 "
                    f"({atomic_novelty_result.novelty_tier.value})"
                )

        # Step 8: Calculate DARLING reward
        self.phase = SessionPhase.LEARNING

        # Stub reward result for testing
        # In production, would use:
        # reward_result = self.reward_calculator.calculate_reward(
        #     scores=DimensionScores(**dimension_scores),
        #     embedding=idea_embedding,
        #     centroid_distance=gate_result.distance,
        #     generator_mode=mode
        # )
        quality = sum(dimension_scores.values()) / len(dimension_scores) if dimension_scores else 0
        reward_result = RewardBreakdown(
            quality_score=quality,
            diversity_bonus=gate_result.distance * 15,
            exploration_bonus=5.0,
            final_reward=quality * 0.5 + gate_result.distance * 15 * 0.3 + 5.0 * 0.2,
            dimension_contributions={k: v * self.config.dimension_weights.get(k, 0.1)
                                    for k, v in dimension_scores.items()},
            region_visited="region_0",
            is_new_region=False,
            generator_mode=mode.value
        )

        # Step 9: Calculate final score
        final_score = self._calculate_final_score(dimension_scores, reward_result)

        # Step 10: Accept or reject
        accepted = final_score >= self.config.acceptance_threshold

        if accepted:
            # In production, would add idea embedding to gate first
            # self.gate.add_idea(idea_embedding)
            self.gate.update_centroid()

        return IdeaResult(
            iteration=self.iteration,
            idea=idea,
            generator_mode=mode,
            gate_decision=gate_result,
            dimension_scores=dimension_scores,
            evaluator_scores=evaluator_scores,
            debate_result=debate_result,
            surprise_analysis=surprise_analysis,
            cross_domain_analysis=cross_domain_analysis,
            atomic_novelty_result=atomic_novelty_result,
            reward_result=reward_result,
            final_score=final_score,
            accepted=accepted,
            rejection_reason=None if accepted else f"Score {final_score:.1f} < threshold {self.config.acceptance_threshold}"
        )

    def _check_escape_needed(self) -> bool:
        """Check if escape attempt should be triggered."""
        if self.iteration < self.config.plateau_window * 2:
            return False

        analysis = self.escape_protocol.check_plateau()
        return self.escape_protocol.should_attempt_escape()

    def _create_simple_embedding(self, idea: Dict, dimension: int = 128) -> np.ndarray:
        """
        Create a simple text-based embedding for an idea.

        Uses a hash-based approach to convert idea text into a deterministic
        embedding vector. This provides consistent embeddings for distance
        calculations without requiring an external embedding model.

        Args:
            idea: The idea dictionary containing title and description
            dimension: Embedding dimension (default 128)

        Returns:
            Normalized numpy array of shape (dimension,)
        """
        # Extract text content from idea
        text_parts = []
        if isinstance(idea, dict):
            text_parts.append(idea.get("title", ""))
            text_parts.append(idea.get("description", ""))
            text_parts.append(idea.get("mechanism", ""))
        else:
            text_parts.append(str(idea))

        text = " ".join(filter(None, text_parts)).lower()

        # Create embedding using hash-based approach
        embedding = np.zeros(dimension)

        # Tokenize and hash each word
        words = text.split()
        for word in words:
            # Hash the word to get a deterministic index
            word_hash = int(hashlib.md5(word.encode()).hexdigest(), 16)
            # Use multiple hash functions for better distribution
            for i in range(3):
                idx = (word_hash + i * 7919) % dimension  # 7919 is a prime
                sign = 1 if ((word_hash >> i) & 1) else -1
                embedding[idx] += sign

        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def _calculate_deviation_from_centroid(self, idea: Dict) -> float:
        """
        Calculate how far an idea deviates from the current idea centroid.

        Uses the SemanticDistanceGate's centroid if available, otherwise
        returns a high deviation value indicating novelty.

        Args:
            idea: The idea dictionary to measure

        Returns:
            Deviation score from 0.0 (identical to centroid) to 1.0 (maximally different)
        """
        # If no ideas have been accepted yet, return high deviation
        if self.gate.centroid is None or len(self.gate.embeddings) == 0:
            return 1.0

        # Create embedding for the new idea
        idea_embedding = self._create_simple_embedding(idea)

        # Resize if needed to match gate centroid dimension
        if len(idea_embedding) != len(self.gate.centroid):
            # Pad or truncate to match centroid dimension
            target_dim = len(self.gate.centroid)
            if len(idea_embedding) < target_dim:
                idea_embedding = np.pad(idea_embedding, (0, target_dim - len(idea_embedding)))
            else:
                idea_embedding = idea_embedding[:target_dim]
            # Re-normalize
            norm = np.linalg.norm(idea_embedding)
            if norm > 0:
                idea_embedding = idea_embedding / norm

        # Calculate cosine distance from centroid
        deviation = self.gate.cosine_distance(idea_embedding, self.gate.centroid)

        # Clamp to [0, 1] range
        return max(0.0, min(1.0, deviation))

    def _attempt_escape(
        self,
        idea_generator: Optional[Callable],
        score_evaluator: Optional[Callable]
    ) -> EscapeResult:
        """Attempt to escape from plateau."""
        self.phase = SessionPhase.ESCAPING

        # Generate escape prompts
        prior_summary = self._summarize_prior_ideas()
        prompts = self.escape_protocol.get_escape_prompts(
            self.domain, self.learnings, prior_summary
        )

        # Generate and evaluate escape ideas
        escape_ideas = []
        for prompt_config in prompts:
            if idea_generator:
                idea = idea_generator(
                    self.domain,
                    GeneratorMode.EXPLORER,
                    [],  # No learnings for escape
                    escape_prompt=prompt_config["prompt"]
                )
            else:
                idea = self._generate_idea_stub(GeneratorMode.EXPLORER)

            if score_evaluator:
                scores = score_evaluator(idea, self.domain)
            else:
                scores = self._evaluate_stub(idea)

            final_score = self._calculate_final_score(scores, None)

            # Calculate actual deviation from centroid
            deviation = self._calculate_deviation_from_centroid(idea)

            escape_ideas.append(EscapeIdea(
                idea=idea,
                score=final_score,
                strategy=prompt_config["strategy"],
                deviation_from_centroid=deviation
            ))

        # Evaluate escape attempt
        return self.escape_protocol.evaluate_escape(escape_ideas, self.iteration)

    def _run_checkpoints(self):
        """Run periodic checkpoints."""
        # Contrarian checkpoint every 5 iterations
        if self.iteration % self.config.contrarian_checkpoint_interval == 0:
            self._contrarian_checkpoint()

        # First principles every 10 iterations
        if self.iteration % self.config.first_principles_interval == 0:
            self._first_principles_checkpoint()

        # v3.1: Dynamic cross-domain injection
        should_inject_cross_domain = False

        # Check for scheduled injection
        if self.iteration % self.config.cross_domain_injection_interval == 0:
            should_inject_cross_domain = True

        # v3.1: Check for dynamic trigger based on low novelty scores
        if self.config.dynamic_cross_domain and not should_inject_cross_domain:
            should_inject_cross_domain = self._check_dynamic_cross_domain_trigger()

        if should_inject_cross_domain:
            self._cross_domain_injection()

    def _check_dynamic_cross_domain_trigger(self) -> bool:
        """
        v3.1: Check if cross-domain injection should be triggered dynamically.

        Triggers when:
        1. Average novelty score of last N ideas is below threshold
        2. Plateau detected (score variance < 2 for 5 iterations)

        Returns:
            True if cross-domain injection should be triggered
        """
        if len(self.accepted_ideas) < self.config.novelty_check_window:
            return False

        # Get last N ideas' novelty scores
        recent_ideas = self.accepted_ideas[-self.config.novelty_check_window:]
        novelty_scores = [
            r.dimension_scores.get("novelty", 50)
            for r in recent_ideas
        ]

        # Check 1: Low average novelty
        avg_novelty = sum(novelty_scores) / len(novelty_scores)
        if avg_novelty < self.config.low_novelty_threshold:
            self.learnings.append(
                f"DYNAMIC INJECTION: Low novelty detected (avg={avg_novelty:.1f}), forcing cross-domain"
            )
            return True

        # Check 2: Low score variance (plateau indicator)
        if len(self.accepted_ideas) >= 5:
            recent_scores = [r.final_score for r in self.accepted_ideas[-5:]]
            mean = sum(recent_scores) / len(recent_scores)
            variance = sum((x - mean) ** 2 for x in recent_scores) / len(recent_scores)

            if variance < 2.0:
                self.learnings.append(
                    f"DYNAMIC INJECTION: Score plateau detected (variance={variance:.2f}), forcing cross-domain"
                )
                return True

        return False

    def _contrarian_checkpoint(self):
        """Run contrarian challenge checkpoint."""
        # Would invoke contrarian-disruptor agent
        pass

    def _first_principles_checkpoint(self):
        """Run first-principles validation checkpoint."""
        # Would invoke first-principles-analyst agent
        if self.accepted_ideas:
            best = max(self.accepted_ideas, key=lambda x: x.final_score)
            # Validate best idea strategically

    def _cross_domain_injection(self):
        """Force cross-domain analogy injection."""
        # Force next idea to include analogy from different field
        self.learnings.append(
            "INJECTION: Next idea must include analogy from unrelated domain"
        )

    def _calculate_final_score(
        self,
        dimension_scores: Dict[str, float],
        reward_result: Optional[RewardBreakdown]
    ) -> float:
        """Calculate final weighted score."""
        weighted_sum = 0
        for dim, weight in self.config.dimension_weights.items():
            score = dimension_scores.get(dim, 50)  # Default 50 if missing
            weighted_sum += score * weight

        # Add DARLING bonuses if available
        if reward_result:
            weighted_sum += reward_result.diversity_bonus * 0.1
            weighted_sum += reward_result.exploration_bonus * 0.1

        return min(100, weighted_sum)

    def _summarize_prior_ideas(self) -> str:
        """Create summary of prior ideas for escape prompts."""
        if not self.accepted_ideas:
            return "No prior ideas yet."

        summaries = []
        for result in self.accepted_ideas[-10:]:
            title = result.idea.get("title", "Untitled")
            score = result.final_score
            summaries.append(f"- {title} (score: {score:.0f})")

        return "\n".join(summaries)

    def _generate_idea_stub(self, mode: GeneratorMode) -> Dict:
        """Stub idea generator for testing."""
        return {
            "title": f"Test Idea {self.iteration} ({mode.value})",
            "description": f"Generated in {mode.value} mode",
            "target_market": "Test market",
            "differentiators": ["Test feature"]
        }

    def _evaluate_stub(self, idea: Dict) -> Dict[str, float]:
        """Stub evaluator for testing."""
        import random
        return {
            "novelty": random.uniform(50, 90),
            "feasibility": random.uniform(50, 90),
            "market": random.uniform(50, 90),
            "complexity": random.uniform(50, 90),
            "scenario": random.uniform(50, 90),
            "contrarian": random.uniform(50, 90)
        }

    def _compile_results(self) -> SessionResult:
        """Compile final session results."""
        duration = time.time() - self.start_time if self.start_time else 0

        all_scores = [r.final_score for r in self.accepted_ideas]
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        best_score = max(all_scores) if all_scores else 0
        best_idea = max(self.accepted_ideas, key=lambda x: x.final_score) if self.accepted_ideas else None

        # Calculate dimension averages
        dim_totals: Dict[str, List[float]] = {}
        for result in self.accepted_ideas:
            for dim, score in result.dimension_scores.items():
                if dim not in dim_totals:
                    dim_totals[dim] = []
                dim_totals[dim].append(score)

        dim_averages = {
            dim: sum(scores) / len(scores) if scores else 0
            for dim, scores in dim_totals.items()
        }

        return SessionResult(
            domain=self.domain,
            total_iterations=self.iteration,
            duration_seconds=duration,
            accepted_ideas=self.accepted_ideas,
            rejected_ideas=self.rejected_ideas,
            escape_attempts=len(self.escape_protocol.escape_attempts),
            escape_successful=self.escape_protocol.status == PlateauStatus.ESCAPE_SUCCESSFUL,
            average_score=avg_score,
            best_score=best_score,
            best_idea=best_idea,
            learnings=self.learnings,
            generator_mode_distribution=self.mode_counts,
            dimension_averages=dim_averages,
            plateau_info=self.escape_protocol.get_statistics()
        )

    def export_results(self, filepath: str):
        """Export results to JSON file."""
        results = self._compile_results()

        # Convert to serializable format
        export_data = {
            "meta": {
                "domain": results.domain,
                "total_iterations": results.total_iterations,
                "duration_seconds": results.duration_seconds,
                "average_score": results.average_score,
                "best_score": results.best_score,
                "escape_attempts": results.escape_attempts,
                "escape_successful": results.escape_successful
            },
            "ideas": [
                {
                    "iteration": r.iteration,
                    "idea": r.idea,
                    "mode": r.generator_mode.value,
                    "final_score": r.final_score,
                    "dimension_scores": r.dimension_scores,
                    "accepted": r.accepted
                }
                for r in results.accepted_ideas
            ],
            "statistics": {
                "mode_distribution": results.generator_mode_distribution,
                "dimension_averages": results.dimension_averages,
                "plateau_info": results.plateau_info
            },
            "learnings": results.learnings
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


# v3.3: Database integration for interview context
def get_initiative_domain(initiative_id: str, db_path: str = None) -> Optional[str]:
    """
    Fetch enriched_domain from the interview database for an initiative.

    Args:
        initiative_id: UUID of the initiative
        db_path: Path to ideation.db (defaults to ~/.claude/data/ideation.db)

    Returns:
        The enriched_domain string if found, None otherwise
    """
    import sqlite3

    if db_path is None:
        db_path = os.path.expanduser("~/.claude/data/ideation.db")

    if not os.path.exists(db_path):
        print(f"ERROR: Database not found at {db_path}")
        return None

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT enriched_domain, name, status FROM initiatives WHERE id = ?",
            (initiative_id,)
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            print(f"ERROR: Initiative {initiative_id} not found")
            return None

        enriched_domain, name, status = row

        if status != "ready":
            print(f"WARNING: Initiative '{name}' status is '{status}' (not 'ready')")

        if not enriched_domain:
            print(f"ERROR: Initiative '{name}' has no enriched_domain")
            return None

        print(f"Loaded initiative: {name}")
        print(f"Status: {status}")
        return enriched_domain

    except sqlite3.Error as e:
        print(f"ERROR: Database error: {e}")
        return None


# v3.3: LLM-powered idea generation
def create_llm_idea_generator(model: str = "claude-sonnet-4-20250514", verbose: bool = False):
    """
    Factory function to create an LLM-powered idea generator.

    Returns a callable that generates ideas using the Anthropic API.
    """
    try:
        import anthropic
    except ImportError:
        print("ERROR: anthropic package not installed. Run: pip install anthropic")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    def generate_idea(
        domain: str,
        mode: GeneratorMode,
        learnings: List[str],
        escape_prompt: Optional[str] = None
    ) -> Dict:
        """Generate idea using Anthropic API."""

        # Build mode-specific instructions
        mode_instructions = {
            GeneratorMode.EXPLORER: """You are in EXPLORER mode. Focus on:
- Maximizing novelty and originality
- Blue ocean thinking - unexplored spaces
- Cross-domain analogies and unconventional combinations
- Bold, potentially risky ideas
- Push boundaries, don't play safe""",

            GeneratorMode.REFINER: """You are in REFINER mode. Focus on:
- Optimizing feasibility and implementation clarity
- Building on known patterns that work
- Practical improvements to existing approaches
- Clear path to execution
- Risk mitigation and pragmatic solutions""",

            GeneratorMode.CONTRARIAN: """You are in CONTRARIAN mode. Focus on:
- Challenging dominant assumptions in the domain
- Finding value in what others dismiss
- Inverting conventional wisdom
- Identifying hidden opportunities others miss
- Question "obvious" best practices"""
        }

        # Build learnings context
        learnings_text = ""
        if learnings:
            relevant = [l for l in learnings if not l.startswith("INJECTION")]
            if relevant:
                learnings_text = "\n\nPrior learnings from this session:\n" + "\n".join(f"- {l}" for l in relevant[-5:])

            # Handle injection directives
            injections = [l for l in learnings if l.startswith("INJECTION")]
            if injections:
                learnings_text += "\n\nMANDATORY REQUIREMENTS:\n" + "\n".join(f"- {l.replace('INJECTION: ', '')}" for l in injections)

        # Build escape context if provided
        escape_text = ""
        if escape_prompt:
            escape_text = f"\n\nESCAPE DIRECTIVE: {escape_prompt}"

        prompt = f"""Generate an innovative idea for the following domain:

DOMAIN: {domain}

{mode_instructions.get(mode, mode_instructions[GeneratorMode.EXPLORER])}
{learnings_text}
{escape_text}

Generate ONE concrete, actionable idea. Your response MUST be valid JSON with this exact structure:
{{
    "title": "Concise descriptive title (max 10 words)",
    "description": "2-3 sentence description of the core concept",
    "mechanism": "How it works technically/practically",
    "target_market": "Who benefits and why",
    "differentiators": ["Key unique aspect 1", "Key unique aspect 2", "Key unique aspect 3"],
    "implementation_path": "High-level steps to make this real",
    "potential_challenges": ["Challenge 1", "Challenge 2"]
}}

Return ONLY the JSON object, no other text."""

        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            idea = json.loads(response_text)

            if verbose:
                print(f"    [LLM] Generated: {idea.get('title', 'Untitled')[:50]}")

            return idea

        except json.JSONDecodeError as e:
            print(f"    [LLM] JSON parse error: {e}")
            # Return minimal valid idea
            return {
                "title": f"Parse Error Idea ({mode.value})",
                "description": f"Failed to parse LLM response",
                "target_market": domain,
                "differentiators": ["Generated during error recovery"]
            }
        except Exception as e:
            print(f"    [LLM] API error: {e}")
            return {
                "title": f"API Error Idea ({mode.value})",
                "description": str(e)[:100],
                "target_market": domain,
                "differentiators": []
            }

    return generate_idea


# Quick test function
def test_orchestrator():
    """Quick test of orchestrator."""
    orchestrator = IdeationOrchestrator(domain="protein beverages")

    def on_iter(iteration, result):
        print(f"Iteration {iteration}: {result.idea['title']} - "
              f"Score: {result.final_score:.1f} - "
              f"{'Accepted' if result.accepted else 'Rejected'}")

    results = orchestrator.run(
        max_iterations=10,
        max_minutes=5,
        on_iteration=on_iter
    )

    print(f"\nSession Complete!")
    print(f"Total iterations: {results.total_iterations}")
    print(f"Accepted: {len(results.accepted_ideas)}")
    print(f"Rejected: {len(results.rejected_ideas)}")
    print(f"Average score: {results.average_score:.1f}")
    print(f"Best score: {results.best_score:.1f}")
    print(f"Mode distribution: {results.generator_mode_distribution}")

    return results


def main():
    """CLI interface for skill invocation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal Ideation v3.1 - Science-Grounded Ideation System"
    )
    parser.add_argument(
        "domain",
        nargs="?",
        default=None,
        help="Domain/focus area for ideation (or use --initiative)"
    )
    parser.add_argument(
        "--initiative",
        type=str,
        default=None,
        help="Initiative ID to load enriched_domain from interview database"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=30,
        help="Maximum iterations (default: 30)"
    )
    parser.add_argument(
        "--minutes", "-m",
        type=int,
        default=30,
        help="Maximum minutes (default: 30)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=65.0,
        help="Acceptance threshold (default: 65.0)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path"
    )

    # v3.1: Constraint-based creativity options
    parser.add_argument(
        "--constraints", "-c",
        type=str,
        choices=["none", "bootstrap", "enterprise", "regulated", "sustainable", "custom"],
        default="none",
        help="Constraint template: bootstrap (MVP/lean), enterprise (scale/compliance), "
             "regulated (FDA/safety), sustainable (eco/circular), custom (use --custom-constraints)"
    )
    parser.add_argument(
        "--custom-constraints",
        type=str,
        nargs="+",
        default=[],
        help="Custom constraints when using --constraints=custom (e.g., 'budget under 10k' 'no AI')"
    )
    parser.add_argument(
        "--budget",
        type=str,
        default=None,
        help="Budget constraint (e.g., '$50k', 'minimal', 'unlimited')"
    )
    parser.add_argument(
        "--timeline",
        type=str,
        default=None,
        help="Time-to-market constraint (e.g., '3 months', 'fast', '1 year')"
    )

    # v3.1: Dynamic cross-domain option
    parser.add_argument(
        "--no-dynamic-crossdomain",
        action="store_true",
        help="Disable dynamic cross-domain injection based on novelty scores"
    )

    # v3.2: Verification gate options
    parser.add_argument(
        "--no-verification",
        action="store_true",
        help="Disable verification gates (not recommended)"
    )
    parser.add_argument(
        "--verification-retries",
        type=int,
        default=3,
        help="Maximum verification retry attempts (default: 3)"
    )
    parser.add_argument(
        "--strict-verification",
        action="store_true",
        help="Enable strict mode (soft warnings also cause rejection)"
    )
    parser.add_argument(
        "--llm-verification",
        action="store_true",
        help="Enable LLM-based semantic verification (more thorough but slower)"
    )

    # v3.2: Reflection Learning options
    parser.add_argument(
        "--no-reflection",
        action="store_true",
        help="Disable reflection learning (not recommended for repeated domains)"
    )
    parser.add_argument(
        "--reflection-batch",
        type=int,
        default=5,
        help="Number of accepted ideas before reflection analysis (default: 5)"
    )
    parser.add_argument(
        "--no-weight-adaptation",
        action="store_true",
        help="Disable adaptive weight adjustment from reflections"
    )
    parser.add_argument(
        "--no-persistence",
        action="store_true",
        help="Disable SQLite persistence (learnings not saved)"
    )
    parser.add_argument(
        "--no-prior-learnings",
        action="store_true",
        help="Start fresh without loading prior domain learnings"
    )

    # v3.2: Atomic Novelty options
    parser.add_argument(
        "--no-atomic-novelty",
        action="store_true",
        help="Disable atomic novelty scoring (uses simpler novelty)"
    )
    parser.add_argument(
        "--atomic-novelty-weight",
        type=float,
        default=0.3,
        help="Weight of atomic novelty in novelty score (default: 0.3)"
    )
    parser.add_argument(
        "--llm-novelty",
        action="store_true",
        help="Use LLM for novelty decomposition/detection (slower but better)"
    )
    parser.add_argument(
        "--min-novelty",
        type=float,
        default=40.0,
        help="Minimum atomic novelty score threshold (default: 40.0)"
    )

    # v3.3: LLM-powered generation
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Anthropic API for real LLM-powered idea generation (requires ANTHROPIC_API_KEY)"
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Anthropic model to use (default: claude-sonnet-4-20250514)"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run quick test with stub generators"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.test:
        test_orchestrator()
        return

    # v3.3: Resolve domain from initiative or argument
    domain = args.domain
    if args.initiative:
        domain = get_initiative_domain(args.initiative)
        if domain is None:
            print("Failed to load initiative. Use --domain instead or check the initiative ID.")
            return
    elif domain is None:
        domain = "general innovation"

    # v3.1: Build constraint configuration
    constraint_config = None
    if args.constraints != "none":
        constraint_template = ConstraintTemplate(args.constraints)
        constraint_config = ConstraintConfig(
            template=constraint_template,
            custom_constraints=args.custom_constraints,
            budget_limit=args.budget,
            time_to_market=args.timeline
        )

    # Configure orchestrator
    config = OrchestratorConfig(
        max_iterations=args.iterations,
        max_minutes=args.minutes,
        acceptance_threshold=args.threshold,
        constraints=constraint_config,
        dynamic_cross_domain=not args.no_dynamic_crossdomain,
        # v3.2: Verification settings
        enable_verification=not args.no_verification,
        verification_max_retries=args.verification_retries,
        verification_strict_mode=args.strict_verification,
        enable_llm_verification=args.llm_verification,
        # v3.2: Reflection Learning settings
        enable_reflection=not args.no_reflection,
        reflection_batch_size=args.reflection_batch,
        enable_weight_adaptation=not args.no_weight_adaptation,
        enable_persistence=not args.no_persistence,
        load_prior_learnings=not args.no_prior_learnings,
        # v3.2: Atomic Novelty settings
        enable_atomic_novelty=not args.no_atomic_novelty,
        atomic_novelty_weight=args.atomic_novelty_weight,
        use_llm_for_novelty=args.llm_novelty,
        min_novelty_score=args.min_novelty
    )

    orchestrator = IdeationOrchestrator(
        domain=domain,
        config=config
    )

    # v3.3: Create LLM generator if requested
    idea_generator = None
    if args.use_llm:
        idea_generator = create_llm_idea_generator(
            model=args.llm_model,
            verbose=args.verbose
        )
        if idea_generator is None:
            print("WARNING: LLM generator unavailable, falling back to stub generators")

    def on_iteration(iteration, result):
        if args.verbose:
            status = "+" if result.accepted else "-"
            print(f"[{status}] {iteration:3d}: {result.idea.get('title', 'Untitled')[:50]} "
                  f"| {result.final_score:.1f} | {result.generator_mode.value}")

    print(f"\n{'='*60}")
    print(f"UNIVERSAL IDEATION v3.3")
    print(f"{'='*60}")
    if args.initiative:
        print(f"Initiative: {args.initiative}")
    print(f"Domain: {domain[:100]}..." if len(domain) > 100 else f"Domain: {domain}")
    print(f"Config: {args.iterations} iterations, {args.minutes} min, threshold {args.threshold}")
    if constraint_config:
        print(f"Constraints: {args.constraints}")
        if args.budget:
            print(f"  Budget: {args.budget}")
        if args.timeline:
            print(f"  Timeline: {args.timeline}")
    print(f"Dynamic cross-domain: {'enabled' if not args.no_dynamic_crossdomain else 'disabled'}")
    print(f"Verification gates: {'enabled' if not args.no_verification else 'DISABLED'}")
    if not args.no_verification:
        print(f"  Retries: {args.verification_retries}")
        print(f"  Strict mode: {'yes' if args.strict_verification else 'no'}")
        print(f"  LLM verification: {'yes' if args.llm_verification else 'no'}")
    print(f"Reflection learning: {'enabled' if not args.no_reflection else 'DISABLED'}")
    if not args.no_reflection:
        print(f"  Batch size: {args.reflection_batch}")
        print(f"  Weight adaptation: {'yes' if not args.no_weight_adaptation else 'no'}")
        print(f"  Persistence: {'yes' if not args.no_persistence else 'no'}")
        print(f"  Prior learnings: {'yes' if not args.no_prior_learnings else 'no'}")
    print(f"Atomic novelty: {'enabled' if not args.no_atomic_novelty else 'DISABLED'}")
    if not args.no_atomic_novelty:
        print(f"  Weight: {args.atomic_novelty_weight:.0%}")
        print(f"  LLM-enhanced: {'yes' if args.llm_novelty else 'no'}")
        print(f"  Min threshold: {args.min_novelty}")
    # v3.3: LLM generation status
    print(f"LLM generation: {'ENABLED' if idea_generator else 'disabled (stub mode)'}")
    if idea_generator:
        print(f"  Model: {args.llm_model}")
    print(f"{'='*60}\n")

    results = orchestrator.run(
        idea_generator=idea_generator,
        on_iteration=on_iteration
    )

    print(f"\n{'='*60}")
    print(f"SESSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {results.total_iterations}")
    print(f"Accepted ideas:   {len(results.accepted_ideas)}")
    print(f"Rejected ideas:   {len(results.rejected_ideas)}")
    print(f"Average score:    {results.average_score:.1f}")
    print(f"Best score:       {results.best_score:.1f}")
    print(f"Escape attempts:  {results.escape_attempts}")
    print(f"Mode distribution: {results.generator_mode_distribution}")

    # v3.2: Verification diagnostics summary
    if orchestrator.verification_diagnostics:
        health = orchestrator.verification_diagnostics.get_session_health()
        print(f"\nVerification Gate Health: {health.status.upper()} ({health.health_score:.0%})")
        print(f"  Pass rate: {health.pass_rate:.0%}")
        print(f"  First-try rate: {health.first_try_rate:.0%}")
        if health.failed_permanently > 0:
            print(f"  Permanently failed: {health.failed_permanently}")

    # v3.2: Reflection learning summary
    if orchestrator.weight_adjuster:
        weight_state = orchestrator.weight_adjuster.get_weight_state()
        stats = orchestrator.weight_adjuster.get_statistics()
        print(f"\nReflection Learning Summary:")
        print(f"  Weight adjustments: {weight_state.total_adjustments}")
        if stats.get("by_dimension"):
            top_dims = sorted(stats["by_dimension"].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  Most adjusted: {', '.join(f'{d}({n})' for d, n in top_dims)}")

        # Show dimension deviations from default
        deviation_report = orchestrator.weight_adjuster.get_deviation_report()
        significant_deviations = [
            (dim, info) for dim, info in deviation_report.items()
            if abs(info["deviation_pct"]) > 5
        ]
        if significant_deviations:
            print(f"  Significant weight changes:")
            for dim, info in sorted(significant_deviations, key=lambda x: abs(x[1]["deviation_pct"]), reverse=True)[:3]:
                print(f"    {dim}: {info['deviation_pct']:+.1f}%")

    if orchestrator.learning_persistence:
        stats = orchestrator.learning_persistence.get_statistics()
        print(f"\n  Persistence: {stats['total_sessions']} sessions, {stats['total_reflections']} reflections")

    # v3.2: Atomic novelty summary
    if orchestrator.atomic_novelty_scorer:
        stats = orchestrator.atomic_novelty_scorer.get_statistics()
        print(f"\nAtomic Novelty Summary:")
        print(f"  Ideas scored: {stats['ideas_scored']}")
        print(f"  Prior claims indexed: {stats['prior_ideas_processed']}")

        # Calculate tier distribution from results
        tier_counts = {"breakthrough": 0, "highly_novel": 0, "novel": 0, "incremental": 0, "derivative": 0}
        for result in results.accepted_ideas:
            if result.atomic_novelty_result:
                tier = result.atomic_novelty_result.novelty_tier.value
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

        if any(tier_counts.values()):
            print(f"  Tier distribution: {', '.join(f'{k}:{v}' for k, v in tier_counts.items() if v > 0)}")

    print(f"{'='*60}\n")

    # Export if output specified
    if args.output:
        orchestrator.export_results(args.output)
        print(f"Results exported to: {args.output}")
    else:
        # Default export to project output dir
        from pathlib import Path
        output_dir = Path(__file__).parent.parent / "output"
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"ideation_{timestamp}.json"
        orchestrator.export_results(str(output_path))
        print(f"Results exported to: {output_path}")


if __name__ == "__main__":
    main()
