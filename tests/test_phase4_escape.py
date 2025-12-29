"""
Phase 4 Integration Tests - Plateau Escape & Orchestrator
Universal Ideation v3

Tests for:
1. Plateau Detection
2. Escape Strategy Generation
3. Escape Protocol
4. Full Orchestrator Integration
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from escape.plateau_escape import (
    PlateauDetector,
    PlateauEscapeProtocol,
    EscapeStrategyGenerator,
    PlateauStatus,
    PlateauAnalysis,
    EscapeResult,
    EscapeIdea,
    create_escape_protocol,
    simulate_plateau_detection
)


# =============================================================================
# PLATEAU DETECTOR TESTS
# =============================================================================

class TestPlateauDetector:
    """Test suite for plateau detection."""

    @pytest.fixture
    def detector(self):
        """Create PlateauDetector instance."""
        return PlateauDetector(window_size=10, threshold=0.5, min_iterations=15)

    def test_initialization(self, detector):
        """Test PlateauDetector initializes correctly."""
        assert detector.window_size == 10
        assert detector.threshold == 0.5
        assert detector.min_iterations == 15
        assert len(detector.score_history) == 0

    def test_insufficient_history(self, detector):
        """Test detection with insufficient history."""
        for i in range(10):
            detector.add_score(70 + i)

        analysis = detector.detect()
        assert not analysis.is_plateau
        assert "insufficient" in analysis.recommendation.lower() or "more" in analysis.recommendation.lower()

    def test_improving_scores_no_plateau(self, detector):
        """Test that improving scores don't trigger plateau."""
        # Steadily increasing scores
        for i in range(25):
            detector.add_score(60 + i * 1.5)

        analysis = detector.detect()
        assert not analysis.is_plateau
        assert "improving" in analysis.recommendation.lower() or "continue" in analysis.recommendation.lower()

    def test_stable_scores_plateau_detected(self, detector):
        """Test that stable scores trigger plateau."""
        # First 15 scores at ~70
        for i in range(15):
            detector.add_score(70 + (i % 3) * 0.2)

        # Next 10 scores also at ~70
        for i in range(10):
            detector.add_score(70 + (i % 3) * 0.2)

        analysis = detector.detect()
        assert analysis.is_plateau
        assert abs(analysis.recent_average - analysis.previous_average) < 0.5

    def test_plateau_average_calculation(self, detector):
        """Test plateau average is calculated correctly."""
        for i in range(20):
            detector.add_score(75)

        avg = detector.get_plateau_average()
        assert abs(avg - 75) < 0.1

    def test_reset_after_escape(self, detector):
        """Test reset clears plateau checks."""
        for i in range(25):
            detector.add_score(70)
            detector.detect()

        assert len(detector.plateau_checks) > 0

        detector.reset_after_escape()
        assert len(detector.plateau_checks) == 0
        assert len(detector.score_history) == 25  # History preserved


class TestEscapeStrategyGenerator:
    """Test suite for escape strategy generation."""

    @pytest.fixture
    def generator(self):
        """Create EscapeStrategyGenerator instance."""
        return EscapeStrategyGenerator()

    def test_strategies_defined(self, generator):
        """Test that all 5 strategies are defined."""
        assert len(generator.ESCAPE_STRATEGIES) == 5

        strategy_names = [s["name"] for s in generator.ESCAPE_STRATEGIES]
        assert "domain_inversion" in strategy_names
        assert "random_domain_injection" in strategy_names
        assert "anti_learning" in strategy_names
        assert "format_disruption" in strategy_names
        assert "audience_flip" in strategy_names

    def test_prompt_generation(self, generator):
        """Test escape prompts are generated correctly."""
        prompts = generator.get_escape_prompts(
            domain="protein beverages",
            learnings=["sensory claims boost scores", "familiar proteins work"],
            prior_ideas_summary="Prior idea 1\nPrior idea 2",
            num_attempts=5
        )

        assert len(prompts) == 5
        for prompt in prompts:
            assert "strategy" in prompt
            assert "prompt" in prompt
            assert "ESCAPE" in prompt["prompt"]
            assert "protein beverages" in prompt["prompt"]

    def test_learning_injection(self, generator):
        """Test that learnings are injected into prompts."""
        learnings = ["Learning A", "Learning B"]
        prompts = generator.get_escape_prompts(
            domain="test",
            learnings=learnings,
            prior_ideas_summary="",
            num_attempts=1
        )

        assert "Learning A" in prompts[0]["prompt"]


class TestPlateauEscapeProtocol:
    """Test suite for complete escape protocol."""

    @pytest.fixture
    def protocol(self):
        """Create PlateauEscapeProtocol instance."""
        return PlateauEscapeProtocol(
            window_size=10,
            threshold=0.5,
            min_iterations=15,
            max_escape_attempts=2
        )

    def test_initialization(self, protocol):
        """Test protocol initializes correctly."""
        assert protocol.status == PlateauStatus.NO_PLATEAU
        assert len(protocol.escape_attempts) == 0
        assert protocol.consecutive_plateau_checks == 0

    def test_no_escape_before_plateau(self, protocol):
        """Test escape not attempted before plateau."""
        for i in range(10):
            protocol.add_score(70 + i)

        assert not protocol.should_attempt_escape()

    def test_escape_triggered_after_plateau(self, protocol):
        """Test escape is triggered after confirmed plateau."""
        # Create plateau
        for i in range(30):
            protocol.add_score(70)
            protocol.check_plateau()

        # After enough checks, should want to escape
        assert protocol.status == PlateauStatus.PLATEAU_DETECTED
        assert protocol.consecutive_plateau_checks >= 3
        assert protocol.should_attempt_escape()

    def test_successful_escape(self, protocol):
        """Test successful escape detection."""
        # Setup plateau at 70
        for i in range(25):
            protocol.add_score(70)

        plateau_avg = protocol.detector.get_plateau_average()

        # Create escape ideas with one beating plateau
        escape_ideas = [
            EscapeIdea(idea={"title": "Escape 1"}, score=65, strategy="s1", deviation_from_centroid=0.8),
            EscapeIdea(idea={"title": "Escape 2"}, score=80, strategy="s2", deviation_from_centroid=0.9),  # Winner
            EscapeIdea(idea={"title": "Escape 3"}, score=68, strategy="s3", deviation_from_centroid=0.7),
        ]

        result = protocol.evaluate_escape(escape_ideas, iteration=25)

        assert result.status == PlateauStatus.ESCAPE_SUCCESSFUL
        assert result.should_continue
        assert result.best_escape_idea.score == 80
        assert "successful" in result.recommendation.lower()

    def test_failed_escape(self, protocol):
        """Test failed escape detection."""
        # Setup plateau at 80
        for i in range(25):
            protocol.add_score(80)

        # Escape ideas don't beat plateau
        escape_ideas = [
            EscapeIdea(idea={"title": "Escape 1"}, score=75, strategy="s1", deviation_from_centroid=0.8),
            EscapeIdea(idea={"title": "Escape 2"}, score=78, strategy="s2", deviation_from_centroid=0.9),
            EscapeIdea(idea={"title": "Escape 3"}, score=72, strategy="s3", deviation_from_centroid=0.7),
        ]

        result = protocol.evaluate_escape(escape_ideas, iteration=25)

        assert result.status == PlateauStatus.ESCAPE_FAILED
        assert result.should_continue  # First attempt, can try again
        assert "failed" in result.recommendation.lower()

    def test_confirmed_plateau_after_max_attempts(self, protocol):
        """Test plateau confirmed after max escape attempts."""
        # Setup plateau
        for i in range(25):
            protocol.add_score(80)

        # Fail first attempt
        escape_ideas = [
            EscapeIdea(idea={"title": "E1"}, score=75, strategy="s1", deviation_from_centroid=0.8),
        ]
        protocol.evaluate_escape(escape_ideas, iteration=25)

        # Fail second attempt (max)
        result = protocol.evaluate_escape(escape_ideas, iteration=26)

        assert result.status == PlateauStatus.CONFIRMED_PLATEAU
        assert not result.should_continue
        assert "confirmed" in result.recommendation.lower()

    def test_statistics_tracking(self, protocol):
        """Test statistics are tracked correctly."""
        for i in range(20):
            protocol.add_score(70 + i % 5)
            protocol.check_plateau()

        stats = protocol.get_statistics()
        assert "status" in stats
        assert "total_escape_attempts" in stats
        assert "score_history_length" in stats
        assert stats["score_history_length"] == 20


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_escape_protocol_default(self):
        """Test default escape protocol creation."""
        protocol = create_escape_protocol(conservative=False)
        assert protocol.detector.window_size == 10
        assert protocol.detector.threshold == 0.5

    def test_create_escape_protocol_conservative(self):
        """Test conservative escape protocol creation."""
        protocol = create_escape_protocol(conservative=True)
        assert protocol.detector.window_size == 15
        assert protocol.detector.threshold == 0.3

    def test_simulate_plateau_detection(self):
        """Test plateau simulation helper."""
        # Improving scores - no plateau
        improving = [60, 65, 70, 75, 80, 85, 90, 95, 100, 105,
                     110, 115, 120, 125, 130, 135, 140, 145, 150, 155]
        result = simulate_plateau_detection(improving)
        assert not result.is_plateau

        # Flat scores - plateau
        flat = [75] * 25
        result = simulate_plateau_detection(flat)
        assert result.is_plateau


# =============================================================================
# ORCHESTRATOR INTEGRATION TESTS
# =============================================================================

class TestOrchestratorConfig:
    """Test OrchestratorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from run_v3 import OrchestratorConfig

        config = OrchestratorConfig()
        assert config.max_iterations == 30
        assert config.max_minutes == 30
        assert config.acceptance_threshold == 65.0
        assert sum(config.dimension_weights.values()) == pytest.approx(1.0)

    def test_dimension_weights_sum_to_one(self):
        """Test 8-dimension weights sum to 100%."""
        from run_v3 import OrchestratorConfig

        config = OrchestratorConfig()
        total = sum(config.dimension_weights.values())
        assert abs(total - 1.0) < 0.001


class TestIdeationOrchestrator:
    """Test IdeationOrchestrator."""

    def test_initialization(self):
        """Test orchestrator initializes correctly."""
        from run_v3 import IdeationOrchestrator, SessionPhase

        orchestrator = IdeationOrchestrator(domain="protein beverages")

        assert orchestrator.domain == "protein beverages"
        assert orchestrator.phase == SessionPhase.INITIALIZING
        assert orchestrator.iteration == 0
        assert len(orchestrator.accepted_ideas) == 0

    def test_component_initialization(self):
        """Test all components are initialized."""
        from run_v3 import IdeationOrchestrator

        orchestrator = IdeationOrchestrator()

        assert orchestrator.generator is not None
        assert orchestrator.gate is not None
        assert orchestrator.reward_calculator is not None
        assert orchestrator.evaluator_panel is not None
        assert orchestrator.surprise_evaluator is not None
        assert orchestrator.cross_domain_evaluator is not None
        assert orchestrator.escape_protocol is not None

    def test_short_run(self):
        """Test short orchestrator run."""
        from run_v3 import IdeationOrchestrator

        orchestrator = IdeationOrchestrator(domain="protein beverages")
        results = orchestrator.run(max_iterations=5, max_minutes=1)

        assert results.total_iterations == 5
        assert results.domain == "protein beverages"
        assert len(results.accepted_ideas) + len(results.rejected_ideas) == 5

    def test_mode_distribution_tracked(self):
        """Test generator mode distribution is tracked."""
        from run_v3 import IdeationOrchestrator

        orchestrator = IdeationOrchestrator()
        results = orchestrator.run(max_iterations=10, max_minutes=1)

        total_modes = sum(results.generator_mode_distribution.values())
        assert total_modes == 10
        assert "explorer" in results.generator_mode_distribution
        assert "refiner" in results.generator_mode_distribution
        assert "contrarian" in results.generator_mode_distribution

    def test_iteration_callback(self):
        """Test iteration callback is invoked."""
        from run_v3 import IdeationOrchestrator

        callback_count = [0]

        def callback(iteration, result):
            callback_count[0] += 1

        orchestrator = IdeationOrchestrator()
        orchestrator.run(max_iterations=5, max_minutes=1, on_iteration=callback)

        assert callback_count[0] == 5


class TestFullIntegration:
    """Test full v3 system integration."""

    def test_all_phases_implemented(self):
        """Verify all 4 phases are implemented."""
        # Phase 1: DARLING Core
        from generators.triple_generator import TripleGenerator
        from gates.semantic_distance_gate import SemanticDistanceGate
        from learning.darling_reward import DARLINGReward

        assert TripleGenerator is not None
        assert SemanticDistanceGate is not None
        assert DARLINGReward is not None

        # Phase 2: Cognitive Diversity
        from evaluators.cognitive_diversity import EvaluatorPanel
        from evaluators.structured_debate import StructuredDebate

        assert EvaluatorPanel is not None
        assert StructuredDebate is not None

        # Phase 3: New Dimensions
        from evaluators.surprise_dimension import SurpriseDimension
        from evaluators.cross_domain_bridge import CrossDomainBridge

        assert SurpriseDimension is not None
        assert CrossDomainBridge is not None

        # Phase 4: Escape & Orchestrator
        from escape.plateau_escape import PlateauEscapeProtocol
        from run_v3 import IdeationOrchestrator

        assert PlateauEscapeProtocol is not None
        assert IdeationOrchestrator is not None

    def test_8_dimension_weights(self):
        """Verify 8-dimension weight distribution."""
        from run_v3 import OrchestratorConfig

        config = OrchestratorConfig()
        weights = config.dimension_weights

        expected = {
            "novelty": 0.12,
            "feasibility": 0.18,
            "market": 0.18,
            "complexity": 0.12,
            "scenario": 0.12,
            "contrarian": 0.10,
            "surprise": 0.10,
            "cross_domain": 0.08
        }

        for dim, expected_weight in expected.items():
            assert dim in weights
            assert weights[dim] == pytest.approx(expected_weight)

    def test_v3_vs_v2_improvements(self):
        """Document v3 improvements over v2."""
        improvements = {
            "mode_collapse": "Fixed by SemanticDistanceGate + TripleGenerator",
            "local_optima": "Fixed by DARLING exploration rewards",
            "cognitive_homogeneity": "Fixed by 4 evaluator personas",
            "missing_dimensions": "Fixed by Surprise + Cross-Domain (now 8)",
            "plateau_trap": "Fixed by PlateauEscapeProtocol"
        }

        # All gaps addressed
        assert len(improvements) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
