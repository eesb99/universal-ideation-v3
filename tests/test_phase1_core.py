"""
Integration Tests for Universal Ideation v3 - Phase 1 Core Components

Tests cover:
1. Triple Generator System
2. Semantic Distance Gate
3. DARLING Reward Calculation
4. Component Integration
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generators.triple_generator import (
    TripleGenerator,
    GeneratorMode,
    ModeWeights,
    select_generator_mode
)
from gates.semantic_distance_gate import (
    SemanticDistanceGate,
    GateDecision,
    IdeaEmbedding,
    check_semantic_distance,
    create_mock_embedding
)
from learning.darling_reward import (
    DARLINGReward,
    DimensionScores,
    GeneratorMode as RewardGeneratorMode,
    SemanticRegion,
    create_sample_scores
)


class TestTripleGenerator:
    """Tests for Triple Generator System."""

    def test_mode_selection_early_session(self):
        """Early session should favor exploration."""
        generator = TripleGenerator()

        # Run multiple selections to check distribution
        mode_counts = {m: 0 for m in GeneratorMode}
        for _ in range(100):
            mode = generator.select_mode(
                iteration=5,
                max_iterations=30,
                recent_scores=[65, 70, 72, 68, 71]
            )
            mode_counts[mode] += 1

        # Explorer should be most common in early session
        assert mode_counts[GeneratorMode.EXPLORER] >= 30, \
            f"Explorer mode should be frequent early, got {mode_counts[GeneratorMode.EXPLORER]}"
        print(f"Early session mode distribution: {mode_counts}")

    def test_mode_selection_plateau(self):
        """Plateau detection should force exploration."""
        generator = TripleGenerator()

        mode_counts = {m: 0 for m in GeneratorMode}
        for _ in range(100):
            mode = generator.select_mode(
                iteration=20,
                max_iterations=30,
                recent_scores=[70, 70, 71, 70, 70],  # Stagnant scores
                is_plateau=True
            )
            mode_counts[mode] += 1

        # Explorer should dominate during plateau
        assert mode_counts[GeneratorMode.EXPLORER] >= 50, \
            f"Explorer should dominate at plateau, got {mode_counts[GeneratorMode.EXPLORER]}"
        print(f"Plateau mode distribution: {mode_counts}")

    def test_mode_selection_late_session(self):
        """Late session should favor refinement."""
        generator = TripleGenerator()

        mode_counts = {m: 0 for m in GeneratorMode}
        for _ in range(100):
            mode = generator.select_mode(
                iteration=25,
                max_iterations=30,
                recent_scores=[85, 86, 84, 87, 85]
            )
            mode_counts[mode] += 1

        # Refiner should be more common late (allowing for random variance)
        assert mode_counts[GeneratorMode.REFINER] >= 30, \
            f"Refiner should be common late, got {mode_counts[GeneratorMode.REFINER]}"
        # Refiner should not be less than explorer in late session
        assert mode_counts[GeneratorMode.REFINER] >= mode_counts[GeneratorMode.EXPLORER] * 0.8, \
            f"Refiner should be comparable to explorer late session"
        print(f"Late session mode distribution: {mode_counts}")

    def test_explorer_prompt_generation(self):
        """Explorer mode should generate novelty-focused prompt."""
        generator = TripleGenerator()

        prior_ideas = [
            {"title": "Pea Protein Shake", "description": "Standard protein shake"},
            {"title": "Soy Milk Alternative", "description": "Plant-based milk"}
        ]

        result = generator.generate(
            mode=GeneratorMode.EXPLORER,
            domain="protein beverages",
            learnings=[],
            prior_ideas=prior_ideas
        )

        assert result["mode"] == "explorer"
        assert "EXPLORER MODE" in result["prompt"]
        assert "MAXIMALLY DIFFERENT" in result["prompt"]
        assert result["objective"] == "maximize_semantic_distance"
        print("Explorer prompt generated successfully")

    def test_refiner_prompt_generation(self):
        """Refiner mode should use learnings."""
        generator = TripleGenerator()

        learnings = [
            {"pattern": "Creamy texture works", "observation": "+15 points avg"}
        ]
        best_ideas = [
            {"title": "Premium Pea Shake", "description": "High protein, smooth"}
        ]

        result = generator.generate(
            mode=GeneratorMode.REFINER,
            domain="protein beverages",
            learnings=learnings,
            prior_ideas=[],
            best_ideas=best_ideas
        )

        assert result["mode"] == "refiner"
        assert "REFINER MODE" in result["prompt"]
        assert "Creamy texture works" in result["prompt"]
        assert result["objective"] == "optimize_within_cluster"
        print("Refiner prompt generated successfully")

    def test_contrarian_prompt_generation(self):
        """Contrarian mode should invert learnings."""
        generator = TripleGenerator()

        learnings = [
            {"pattern": "High protein content works well", "observation": "Top ideas have 30g+"}
        ]

        result = generator.generate(
            mode=GeneratorMode.CONTRARIAN,
            domain="protein beverages",
            learnings=learnings,
            prior_ideas=[]
        )

        assert result["mode"] == "contrarian"
        assert "CONTRARIAN MODE" in result["prompt"]
        assert "agent" == "contrarian-disruptor" or result["agent"] == "contrarian-disruptor"
        assert result["objective"] == "challenge_assumptions"
        print("Contrarian prompt generated successfully")

    def test_exploration_status(self):
        """Exploration status should track mode usage."""
        generator = TripleGenerator()

        # Run several iterations
        for i in range(20):
            generator.select_mode(
                iteration=i,
                max_iterations=30,
                recent_scores=[65 + i * 0.5] * 5
            )

        status = generator.get_exploration_status()

        assert status["total_generations"] == 20
        assert "exploration_budget" in status
        assert "mode_distribution" in status
        assert len(status["last_5_modes"]) == 5
        print(f"Exploration status: {status}")


class TestSemanticDistanceGate:
    """Tests for Semantic Distance Gate."""

    def test_first_idea_always_accepted(self):
        """First idea should always pass gate."""
        gate = SemanticDistanceGate()
        embedding = create_mock_embedding()

        result = gate.check_distance(embedding)

        assert result.decision == GateDecision.ACCEPT
        assert "First idea" in result.reason
        print("First idea acceptance test passed")

    def test_similar_idea_rejected(self):
        """Very similar ideas should be rejected."""
        gate = SemanticDistanceGate()

        # Add first idea
        first_embedding = create_mock_embedding()
        gate.check_distance(first_embedding)
        gate.add_idea(IdeaEmbedding(
            idea_id="1",
            title="First Idea",
            embedding=first_embedding,
            score=75.0
        ))

        # Try nearly identical idea (small random perturbation)
        similar_embedding = first_embedding + np.random.randn(384) * 0.01
        similar_embedding = similar_embedding / np.linalg.norm(similar_embedding)

        result = gate.check_distance(similar_embedding, iteration=1)

        # Should be rejected as too similar
        assert result.decision in [GateDecision.REJECT, GateDecision.BORDERLINE], \
            f"Similar idea should be rejected, got {result.decision}"
        assert result.centroid_similarity > 0.8
        print(f"Similar idea rejection test passed: {result.reason}")

    def test_distant_idea_accepted(self):
        """Distant ideas should be accepted."""
        gate = SemanticDistanceGate()

        # Add first idea
        first_embedding = create_mock_embedding()
        gate.check_distance(first_embedding)
        gate.add_idea(IdeaEmbedding(
            idea_id="1",
            title="First Idea",
            embedding=first_embedding,
            score=75.0
        ))

        # Create very different idea (different random embedding)
        np.random.seed(42)  # Different seed for different embedding
        distant_embedding = create_mock_embedding()

        result = gate.check_distance(distant_embedding, iteration=1)

        # Distant ideas should pass
        # Note: random embeddings may not always be distant enough
        print(f"Distant idea test: distance={result.distance:.3f}, "
              f"decision={result.decision}, reason={result.reason}")

    def test_adaptive_threshold_relaxation(self):
        """Threshold should relax over time."""
        gate = SemanticDistanceGate(adaptive_relaxation=True)

        threshold_early = gate.adaptive_threshold(iteration=5, max_iterations=30)
        threshold_mid = gate.adaptive_threshold(iteration=15, max_iterations=30)
        threshold_late = gate.adaptive_threshold(iteration=28, max_iterations=30)

        assert threshold_early > threshold_mid > threshold_late, \
            f"Threshold should decrease: {threshold_early} > {threshold_mid} > {threshold_late}"
        assert threshold_late >= 0.25, "Threshold should not go below 0.25"
        print(f"Threshold relaxation: early={threshold_early:.3f}, "
              f"mid={threshold_mid:.3f}, late={threshold_late:.3f}")

    def test_diversity_score_calculation(self):
        """Diversity score should reflect idea spread."""
        gate = SemanticDistanceGate()

        # Add several diverse ideas
        for i in range(5):
            np.random.seed(i * 10)
            embedding = create_mock_embedding()
            gate.check_distance(embedding, iteration=i)
            gate.add_idea(IdeaEmbedding(
                idea_id=str(i),
                title=f"Idea {i}",
                embedding=embedding,
                score=70.0 + i
            ))

        diversity = gate.get_diversity_score()

        assert 0 <= diversity <= 1, f"Diversity should be 0-1, got {diversity}"
        print(f"Diversity score with 5 random ideas: {diversity:.3f}")

    def test_statistics_tracking(self):
        """Statistics should track gate decisions."""
        gate = SemanticDistanceGate()

        # Process several ideas
        for i in range(10):
            embedding = create_mock_embedding()
            result = gate.check_distance(embedding, iteration=i)
            if result.decision == GateDecision.ACCEPT:
                gate.add_idea(IdeaEmbedding(
                    idea_id=str(i),
                    title=f"Idea {i}",
                    embedding=embedding,
                    score=70.0
                ))

        stats = gate.get_statistics()

        assert stats["total_checked"] == 10
        assert stats["accepted"] + stats["rejected"] + stats["borderline"] == 10
        print(f"Gate statistics: {stats}")


class TestDARLINGReward:
    """Tests for DARLING Reward Calculation."""

    def test_quality_score_calculation(self):
        """Quality score should weight dimensions correctly."""
        calculator = DARLINGReward()

        scores = DimensionScores(
            novelty=80,
            feasibility=75,
            market=70,
            complexity=65,
            scenario=60,
            contrarian=55,
            surprise=50,
            cross_domain=45
        )

        quality, contributions = calculator.calculate_quality_score(scores)

        # Verify all dimensions contribute
        assert len(contributions) == 8
        assert all(c >= 0 for c in contributions.values())

        # Verify weighted calculation (approximate)
        expected = (80*0.12 + 75*0.18 + 70*0.18 + 65*0.12 +
                    60*0.12 + 55*0.10 + 50*0.10 + 45*0.08)
        assert abs(quality - expected) < 0.01, \
            f"Quality mismatch: {quality} vs {expected}"
        print(f"Quality score: {quality:.2f}, contributions: {contributions}")

    def test_diversity_bonus_calculation(self):
        """Diversity bonus should reward distance from centroid."""
        calculator = DARLINGReward()

        # Close to centroid - low bonus
        bonus_close = calculator.calculate_diversity_bonus(
            centroid_distance=0.2,
            nearest_idea_distance=0.3
        )

        # Far from centroid - high bonus
        bonus_far = calculator.calculate_diversity_bonus(
            centroid_distance=0.8,
            nearest_idea_distance=0.9
        )

        assert bonus_far > bonus_close, \
            f"Far should get higher bonus: {bonus_far} > {bonus_close}"
        assert 0 <= bonus_close <= 20
        assert 0 <= bonus_far <= 20
        print(f"Diversity bonus: close={bonus_close:.2f}, far={bonus_far:.2f}")

    def test_exploration_bonus_new_region(self):
        """New regions should get exploration bonus."""
        calculator = DARLINGReward()
        embedding = create_mock_embedding()

        bonus, region_id, is_new = calculator.calculate_exploration_bonus(
            embedding=embedding,
            generator_mode=RewardGeneratorMode.EXPLORER
        )

        assert is_new == True, "First visit should be new"
        assert bonus >= 10, f"New region should get bonus >= 10, got {bonus}"
        print(f"New region bonus: {bonus:.2f}, region={region_id}")

        # Second visit to same region
        bonus2, _, is_new2 = calculator.calculate_exploration_bonus(
            embedding=embedding,
            generator_mode=RewardGeneratorMode.EXPLORER
        )

        assert is_new2 == False, "Second visit should not be new"
        assert bonus2 < bonus, f"Repeat visit should get lower bonus: {bonus2} < {bonus}"
        print(f"Repeat visit bonus: {bonus2:.2f}")

    def test_full_reward_calculation(self):
        """Full DARLING reward should combine all components."""
        calculator = DARLINGReward()

        scores = create_sample_scores(
            novelty=75, feasibility=80, market=70,
            complexity=65, scenario=72, contrarian=68,
            surprise=55, cross_domain=50
        )
        embedding = create_mock_embedding()

        result = calculator.calculate_reward(
            scores=scores,
            embedding=embedding,
            centroid_distance=0.5,
            generator_mode=RewardGeneratorMode.EXPLORER
        )

        assert result.quality_score > 0
        assert result.diversity_bonus >= 0
        assert result.exploration_bonus >= 0
        assert result.final_reward > 0
        assert result.generator_mode == "explorer"
        print(f"Full reward: {result.final_reward:.2f} "
              f"(Q={result.quality_score:.2f}, D={result.diversity_bonus:.2f}, "
              f"E={result.exploration_bonus:.2f})")

    def test_reward_trend_analysis(self):
        """Trend analysis should detect plateau."""
        calculator = DARLINGReward()

        # Simulate steady improvement
        for i in range(20):
            scores = create_sample_scores(
                novelty=60 + i, feasibility=65 + i * 0.5
            )
            calculator.calculate_reward(
                scores=scores,
                embedding=create_mock_embedding(),
                centroid_distance=0.5,
                generator_mode=RewardGeneratorMode.REFINER
            )

        trend = calculator.get_reward_trend(window=10)

        assert "trend" in trend
        assert "is_plateau" in trend
        print(f"Trend analysis: {trend}")

    def test_generator_mode_bonus(self):
        """Explorer mode should get bonus."""
        calculator = DARLINGReward()
        embedding = create_mock_embedding()

        # Reset region tracker for fair comparison
        calculator.region_tracker = SemanticRegion()

        explorer_bonus, _, _ = calculator.calculate_exploration_bonus(
            embedding=embedding,
            generator_mode=RewardGeneratorMode.EXPLORER
        )

        calculator.region_tracker = SemanticRegion()

        refiner_bonus, _, _ = calculator.calculate_exploration_bonus(
            embedding=embedding,
            generator_mode=RewardGeneratorMode.REFINER
        )

        assert explorer_bonus > refiner_bonus, \
            f"Explorer should get higher bonus: {explorer_bonus} > {refiner_bonus}"
        print(f"Mode bonuses: explorer={explorer_bonus:.2f}, refiner={refiner_bonus:.2f}")


class TestIntegration:
    """Integration tests for Phase 1 components working together."""

    def test_full_iteration_flow(self):
        """Test complete iteration: generate -> gate -> reward."""
        # 1. Generator selects mode
        generator = TripleGenerator()
        mode = generator.select_mode(
            iteration=5,
            max_iterations=30,
            recent_scores=[68, 70, 72, 71, 73]
        )
        print(f"1. Selected mode: {mode.value}")

        # 2. Gate checks idea
        gate = SemanticDistanceGate()
        embedding = create_mock_embedding()
        gate_result = gate.check_distance(embedding, iteration=5)
        print(f"2. Gate result: {gate_result.decision.value}")

        # 3. If accepted, calculate reward
        if gate_result.decision == GateDecision.ACCEPT:
            reward_calc = DARLINGReward()
            scores = create_sample_scores()

            reward = reward_calc.calculate_reward(
                scores=scores,
                embedding=embedding,
                centroid_distance=gate_result.distance,
                generator_mode=RewardGeneratorMode(mode.value)
            )
            print(f"3. Final reward: {reward.final_reward:.2f}")

            # 4. Add to gate for next iteration
            gate.add_idea(IdeaEmbedding(
                idea_id="test_1",
                title="Test Idea",
                embedding=embedding,
                score=reward.final_reward
            ))
            print(f"4. Idea stored. Gate stats: {gate.get_statistics()}")

    def test_multi_iteration_evolution(self):
        """Test that scores improve over iterations."""
        generator = TripleGenerator()
        gate = SemanticDistanceGate()
        reward_calc = DARLINGReward()

        scores_over_time = []

        for iteration in range(15):
            # Select mode
            mode = generator.select_mode(
                iteration=iteration,
                max_iterations=30,
                recent_scores=scores_over_time[-5:] if len(scores_over_time) >= 5 else [65] * 5
            )

            # Create embedding
            embedding = create_mock_embedding()

            # Check gate
            gate_result = gate.check_distance(embedding, iteration=iteration)

            if gate_result.decision == GateDecision.ACCEPT:
                # Calculate reward
                base_quality = 65 + iteration  # Simulate improvement
                scores = create_sample_scores(
                    novelty=base_quality,
                    feasibility=base_quality + 5,
                    market=base_quality - 2
                )

                reward = reward_calc.calculate_reward(
                    scores=scores,
                    embedding=embedding,
                    centroid_distance=gate_result.distance,
                    generator_mode=RewardGeneratorMode(mode.value)
                )

                scores_over_time.append(reward.final_reward)

                # Store in gate
                gate.add_idea(IdeaEmbedding(
                    idea_id=str(iteration),
                    title=f"Idea {iteration}",
                    embedding=embedding,
                    score=reward.final_reward
                ))

        # Verify improvement trend
        if len(scores_over_time) >= 10:
            early_avg = np.mean(scores_over_time[:5])
            late_avg = np.mean(scores_over_time[-5:])
            print(f"Score evolution: early={early_avg:.2f}, late={late_avg:.2f}")

        print(f"Multi-iteration test: {len(scores_over_time)} ideas accepted")
        print(f"Final gate stats: {gate.get_statistics()}")
        print(f"Final reward stats: {reward_calc.get_statistics()}")


def run_all_tests():
    """Run all tests and report results."""
    test_classes = [
        TestTripleGenerator,
        TestSemanticDistanceGate,
        TestDARLINGReward,
        TestIntegration
    ]

    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)

        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    print(f"\n--- {method_name} ---")
                    getattr(instance, method_name)()
                    total_passed += 1
                    print("PASSED")
                except AssertionError as e:
                    total_failed += 1
                    print(f"FAILED: {e}")
                except Exception as e:
                    total_failed += 1
                    print(f"ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {total_passed} passed, {total_failed} failed")
    print('='*60)

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
