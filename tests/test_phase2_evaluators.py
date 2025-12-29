"""
Integration Tests for Universal Ideation v3 - Phase 2 Evaluators

Tests cover:
1. Cognitive Diversity Evaluators (4 personas)
2. Structured Debate System
3. Evaluator Panel aggregation
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluators.cognitive_diversity import (
    EvaluatorPersona,
    DimensionWeights,
    EvaluationResult,
    ConservativeEvaluator,
    RadicalEvaluator,
    ShortTermEvaluator,
    LongTermEvaluator,
    EvaluatorPanel,
    create_evaluation_result
)
from evaluators.structured_debate import (
    StructuredDebate,
    DebatePosition,
    DebateChallenge,
    DebateSynthesis,
    QuickDebate,
    parse_propose_response,
    parse_challenge_response,
    parse_synthesis_response
)


class TestDimensionWeights:
    """Tests for dimension weight configuration."""

    def test_default_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        weights = DimensionWeights()
        total = sum(weights.as_dict().values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, not 1.0"
        print("Default weights validation: PASSED")

    def test_custom_weights_validation(self):
        """Custom weights should be validatable."""
        valid_weights = DimensionWeights(
            novelty=0.20, feasibility=0.20, market=0.20,
            complexity=0.10, scenario=0.10, contrarian=0.10,
            surprise=0.05, cross_domain=0.05
        )
        assert valid_weights.validate(), "Valid weights should pass validation"

        invalid_weights = DimensionWeights(
            novelty=0.50, feasibility=0.50  # All others default, sum > 1
        )
        # This will fail validation
        total = sum(invalid_weights.as_dict().values())
        print(f"Invalid weights total: {total:.2f} (should fail validation)")


class TestConservativeEvaluator:
    """Tests for Conservative Evaluator."""

    def test_weight_overrides(self):
        """Conservative should prioritize feasibility over novelty."""
        evaluator = ConservativeEvaluator()
        weights = evaluator.config.weight_overrides.as_dict()

        assert weights['feasibility'] > weights['novelty'], \
            "Conservative should weight feasibility > novelty"
        assert weights['feasibility'] == 0.30, \
            f"Feasibility should be 30%, got {weights['feasibility']}"
        assert weights['novelty'] == 0.05, \
            f"Novelty should be 5%, got {weights['novelty']}"
        print(f"Conservative weights: feasibility={weights['feasibility']}, novelty={weights['novelty']}")

    def test_scoring_biases(self):
        """Conservative should penalize novelty."""
        evaluator = ConservativeEvaluator()
        biases = evaluator.config.scoring_biases

        assert biases.get('novelty', 0) < 0, \
            "Conservative should have negative novelty bias"
        assert biases.get('feasibility', 0) > 0, \
            "Conservative should have positive feasibility bias"
        print(f"Conservative biases: {biases}")

    def test_weighted_score_calculation(self):
        """Weighted score should apply biases."""
        evaluator = ConservativeEvaluator()

        # High novelty, low feasibility idea
        radical_idea_scores = {
            'novelty': 90, 'feasibility': 50, 'market': 60,
            'complexity': 70, 'scenario': 60, 'contrarian': 80,
            'surprise': 85, 'cross_domain': 75
        }

        # Low novelty, high feasibility idea
        safe_idea_scores = {
            'novelty': 40, 'feasibility': 90, 'market': 80,
            'complexity': 50, 'scenario': 70, 'contrarian': 30,
            'surprise': 25, 'cross_domain': 40
        }

        radical_score = evaluator.calculate_weighted_score(radical_idea_scores)
        safe_score = evaluator.calculate_weighted_score(safe_idea_scores)

        assert safe_score > radical_score, \
            f"Conservative should prefer safe idea: {safe_score:.1f} > {radical_score:.1f}"
        print(f"Conservative scoring: safe={safe_score:.1f}, radical={radical_score:.1f}")

    def test_prompt_generation(self):
        """Should generate valid evaluation prompt."""
        evaluator = ConservativeEvaluator()

        idea = {"title": "Test Idea", "description": "A test concept"}
        invocation = evaluator.get_agent_invocation(idea, "protein beverages")

        assert invocation['agent'] == "business-advisor"
        assert "RISK-AVERSE" in invocation['prompt']
        assert "CONSERVATIVE" in invocation['prompt']
        assert "Test Idea" in invocation['prompt']
        print("Conservative prompt generation: PASSED")


class TestRadicalEvaluator:
    """Tests for Radical Evaluator."""

    def test_weight_overrides(self):
        """Radical should prioritize novelty over feasibility."""
        evaluator = RadicalEvaluator()
        weights = evaluator.config.weight_overrides.as_dict()

        assert weights['novelty'] > weights['feasibility'], \
            "Radical should weight novelty > feasibility"
        assert weights['novelty'] == 0.25
        assert weights['contrarian'] == 0.20
        print(f"Radical weights: novelty={weights['novelty']}, contrarian={weights['contrarian']}")

    def test_scoring_biases(self):
        """Radical should reward novelty."""
        evaluator = RadicalEvaluator()
        biases = evaluator.config.scoring_biases

        assert biases.get('novelty', 0) > 0, \
            "Radical should have positive novelty bias"
        assert biases.get('contrarian', 0) > 0, \
            "Radical should have positive contrarian bias"
        print(f"Radical biases: {biases}")

    def test_opposite_preference_to_conservative(self):
        """Radical and Conservative should have opposite preferences."""
        conservative = ConservativeEvaluator()
        radical = RadicalEvaluator()

        # Disruptive idea
        disruptive_scores = {
            'novelty': 95, 'feasibility': 40, 'market': 50,
            'complexity': 60, 'scenario': 45, 'contrarian': 90,
            'surprise': 88, 'cross_domain': 80
        }

        conservative_score = conservative.calculate_weighted_score(disruptive_scores)
        radical_score = radical.calculate_weighted_score(disruptive_scores)

        assert radical_score > conservative_score, \
            f"Radical should rate disruptive higher: {radical_score:.1f} > {conservative_score:.1f}"
        assert radical_score - conservative_score > 10, \
            "Difference should be significant (>10 points)"
        print(f"Disruptive idea: radical={radical_score:.1f}, conservative={conservative_score:.1f}")


class TestShortTermEvaluator:
    """Tests for Short-Term Evaluator."""

    def test_weight_overrides(self):
        """Short-term should heavily weight feasibility."""
        evaluator = ShortTermEvaluator()
        weights = evaluator.config.weight_overrides.as_dict()

        assert weights['feasibility'] == 0.35, \
            f"Feasibility should be 35%, got {weights['feasibility']}"
        assert weights['scenario'] < 0.10, \
            "Short-term should not care about long-term scenarios"
        print(f"Short-term weights: feasibility={weights['feasibility']}, scenario={weights['scenario']}")

    def test_quick_win_preference(self):
        """Short-term should prefer quick wins."""
        evaluator = ShortTermEvaluator()

        # Quick win (high feasibility, immediate market)
        quick_win = {
            'novelty': 50, 'feasibility': 95, 'market': 85,
            'complexity': 30, 'scenario': 40, 'contrarian': 40,
            'surprise': 35, 'cross_domain': 45
        }

        # Long build (high complexity, long-term scenario)
        long_build = {
            'novelty': 70, 'feasibility': 50, 'market': 60,
            'complexity': 90, 'scenario': 85, 'contrarian': 60,
            'surprise': 55, 'cross_domain': 70
        }

        quick_score = evaluator.calculate_weighted_score(quick_win)
        long_score = evaluator.calculate_weighted_score(long_build)

        assert quick_score > long_score, \
            f"Short-term should prefer quick win: {quick_score:.1f} > {long_score:.1f}"
        print(f"Short-term preference: quick={quick_score:.1f}, long={long_score:.1f}")


class TestLongTermEvaluator:
    """Tests for Long-Term Evaluator."""

    def test_weight_overrides(self):
        """Long-term should heavily weight scenario and complexity."""
        evaluator = LongTermEvaluator()
        weights = evaluator.config.weight_overrides.as_dict()

        assert weights['scenario'] == 0.25, \
            f"Scenario should be 25%, got {weights['scenario']}"
        assert weights['complexity'] == 0.20, \
            f"Complexity should be 20%, got {weights['complexity']}"
        assert weights['feasibility'] < 0.10, \
            "Long-term should not prioritize immediate feasibility"
        print(f"Long-term weights: scenario={weights['scenario']}, complexity={weights['complexity']}")

    def test_moat_preference(self):
        """Long-term should prefer ideas with moats."""
        evaluator = LongTermEvaluator()

        # Strong moat (network effects, scenario resilience)
        moat_idea = {
            'novelty': 60, 'feasibility': 50, 'market': 65,
            'complexity': 95, 'scenario': 90, 'contrarian': 70,
            'surprise': 55, 'cross_domain': 80
        }

        # No moat (easily copied)
        no_moat_idea = {
            'novelty': 70, 'feasibility': 90, 'market': 80,
            'complexity': 30, 'scenario': 40, 'contrarian': 50,
            'surprise': 45, 'cross_domain': 35
        }

        moat_score = evaluator.calculate_weighted_score(moat_idea)
        no_moat_score = evaluator.calculate_weighted_score(no_moat_idea)

        assert moat_score > no_moat_score, \
            f"Long-term should prefer moat: {moat_score:.1f} > {no_moat_score:.1f}"
        print(f"Long-term moat preference: moat={moat_score:.1f}, no_moat={no_moat_score:.1f}")

    def test_opposite_to_short_term(self):
        """Long-term and Short-term should have opposite preferences."""
        short_term = ShortTermEvaluator()
        long_term = LongTermEvaluator()

        # Complex, slow-build idea with strong moat
        slow_build = {
            'novelty': 65, 'feasibility': 45, 'market': 55,
            'complexity': 92, 'scenario': 88, 'contrarian': 60,
            'surprise': 50, 'cross_domain': 75
        }

        short_score = short_term.calculate_weighted_score(slow_build)
        long_score = long_term.calculate_weighted_score(slow_build)

        assert long_score > short_score, \
            f"Long-term should prefer slow-build: {long_score:.1f} > {short_score:.1f}"
        print(f"Time horizon difference: long={long_score:.1f}, short={short_score:.1f}")


class TestEvaluatorPanel:
    """Tests for the Evaluator Panel."""

    def test_panel_initialization(self):
        """Panel should have all 4 evaluators."""
        panel = EvaluatorPanel()

        assert len(panel.evaluators) == 4
        assert EvaluatorPersona.CONSERVATIVE in panel.evaluators
        assert EvaluatorPersona.RADICAL in panel.evaluators
        assert EvaluatorPersona.SHORT_TERM in panel.evaluators
        assert EvaluatorPersona.LONG_TERM in panel.evaluators
        print("Panel initialization: 4 evaluators present")

    def test_get_all_prompts(self):
        """Should generate prompts for all 4 evaluators."""
        panel = EvaluatorPanel()

        idea = {"title": "Test Idea", "description": "A test concept"}
        prompts = panel.get_all_prompts(idea, "protein beverages")

        assert len(prompts) == 4
        personas_covered = {p['persona'] for p in prompts}
        assert len(personas_covered) == 4
        print(f"Generated {len(prompts)} prompts for personas: {personas_covered}")

    def test_aggregate_scores(self):
        """Should aggregate scores and detect disagreement."""
        panel = EvaluatorPanel()

        # Simulate evaluation results with disagreement
        results = {
            EvaluatorPersona.CONSERVATIVE: create_evaluation_result(
                persona=EvaluatorPersona.CONSERVATIVE,
                dimension_scores={
                    'novelty': 40, 'feasibility': 85, 'market': 75,
                    'complexity': 50, 'scenario': 60, 'contrarian': 35,
                    'surprise': 30, 'cross_domain': 40
                },
                reasoning="Too risky, unproven approach",
                recommendation="reject"
            ),
            EvaluatorPersona.RADICAL: create_evaluation_result(
                persona=EvaluatorPersona.RADICAL,
                dimension_scores={
                    'novelty': 90, 'feasibility': 55, 'market': 60,
                    'complexity': 70, 'scenario': 50, 'contrarian': 85,
                    'surprise': 80, 'cross_domain': 75
                },
                reasoning="Exciting disruption potential",
                recommendation="strong_accept"
            ),
            EvaluatorPersona.SHORT_TERM: create_evaluation_result(
                persona=EvaluatorPersona.SHORT_TERM,
                dimension_scores={
                    'novelty': 60, 'feasibility': 70, 'market': 75,
                    'complexity': 45, 'scenario': 55, 'contrarian': 50,
                    'surprise': 45, 'cross_domain': 50
                },
                reasoning="Could launch quickly with iteration",
                recommendation="accept"
            ),
            EvaluatorPersona.LONG_TERM: create_evaluation_result(
                persona=EvaluatorPersona.LONG_TERM,
                dimension_scores={
                    'novelty': 65, 'feasibility': 60, 'market': 65,
                    'complexity': 80, 'scenario': 75, 'contrarian': 60,
                    'surprise': 55, 'cross_domain': 70
                },
                reasoning="Good defensibility if executed well",
                recommendation="accept"
            )
        }

        aggregation = panel.aggregate_scores(results)

        assert 'consensus_score' in aggregation
        assert 'disagreement_level' in aggregation
        assert 'needs_debate' in aggregation
        assert aggregation['disagreement_level'] > 0
        print(f"Aggregation: consensus={aggregation['consensus_score']:.1f}, "
              f"disagreement={aggregation['disagreement_level']:.1f}, "
              f"needs_debate={aggregation['needs_debate']}")

    def test_detect_debate_triggers(self):
        """Should detect when debate is needed."""
        panel = EvaluatorPanel()

        # High disagreement results
        high_disagreement_results = {
            EvaluatorPersona.CONSERVATIVE: create_evaluation_result(
                persona=EvaluatorPersona.CONSERVATIVE,
                dimension_scores={
                    'novelty': 30, 'feasibility': 90, 'market': 80,
                    'complexity': 40, 'scenario': 70, 'contrarian': 25,
                    'surprise': 20, 'cross_domain': 35
                },
                reasoning="Too radical",
                recommendation="strong_reject"
            ),
            EvaluatorPersona.RADICAL: create_evaluation_result(
                persona=EvaluatorPersona.RADICAL,
                dimension_scores={
                    'novelty': 95, 'feasibility': 40, 'market': 50,
                    'complexity': 75, 'scenario': 45, 'contrarian': 95,
                    'surprise': 90, 'cross_domain': 85
                },
                reasoning="Revolutionary!",
                recommendation="strong_accept"
            )
        }

        triggers = panel.detect_debate_triggers(high_disagreement_results)

        assert len(triggers) > 0, "Should detect debate triggers"
        print(f"Debate triggers detected: {triggers}")


class TestStructuredDebate:
    """Tests for Structured Debate System."""

    def test_should_debate_detection(self):
        """Should detect when debate is needed."""
        debate = StructuredDebate()

        # High disagreement
        high_disagree = {
            EvaluatorPersona.CONSERVATIVE: create_evaluation_result(
                persona=EvaluatorPersona.CONSERVATIVE,
                dimension_scores={
                    'novelty': 30, 'feasibility': 85, 'market': 75,
                    'complexity': 40, 'scenario': 65, 'contrarian': 25,
                    'surprise': 20, 'cross_domain': 35
                },
                reasoning="Risky",
                recommendation="reject"
            ),
            EvaluatorPersona.RADICAL: create_evaluation_result(
                persona=EvaluatorPersona.RADICAL,
                dimension_scores={
                    'novelty': 95, 'feasibility': 45, 'market': 55,
                    'complexity': 80, 'scenario': 50, 'contrarian': 92,
                    'surprise': 88, 'cross_domain': 82
                },
                reasoning="Brilliant!",
                recommendation="strong_accept"
            )
        }

        should, reason = debate.should_debate(high_disagree)
        assert should, "Should trigger debate for high disagreement"
        assert reason is not None
        print(f"Debate triggered: {reason}")

        # Low disagreement
        low_disagree = {
            EvaluatorPersona.CONSERVATIVE: create_evaluation_result(
                persona=EvaluatorPersona.CONSERVATIVE,
                dimension_scores={
                    'novelty': 65, 'feasibility': 70, 'market': 72,
                    'complexity': 60, 'scenario': 68, 'contrarian': 55,
                    'surprise': 50, 'cross_domain': 55
                },
                reasoning="Acceptable",
                recommendation="accept"
            ),
            EvaluatorPersona.RADICAL: create_evaluation_result(
                persona=EvaluatorPersona.RADICAL,
                dimension_scores={
                    'novelty': 70, 'feasibility': 65, 'market': 68,
                    'complexity': 65, 'scenario': 60, 'contrarian': 62,
                    'surprise': 58, 'cross_domain': 60
                },
                reasoning="OK idea",
                recommendation="accept"
            )
        }

        should, reason = debate.should_debate(low_disagree)
        assert not should, "Should not trigger debate for low disagreement"
        print("Low disagreement correctly not triggering debate")

    def test_identify_debaters(self):
        """Should identify highest and lowest scorers."""
        debate = StructuredDebate()

        results = {
            EvaluatorPersona.CONSERVATIVE: create_evaluation_result(
                persona=EvaluatorPersona.CONSERVATIVE,
                dimension_scores={
                    'novelty': 40, 'feasibility': 80, 'market': 70,
                    'complexity': 50, 'scenario': 60, 'contrarian': 35,
                    'surprise': 30, 'cross_domain': 40
                },
                reasoning="Low",
                recommendation="reject"
            ),
            EvaluatorPersona.RADICAL: create_evaluation_result(
                persona=EvaluatorPersona.RADICAL,
                dimension_scores={
                    'novelty': 90, 'feasibility': 50, 'market': 60,
                    'complexity': 70, 'scenario': 50, 'contrarian': 85,
                    'surprise': 80, 'cross_domain': 75
                },
                reasoning="High",
                recommendation="strong_accept"
            )
        }

        high, low = debate.identify_debaters(results)

        assert high.weighted_score > low.weighted_score
        print(f"Debaters: high={high.persona.value} ({high.weighted_score:.1f}), "
              f"low={low.persona.value} ({low.weighted_score:.1f})")

    def test_propose_prompt_generation(self):
        """Should generate valid PROPOSE phase prompts."""
        debate = StructuredDebate()

        idea = {"title": "Disruptive Idea", "description": "Something new"}
        evaluation = create_evaluation_result(
            persona=EvaluatorPersona.RADICAL,
            dimension_scores={
                'novelty': 90, 'feasibility': 50, 'market': 60,
                'complexity': 70, 'scenario': 50, 'contrarian': 85,
                'surprise': 80, 'cross_domain': 75
            },
            reasoning="Exciting!",
            recommendation="strong_accept"
        )

        prompt = debate.build_propose_prompt(idea, evaluation, "protein beverages")

        assert "PROPOSE PHASE" in prompt
        assert "RADICAL" in prompt
        assert "SUPPORT" in prompt  # High score = support
        assert "Disruptive Idea" in prompt
        print("PROPOSE prompt generation: PASSED")

    def test_synthesis_prompt_generation(self):
        """Should generate valid SYNTHESIZE phase prompt."""
        debate = StructuredDebate()

        idea = {"title": "Contested Idea"}
        positions = [
            DebatePosition(
                persona=EvaluatorPersona.RADICAL,
                stance="support",
                score=85.0,
                key_arguments=["Breaks paradigm", "New market"],
                assumptions=["Market ready for disruption"],
                evidence=["Trend analysis"]
            ),
            DebatePosition(
                persona=EvaluatorPersona.CONSERVATIVE,
                stance="oppose",
                score=55.0,
                key_arguments=["Unproven", "Too risky"],
                assumptions=["Market prefers incremental"],
                evidence=["Past failures"]
            )
        ]
        challenges = [
            DebateChallenge(
                challenger=EvaluatorPersona.RADICAL,
                target=EvaluatorPersona.CONSERVATIVE,
                challenged_assumptions=["Market is not static"],
                counter_arguments=["Past is not future"],
                alternative_interpretation="Risk = opportunity"
            )
        ]

        prompt = debate.build_synthesis_prompt(idea, positions, challenges, "test")

        assert "SYNTHESIZE" in prompt
        assert "FIRST-PRINCIPLES" in prompt
        assert "RADICAL" in prompt
        assert "CONSERVATIVE" in prompt
        print("SYNTHESIZE prompt generation: PASSED")


class TestResponseParsing:
    """Tests for parsing debate responses."""

    def test_parse_propose_response(self):
        """Should parse valid PROPOSE response."""
        response = '''
{
    "stance": "support",
    "conviction": 8,
    "key_arguments": ["Arg1", "Arg2", "Arg3"],
    "assumptions": ["Assumption1"],
    "evidence": ["Evidence1"]
}
'''
        position = parse_propose_response(
            response,
            EvaluatorPersona.RADICAL,
            85.0
        )

        assert position.stance == "support"
        assert len(position.key_arguments) == 3
        assert position.persona == EvaluatorPersona.RADICAL
        print("PROPOSE parsing: PASSED")

    def test_parse_challenge_response(self):
        """Should parse valid CHALLENGE response."""
        response = '''
{
    "challenged_assumptions": ["They assume X"],
    "counter_arguments": ["Counter 1", "Counter 2"],
    "what_they_miss": ["Missing Y"],
    "alternative_interpretation": "Actually..."
}
'''
        challenge = parse_challenge_response(
            response,
            EvaluatorPersona.RADICAL,
            EvaluatorPersona.CONSERVATIVE
        )

        assert len(challenge.challenged_assumptions) == 1
        assert len(challenge.counter_arguments) == 2
        assert challenge.challenger == EvaluatorPersona.RADICAL
        print("CHALLENGE parsing: PASSED")

    def test_parse_synthesis_response(self):
        """Should parse valid SYNTHESIZE response."""
        response = '''
{
    "revised_score": 72.5,
    "confidence": 0.85,
    "key_insights": ["Insight 1", "Insight 2"],
    "resolved_disagreements": ["Resolved X"],
    "unresolved_tensions": ["Still unclear: Y"],
    "final_recommendation": "accept",
    "reasoning": "Balanced view after debate"
}
'''
        synthesis = parse_synthesis_response(response)

        assert synthesis.revised_score == 72.5
        assert synthesis.confidence == 0.85
        assert synthesis.final_recommendation == "accept"
        assert len(synthesis.key_insights) == 2
        print("SYNTHESIZE parsing: PASSED")

    def test_parse_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        invalid_response = "This is not JSON"

        position = parse_propose_response(
            invalid_response,
            EvaluatorPersona.RADICAL,
            75.0
        )
        assert position.stance == 'neutral'  # Fallback

        synthesis = parse_synthesis_response(invalid_response)
        assert synthesis.revised_score == 70.0  # Fallback

        print("Invalid JSON handling: PASSED")


class TestQuickDebate:
    """Tests for Quick Debate (simplified synthesis)."""

    def test_quick_synthesis_prompt(self):
        """Should generate valid quick synthesis prompt."""
        quick = QuickDebate()

        idea = {"title": "Quick Test Idea"}
        results = {
            EvaluatorPersona.CONSERVATIVE: create_evaluation_result(
                persona=EvaluatorPersona.CONSERVATIVE,
                dimension_scores={
                    'novelty': 40, 'feasibility': 80, 'market': 70,
                    'complexity': 50, 'scenario': 60, 'contrarian': 35,
                    'surprise': 30, 'cross_domain': 40
                },
                reasoning="Too risky",
                recommendation="reject"
            ),
            EvaluatorPersona.RADICAL: create_evaluation_result(
                persona=EvaluatorPersona.RADICAL,
                dimension_scores={
                    'novelty': 85, 'feasibility': 55, 'market': 60,
                    'complexity': 70, 'scenario': 50, 'contrarian': 80,
                    'surprise': 75, 'cross_domain': 70
                },
                reasoning="Interesting!",
                recommendation="accept"
            )
        }

        prompt_config = quick.build_quick_synthesis_prompt(idea, results, "test")

        assert prompt_config['agent'] == "first-principles-analyst"
        assert "synthesizing" in prompt_config['prompt'].lower()
        assert "CONSERVATIVE" in prompt_config['prompt']
        assert "RADICAL" in prompt_config['prompt']
        print("Quick synthesis prompt: PASSED")


def run_all_tests():
    """Run all tests and report results."""
    test_classes = [
        TestDimensionWeights,
        TestConservativeEvaluator,
        TestRadicalEvaluator,
        TestShortTermEvaluator,
        TestLongTermEvaluator,
        TestEvaluatorPanel,
        TestStructuredDebate,
        TestResponseParsing,
        TestQuickDebate
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
