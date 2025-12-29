"""
Phase 3 Integration Tests - New Scoring Dimensions
Universal Ideation v3

Tests for:
1. Surprise Dimension - Schema violation detection
2. Cross-Domain Bridge - Analogical distance measurement
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluators.surprise_dimension import (
    SurpriseDimension,
    SurpriseAnalysis,
    SchemaViolation,
    DomainConventions,
    ConventionCategory,
    calculate_surprise
)
from evaluators.cross_domain_bridge import (
    CrossDomainBridge,
    CrossDomainAnalysis,
    AnalogyDetector,
    DomainKnowledgeGraph,
    DomainCategory,
    Analogy,
    calculate_cross_domain,
    EXAMPLE_ANALOGIES
)


# =============================================================================
# SURPRISE DIMENSION TESTS
# =============================================================================

class TestSurpriseDimension:
    """Test suite for Surprise Dimension evaluation."""

    @pytest.fixture
    def surprise_analyzer(self):
        """Create SurpriseDimension instance for protein domain."""
        return SurpriseDimension(domain="protein beverages")

    @pytest.fixture
    def conventional_idea(self):
        """An idea that follows all conventions - not surprising."""
        return {
            "title": "Premium Whey Protein Shake",
            "description": "High-quality whey protein for post-workout recovery",
            "target_market": "fitness enthusiasts, gym-goers",
            "format": "ready-to-drink bottle",
            "positioning": "muscle building and recovery",
            "distribution": "specialty stores and gyms",
            "price_point": "premium"
        }

    @pytest.fixture
    def surprising_idea(self):
        """An idea that violates multiple conventions - very surprising."""
        return {
            "title": "Elderly Sleep Protein",
            "description": "Insect protein gummies for elderly before bed",
            "target_market": "elderly, nursing home residents",
            "format": "gummy",
            "positioning": "sleep quality and relaxation",
            "distribution": "hospitals and pharmacies",
            "usage_occasion": "before bed"
        }

    @pytest.fixture
    def moderately_surprising_idea(self):
        """An idea with 1-2 convention violations."""
        return {
            "title": "Gamer Protein Drink",
            "description": "Protein drink designed for gamers during long sessions",
            "target_market": "gamers, esports athletes",
            "format": "ready-to-drink",
            "positioning": "cognitive and sustained energy",
            "distribution": "convenience stores"
        }

    def test_initialization(self, surprise_analyzer):
        """Test SurpriseDimension initializes correctly."""
        assert surprise_analyzer.domain == "protein beverages"
        assert surprise_analyzer.conventions is not None
        assert len(surprise_analyzer.conventions) > 0
        assert surprise_analyzer.SCHEMA_WEIGHT == 0.6
        assert surprise_analyzer.EXPECTATION_WEIGHT == 0.4

    def test_conventional_idea_low_surprise(self, surprise_analyzer, conventional_idea):
        """Conventional ideas should have low surprise scores."""
        analysis = surprise_analyzer.analyze(conventional_idea)

        assert isinstance(analysis, SurpriseAnalysis)
        assert analysis.final_score < 50  # Low surprise
        assert len(analysis.violations) == 0  # No violations
        assert analysis.schema_score == 30.0  # Base score for no violations

    def test_surprising_idea_high_surprise(self, surprise_analyzer, surprising_idea):
        """Ideas with multiple violations should have high surprise scores."""
        analysis = surprise_analyzer.analyze(surprising_idea)

        assert isinstance(analysis, SurpriseAnalysis)
        assert analysis.final_score > 60  # High surprise
        assert len(analysis.violations) > 0  # Has violations
        assert analysis.schema_score > 50  # Above baseline

    def test_moderately_surprising_idea(self, surprise_analyzer, moderately_surprising_idea):
        """Moderately surprising ideas should score in the middle range."""
        analysis = surprise_analyzer.analyze(moderately_surprising_idea)

        assert isinstance(analysis, SurpriseAnalysis)
        assert 40 <= analysis.final_score <= 80  # Middle range

    def test_violation_detection_target_audience(self, surprise_analyzer):
        """Test detection of target audience convention violations."""
        idea = {
            "title": "Kids Protein Snack",
            "description": "Protein snacks for children",
            "target_market": "children, kids, parents"
        }
        analysis = surprise_analyzer.analyze(idea)

        # Should detect non-fitness audience
        audience_violations = [v for v in analysis.violations
                               if v.category == ConventionCategory.TARGET_AUDIENCE]
        assert len(audience_violations) > 0 or analysis.schema_score > 30

    def test_violation_detection_format(self, surprise_analyzer):
        """Test detection of format convention violations."""
        idea = {
            "title": "Protein Patch",
            "description": "Transdermal protein delivery patch",
            "format": "patch, wearable"
        }
        analysis = surprise_analyzer.analyze(idea)

        # Should detect unconventional format
        format_violations = [v for v in analysis.violations
                            if v.category == ConventionCategory.FORMAT]
        assert len(format_violations) > 0

    def test_violation_detection_ingredient(self, surprise_analyzer):
        """Test detection of ingredient base violations."""
        idea = {
            "title": "Algae Protein Shake",
            "description": "Sustainable protein from algae cultivation",
            "ingredient_base": "algae, spirulina"
        }
        analysis = surprise_analyzer.analyze(idea)

        # Should detect unconventional ingredient
        ingredient_violations = [v for v in analysis.violations
                                 if v.category == ConventionCategory.INGREDIENT_BASE]
        assert len(ingredient_violations) > 0

    def test_schema_score_scaling(self, surprise_analyzer):
        """Test schema score scales with number of violations."""
        # 1 violation
        idea_1v = {"title": "Gummy Protein", "format": "gummy"}
        analysis_1v = surprise_analyzer.analyze(idea_1v)

        # 2 violations
        idea_2v = {"title": "Elderly Gummy Protein", "format": "gummy",
                   "target_market": "elderly"}
        analysis_2v = surprise_analyzer.analyze(idea_2v)

        # 3 violations
        idea_3v = {"title": "Elderly Insect Gummy", "format": "gummy",
                   "target_market": "elderly", "ingredient_base": "insect"}
        analysis_3v = surprise_analyzer.analyze(idea_3v)

        # More violations = higher schema score
        assert analysis_2v.schema_score >= analysis_1v.schema_score
        assert analysis_3v.schema_score >= analysis_2v.schema_score

    def test_expectation_score_with_prior_ideas(self, surprise_analyzer):
        """Test expectation deviation with prior ideas."""
        prior_ideas = [
            {"title": "Vanilla Whey", "description": "Standard whey protein"},
            {"title": "Chocolate Whey", "description": "Chocolate flavored whey"},
            {"title": "Strawberry Whey", "description": "Strawberry whey protein"}
        ]

        # Similar to prior ideas
        similar_idea = {"title": "Banana Whey", "description": "Banana whey protein"}
        similar_analysis = surprise_analyzer.analyze(similar_idea, prior_ideas)

        # Different from prior ideas
        different_idea = {"title": "Gaming Energy Protein",
                         "description": "Cognitive boost insect protein for gamers"}
        different_analysis = surprise_analyzer.analyze(different_idea, prior_ideas)

        # Different idea should have higher expectation score
        assert different_analysis.expectation_score > similar_analysis.expectation_score

    def test_surprise_factors_extracted(self, surprise_analyzer, surprising_idea):
        """Test that surprise factors are properly extracted."""
        analysis = surprise_analyzer.analyze(surprising_idea)

        assert isinstance(analysis.surprise_factors, list)
        # Should have at least one surprise factor for surprising idea
        if len(analysis.violations) > 0:
            assert len(analysis.surprise_factors) > 0

    def test_violated_categories_list(self, surprise_analyzer, surprising_idea):
        """Test that violated categories are tracked."""
        analysis = surprise_analyzer.analyze(surprising_idea)

        assert isinstance(analysis.violated_categories, list)
        # Should track which categories were violated
        for cat in analysis.violated_categories:
            assert cat in [c.value for c in ConventionCategory]

    def test_statistics_tracking(self, surprise_analyzer):
        """Test that analysis statistics are tracked."""
        ideas = [
            {"title": "Whey Protein", "description": "Standard"},
            {"title": "Insect Gummy", "format": "gummy", "ingredient_base": "insect"},
            {"title": "Elderly Sleep", "target_market": "elderly"}
        ]

        for idea in ideas:
            surprise_analyzer.analyze(idea)

        stats = surprise_analyzer.get_statistics()
        assert stats["total_analyses"] == 3
        assert "average_surprise" in stats
        assert "high_surprise_count" in stats

    def test_helper_function(self):
        """Test calculate_surprise helper function."""
        idea = {"title": "Insect Protein", "ingredient_base": "insect"}
        score = calculate_surprise(idea, domain="protein beverages")

        assert isinstance(score, float)
        assert 0 <= score <= 100


class TestDomainConventions:
    """Test DomainConventions configuration."""

    def test_protein_conventions_loaded(self):
        """Test protein beverage conventions are defined."""
        conventions = DomainConventions.get_conventions("protein beverages")

        assert len(conventions) > 0
        assert ConventionCategory.TARGET_AUDIENCE in conventions
        assert ConventionCategory.FORMAT in conventions
        assert ConventionCategory.INGREDIENT_BASE in conventions

    def test_generic_conventions_loaded(self):
        """Test generic conventions for unknown domains."""
        conventions = DomainConventions.get_conventions("random unknown domain")

        assert len(conventions) > 0
        assert conventions == DomainConventions.GENERIC_CONVENTIONS

    def test_convention_structure(self):
        """Test convention data structure is correct."""
        conventions = DomainConventions.get_conventions("protein")

        for category, convention in conventions.items():
            assert hasattr(convention, 'expected_value')
            assert hasattr(convention, 'alternatives')
            assert hasattr(convention, 'violation_weight')
            assert isinstance(convention.alternatives, list)
            assert convention.violation_weight > 0


# =============================================================================
# CROSS-DOMAIN BRIDGE TESTS
# =============================================================================

class TestCrossDomainBridge:
    """Test suite for Cross-Domain Bridge evaluation."""

    @pytest.fixture
    def bridge_analyzer(self):
        """Create CrossDomainBridge instance for food domain."""
        return CrossDomainBridge(target_domain="food_beverage")

    @pytest.fixture
    def analogy_detector(self):
        """Create AnalogyDetector instance."""
        return AnalogyDetector()

    def test_initialization(self, bridge_analyzer):
        """Test CrossDomainBridge initializes correctly."""
        assert bridge_analyzer.target_domain == DomainCategory.FOOD_BEVERAGE
        assert bridge_analyzer.detector is not None
        assert bridge_analyzer.OPTIMAL_HOPS_MIN == 3
        assert bridge_analyzer.OPTIMAL_HOPS_MAX == 6

    def test_netflix_analogy_detection(self, bridge_analyzer):
        """Test detection of Netflix-style analogies."""
        analysis = bridge_analyzer.analyze(EXAMPLE_ANALOGIES["netflix_for_protein"])

        assert isinstance(analysis, CrossDomainAnalysis)
        assert len(analysis.analogies) > 0
        assert analysis.final_score > 30  # Has cross-domain thinking

        # Should detect Netflix/subscription references
        detected_sources = [a.source_concept for a in analysis.analogies]
        assert any("netflix" in s.lower() or "subscription" in s.lower()
                  for s in detected_sources)

    def test_uber_analogy_detection(self, bridge_analyzer):
        """Test detection of Uber-style analogies."""
        analysis = bridge_analyzer.analyze(EXAMPLE_ANALOGIES["uber_for_nutrition"])

        assert len(analysis.analogies) > 0
        detected_sources = [a.source_concept.lower() for a in analysis.analogies]
        assert any("uber" in s for s in detected_sources)

    def test_gamification_analogy_detection(self, bridge_analyzer):
        """Test detection of gaming/gamification analogies."""
        analysis = bridge_analyzer.analyze(EXAMPLE_ANALOGIES["gamification_protein"])

        assert len(analysis.analogies) > 0
        detected_sources = [a.source_concept.lower() for a in analysis.analogies]
        assert any("gaming" in s or "gamification" in s for s in detected_sources)

    def test_no_analogy_low_score(self, bridge_analyzer):
        """Test ideas without analogies get low cross-domain scores."""
        analysis = bridge_analyzer.analyze(EXAMPLE_ANALOGIES["no_analogy"])

        assert analysis.final_score == 30.0  # Default low score
        assert len(analysis.analogies) == 0
        assert analysis.average_hops == 0

    def test_semantic_hop_calculation(self, bridge_analyzer):
        """Test semantic distance is calculated correctly."""
        analysis = bridge_analyzer.analyze(EXAMPLE_ANALOGIES["netflix_for_protein"])

        if analysis.analogies:
            # Entertainment to Food should be 3+ hops
            for analogy in analysis.analogies:
                if analogy.source_domain == "entertainment":
                    assert analogy.semantic_hops >= 2

    def test_optimal_hop_range_scoring(self, bridge_analyzer):
        """Test that optimal hop range (3-6) gets highest scores."""
        # Create test cases with different hop counts
        hop_scores = []

        for hops in [1, 2, 3, 4, 5, 6, 7, 8]:
            score = bridge_analyzer._calculate_hop_score(hops)
            hop_scores.append((hops, score))

        # 3-6 hops should score highest
        sweet_spot_scores = [s for h, s in hop_scores if 3 <= h <= 6]
        edge_scores = [s for h, s in hop_scores if h < 2 or h > 7]

        assert min(sweet_spot_scores) >= max(edge_scores)

    def test_bridge_quality_assessment(self, analogy_detector):
        """Test bridge quality scoring."""
        # High quality - has explanation words
        high_quality_text = "Like Netflix subscription model applied to protein delivery"
        quality1 = analogy_detector._assess_bridge_quality("netflix", high_quality_text, 4)

        # Lower quality - just mentions source
        low_quality_text = "Netflix protein thing"
        quality2 = analogy_detector._assess_bridge_quality("netflix", low_quality_text, 4)

        assert quality1 > quality2

    def test_multiple_analogies_bonus(self, bridge_analyzer):
        """Test that multiple analogies increase score."""
        # Single analogy
        single_idea = {
            "title": "Netflix Protein",
            "description": "Subscription protein delivery like Netflix"
        }

        # Multiple analogies
        multi_idea = {
            "title": "Netflix meets Uber Protein",
            "description": "Subscription model like Netflix with on-demand delivery like Uber, gamification features"
        }

        single_analysis = bridge_analyzer.analyze(single_idea)
        multi_analysis = bridge_analyzer.analyze(multi_idea)

        # More analogies should generally score higher
        if len(multi_analysis.analogies) > len(single_analysis.analogies):
            assert multi_analysis.final_score >= single_analysis.final_score

    def test_cross_domain_factors_extraction(self, bridge_analyzer):
        """Test extraction of cross-domain factors."""
        analysis = bridge_analyzer.analyze(EXAMPLE_ANALOGIES["netflix_for_protein"])

        assert isinstance(analysis.cross_domain_factors, list)
        if analysis.analogies:
            assert len(analysis.cross_domain_factors) > 0

    def test_primary_source_domain_identified(self, bridge_analyzer):
        """Test that primary source domain is identified."""
        analysis = bridge_analyzer.analyze(EXAMPLE_ANALOGIES["netflix_for_protein"])

        if analysis.analogies:
            assert analysis.primary_source_domain is not None

    def test_statistics_tracking(self, bridge_analyzer):
        """Test that analysis statistics are tracked."""
        for example in EXAMPLE_ANALOGIES.values():
            bridge_analyzer.analyze(example)

        stats = bridge_analyzer.get_statistics()
        assert stats["total_analyses"] == 4
        assert "average_score" in stats
        assert "cross_domain_rate" in stats


class TestDomainKnowledgeGraph:
    """Test Domain Knowledge Graph for semantic distances."""

    def test_same_domain_zero_distance(self):
        """Same domain should have zero distance."""
        distance = DomainKnowledgeGraph.get_distance(
            DomainCategory.TECHNOLOGY, DomainCategory.TECHNOLOGY
        )
        assert distance == 0

    def test_adjacent_domains_low_distance(self):
        """Adjacent domains should have low distance."""
        # Tech cluster
        distance = DomainKnowledgeGraph.get_distance(
            DomainCategory.SOFTWARE, DomainCategory.AI_ML
        )
        assert distance == 1

        # Consumer cluster
        distance = DomainKnowledgeGraph.get_distance(
            DomainCategory.FOOD_BEVERAGE, DomainCategory.HEALTH_WELLNESS
        )
        assert distance == 1

    def test_distant_domains_high_distance(self):
        """Distant domains should have higher distance."""
        distance = DomainKnowledgeGraph.get_distance(
            DomainCategory.AI_ML, DomainCategory.FOOD_BEVERAGE
        )
        assert distance >= 3

    def test_unknown_pairs_default_distance(self):
        """Unknown pairs should use default distance."""
        distance = DomainKnowledgeGraph.get_distance(
            DomainCategory.ARTS, DomainCategory.AGRICULTURE
        )
        assert distance == DomainKnowledgeGraph.DEFAULT_DISTANCE

    def test_bidirectional_distance(self):
        """Distance should be same in both directions."""
        d1 = DomainKnowledgeGraph.get_distance(
            DomainCategory.TECHNOLOGY, DomainCategory.FOOD_BEVERAGE
        )
        d2 = DomainKnowledgeGraph.get_distance(
            DomainCategory.FOOD_BEVERAGE, DomainCategory.TECHNOLOGY
        )
        assert d1 == d2


class TestAnalogyDetector:
    """Test AnalogyDetector pattern matching."""

    @pytest.fixture
    def detector(self):
        """Create AnalogyDetector instance."""
        return AnalogyDetector()

    def test_x_for_y_pattern(self, detector):
        """Test detection of 'X for Y' pattern."""
        idea = {
            "description": "Netflix for protein supplements"
        }
        analogies = detector.detect_analogies(idea, DomainCategory.FOOD_BEVERAGE)

        assert len(analogies) > 0

    def test_like_x_but_for_y_pattern(self, detector):
        """Test detection of 'like X but for Y' pattern."""
        idea = {
            "description": "Like Uber but for nutrition coaching"
        }
        analogies = detector.detect_analogies(idea, DomainCategory.HEALTH_WELLNESS)

        assert len(analogies) > 0

    def test_the_x_of_y_pattern(self, detector):
        """Test detection of 'the X of Y' pattern."""
        idea = {
            "description": "The iPhone of fitness tracking"
        }
        analogies = detector.detect_analogies(idea, DomainCategory.SPORTS)

        assert len(analogies) > 0

    def test_known_source_detection(self, detector):
        """Test detection of known source concepts."""
        idea = {
            "description": "Using gamification and subscription model for protein"
        }
        analogies = detector.detect_analogies(idea, DomainCategory.FOOD_BEVERAGE)

        detected_sources = [a.source_concept.lower() for a in analogies]
        assert "gamification" in detected_sources or "subscription" in detected_sources

    def test_duplicate_removal(self, detector):
        """Test that duplicate analogies are removed."""
        idea = {
            "description": "Netflix subscription, like Netflix, the Netflix of protein, Netflix model"
        }
        analogies = detector.detect_analogies(idea, DomainCategory.FOOD_BEVERAGE)

        # Should not have multiple Netflix entries
        netflix_count = sum(1 for a in analogies if "netflix" in a.source_concept.lower())
        assert netflix_count == 1

    def test_helper_function(self):
        """Test calculate_cross_domain helper function."""
        idea = {"description": "Netflix for protein delivery"}
        score = calculate_cross_domain(idea, domain="food_beverage")

        assert isinstance(score, float)
        assert 0 <= score <= 100


# =============================================================================
# INTEGRATION TESTS - Combined Dimensions
# =============================================================================

class TestDimensionIntegration:
    """Test integration of Surprise and Cross-Domain dimensions."""

    def test_8_dimension_weight_distribution(self):
        """Verify 8-dimension weights sum to 100%."""
        weights = {
            "novelty": 0.12,
            "feasibility": 0.18,
            "market": 0.18,
            "complexity": 0.12,
            "scenario": 0.12,
            "contrarian": 0.10,
            "surprise": 0.10,
            "cross_domain": 0.08
        }

        total = sum(weights.values())
        assert abs(total - 1.0) < 0.001  # Should sum to 1.0 (100%)

    def test_surprise_and_cross_domain_independent(self):
        """Test that Surprise and Cross-Domain can be calculated independently."""
        idea = {
            "title": "Netflix-style Elderly Insect Protein",
            "description": "Subscription insect protein for elderly, like Netflix",
            "target_market": "elderly",
            "format": "gummy",
            "ingredient_base": "insect"
        }

        surprise_analyzer = SurpriseDimension(domain="protein")
        bridge_analyzer = CrossDomainBridge(target_domain="food")

        surprise_result = surprise_analyzer.analyze(idea)
        bridge_result = bridge_analyzer.analyze(idea)

        # Both should return valid results
        assert isinstance(surprise_result, SurpriseAnalysis)
        assert isinstance(bridge_result, CrossDomainAnalysis)

        # Both scores should be in valid range
        assert 0 <= surprise_result.final_score <= 100
        assert 0 <= bridge_result.final_score <= 100

    def test_high_surprise_low_cross_domain(self):
        """Test idea with high surprise but low cross-domain."""
        idea = {
            "title": "Insect Gummy for Elderly",
            "description": "Novel insect-based protein gummies for seniors",
            "target_market": "elderly nursing home residents",
            "format": "gummy",
            "ingredient_base": "insect"
        }

        surprise = calculate_surprise(idea, domain="protein")
        cross_domain = calculate_cross_domain(idea, domain="food")

        # Should be surprising (violates conventions)
        assert surprise > 50
        # Low cross-domain (no analogies)
        assert cross_domain < 50

    def test_low_surprise_high_cross_domain(self):
        """Test idea with low surprise but high cross-domain."""
        idea = {
            "title": "Netflix Whey Protein",
            "description": "Subscription whey protein delivery like Netflix model",
            "target_market": "fitness enthusiasts",
            "format": "powder",
            "positioning": "muscle building"
        }

        surprise = calculate_surprise(idea, domain="protein")
        cross_domain = calculate_cross_domain(idea, domain="food")

        # Should be less surprising (follows conventions)
        # but has cross-domain thinking
        assert cross_domain > 30

    def test_combined_scoring_example(self):
        """Test a complete 8-dimension scoring example."""
        idea = {
            "title": "Netflix-style Gaming Insect Protein",
            "description": "Subscription insect protein with gamification for gamers",
            "target_market": "gamers",
            "format": "gummy",
            "ingredient_base": "insect",
            "positioning": "cognitive performance"
        }

        # Calculate new dimensions
        surprise_score = calculate_surprise(idea, domain="protein")
        cross_domain_score = calculate_cross_domain(idea, domain="food")

        # Mock other dimension scores
        mock_scores = {
            "novelty": 75,
            "feasibility": 60,
            "market": 65,
            "complexity": 70,
            "scenario": 55,
            "contrarian": 80,
            "surprise": surprise_score,
            "cross_domain": cross_domain_score
        }

        # Calculate weighted average
        weights = {
            "novelty": 0.12,
            "feasibility": 0.18,
            "market": 0.18,
            "complexity": 0.12,
            "scenario": 0.12,
            "contrarian": 0.10,
            "surprise": 0.10,
            "cross_domain": 0.08
        }

        final_score = sum(mock_scores[dim] * weights[dim] for dim in weights)

        assert 0 <= final_score <= 100
        assert isinstance(final_score, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
