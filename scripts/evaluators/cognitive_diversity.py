"""
Cognitive Diversity Evaluators for Universal Ideation v3

Implements 4 evaluator personas with DIFFERENT REASONING STYLES.
v2 had agents with same cognition, different topics.
v3 has agents with fundamentally different cognitive approaches.

Based on multi-agent creativity studies showing cognitive diversity > topic diversity.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json


class EvaluatorPersona(Enum):
    CONSERVATIVE = "conservative"
    RADICAL = "radical"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class DimensionWeights:
    """Customizable weights for 8-dimension scoring."""
    novelty: float = 0.12
    feasibility: float = 0.18
    market: float = 0.18
    complexity: float = 0.12
    scenario: float = 0.12
    contrarian: float = 0.10
    surprise: float = 0.10
    cross_domain: float = 0.08

    def as_dict(self) -> Dict[str, float]:
        return {
            'novelty': self.novelty,
            'feasibility': self.feasibility,
            'market': self.market,
            'complexity': self.complexity,
            'scenario': self.scenario,
            'contrarian': self.contrarian,
            'surprise': self.surprise,
            'cross_domain': self.cross_domain
        }

    def validate(self) -> bool:
        """Ensure weights sum to 1.0."""
        total = sum(self.as_dict().values())
        return abs(total - 1.0) < 0.01


@dataclass
class EvaluationResult:
    """Result from a single evaluator."""
    persona: EvaluatorPersona
    dimension_scores: Dict[str, float]
    weighted_score: float
    reasoning: str
    confidence: float
    key_strengths: List[str]
    key_weaknesses: List[str]
    recommendation: str  # "strong_accept", "accept", "neutral", "reject", "strong_reject"


@dataclass
class EvaluatorConfig:
    """Configuration for a cognitive diversity evaluator."""
    persona: EvaluatorPersona
    agent_type: str
    system_prompt: str
    weight_overrides: DimensionWeights
    scoring_biases: Dict[str, int]  # Dimension -> bonus/penalty points


# Default v3 weights for reference
DEFAULT_WEIGHTS = DimensionWeights()


class CognitiveEvaluator:
    """
    Base class for cognitively diverse evaluators.

    Each evaluator has:
    1. Different weight overrides (prioritize different dimensions)
    2. Different system prompts (different reasoning styles)
    3. Different scoring biases (bonuses/penalties)
    """

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self.evaluation_history: List[EvaluationResult] = []

    def build_evaluation_prompt(self, idea: Dict, domain: str) -> str:
        """Build the evaluation prompt with persona-specific framing."""
        idea_text = json.dumps(idea, indent=2) if isinstance(idea, dict) else str(idea)

        prompt = f"""{self.config.system_prompt}

DOMAIN: {domain}

IDEA TO EVALUATE:
{idea_text}

EVALUATION TASK:
Score this idea on each dimension (0-100), applying your cognitive lens.

YOUR WEIGHT PRIORITIES (what matters most to you):
{self._format_weight_priorities()}

SCORING DIMENSIONS:
1. Novelty (0-100): How statistically rare is this idea?
2. Feasibility (0-100): Can this be built with current capabilities?
3. Market (0-100): Is there addressable demand and differentiation?
4. Complexity (0-100): Does it have network effects or emergent properties?
5. Scenario (0-100): Will it survive future economic/tech/market shifts?
6. Contrarian (0-100): Does it challenge conventional assumptions?
7. Surprise (0-100): Does it violate expected schemas?
8. Cross-Domain (0-100): Does it bridge distant conceptual domains?

OUTPUT FORMAT (JSON):
{{
    "dimension_scores": {{
        "novelty": <0-100>,
        "feasibility": <0-100>,
        "market": <0-100>,
        "complexity": <0-100>,
        "scenario": <0-100>,
        "contrarian": <0-100>,
        "surprise": <0-100>,
        "cross_domain": <0-100>
    }},
    "reasoning": "<2-3 sentences explaining your evaluation from your cognitive lens>",
    "confidence": <0.0-1.0>,
    "key_strengths": ["strength1", "strength2"],
    "key_weaknesses": ["weakness1", "weakness2"],
    "recommendation": "<strong_accept|accept|neutral|reject|strong_reject>"
}}
"""
        return prompt

    def _format_weight_priorities(self) -> str:
        """Format weight priorities for prompt."""
        weights = self.config.weight_overrides.as_dict()
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)

        lines = []
        for dim, weight in sorted_weights[:4]:
            lines.append(f"- {dim.title()}: {weight:.0%} (HIGH priority)")
        for dim, weight in sorted_weights[4:]:
            lines.append(f"- {dim.title()}: {weight:.0%}")

        return "\n".join(lines)

    def calculate_weighted_score(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate weighted score using this evaluator's weight overrides."""
        weights = self.config.weight_overrides.as_dict()

        total = 0.0
        for dimension, score in dimension_scores.items():
            weight = weights.get(dimension, 0.1)

            # Apply scoring biases
            bias = self.config.scoring_biases.get(dimension, 0)
            adjusted_score = max(0, min(100, score + bias))

            total += adjusted_score * weight

        return total

    def get_agent_invocation(self, idea: Dict, domain: str) -> Dict:
        """
        Get the agent invocation parameters for Task tool.

        Returns dict ready for Task tool call.
        """
        return {
            "agent": self.config.agent_type,
            "prompt": self.build_evaluation_prompt(idea, domain),
            "persona": self.config.persona.value,
            "description": f"{self.config.persona.value} evaluation"
        }


class ConservativeEvaluator(CognitiveEvaluator):
    """
    Risk-averse evaluator that penalizes unproven approaches.

    Cognitive style:
    - Skeptical of novelty without precedent
    - Rewards incremental improvements
    - Heavy weight on execution risk
    - Prefers ideas with clear precedent
    """

    def __init__(self):
        config = EvaluatorConfig(
            persona=EvaluatorPersona.CONSERVATIVE,
            agent_type="business-advisor",
            system_prompt="""You are a RISK-AVERSE, CONSERVATIVE evaluator.

YOUR COGNITIVE LENS:
- You are deeply skeptical of unproven approaches
- You penalize ideas that lack clear precedent (-20 points novelty bias)
- You reward incremental improvements over existing solutions (+15 points feasibility)
- You weight execution risk heavily
- You prefer ideas with demonstrated market validation

YOUR THINKING STYLE:
- "What could go wrong?"
- "Has this been done before successfully?"
- "What's the execution risk?"
- "Is this a proven model with a twist, or completely uncharted?"

When scoring, apply your conservative lens. An idea that seems "innovative"
to others might seem "risky and unproven" to you. That's your job.""",
            weight_overrides=DimensionWeights(
                novelty=0.05,      # LOW - novelty is risky
                feasibility=0.30,  # HIGH - execution matters most
                market=0.25,       # HIGH - proven demand
                complexity=0.10,
                scenario=0.12,
                contrarian=0.05,   # LOW - contrarian is risky
                surprise=0.05,     # LOW - surprise is unpredictable
                cross_domain=0.08
            ),
            scoring_biases={
                'novelty': -20,      # Penalize high novelty
                'feasibility': +15,  # Reward feasibility
                'contrarian': -15,   # Penalize assumption-challenging
            }
        )
        super().__init__(config)


class RadicalEvaluator(CognitiveEvaluator):
    """
    Disruption-seeking evaluator that penalizes incremental ideas.

    Cognitive style:
    - Bored by incremental improvements
    - Excited by paradigm-breaking approaches
    - Heavy weight on novelty and surprise
    - Prefers ideas that challenge industry assumptions
    """

    def __init__(self):
        config = EvaluatorConfig(
            persona=EvaluatorPersona.RADICAL,
            agent_type="contrarian-disruptor",
            system_prompt="""You are a DISRUPTION-SEEKING, RADICAL evaluator.

YOUR COGNITIVE LENS:
- You are bored by incremental improvements (-15 points if just "better")
- You reward paradigm-breaking approaches (+25 points for true disruption)
- You weight novelty and surprise heavily
- You prefer ideas that challenge entire industry assumptions
- You get excited by "crazy" ideas others dismiss

YOUR THINKING STYLE:
- "Does this break the rules?"
- "Would incumbents hate this?"
- "Is this a 10x improvement or just 10%?"
- "What sacred cows does this slaughter?"

When scoring, apply your radical lens. An idea that seems "risky"
to conservatives might seem "finally interesting" to you. That's your job.""",
            weight_overrides=DimensionWeights(
                novelty=0.25,      # HIGH - novelty is the point
                feasibility=0.10,  # LOW - execution can be figured out
                market=0.12,       # MEDIUM - markets can be created
                complexity=0.08,
                scenario=0.05,     # LOW - the future is uncertain anyway
                contrarian=0.20,   # HIGH - challenge assumptions
                surprise=0.15,     # HIGH - surprise means new
                cross_domain=0.05
            ),
            scoring_biases={
                'novelty': +25,      # Reward high novelty
                'feasibility': -10,  # Don't care about feasibility
                'contrarian': +20,   # Reward assumption-challenging
                'surprise': +15,     # Reward surprise
            }
        )
        super().__init__(config)


class ShortTermEvaluator(CognitiveEvaluator):
    """
    Speed-focused evaluator that prioritizes time-to-market.

    Cognitive style:
    - Impatient with long development cycles
    - Rewards quick time-to-market
    - Heavy weight on feasibility and execution
    - Prefers ideas that can launch in < 6 months
    """

    def __init__(self):
        config = EvaluatorConfig(
            persona=EvaluatorPersona.SHORT_TERM,
            agent_type="sme-business-strategist",
            system_prompt="""You are a SPEED-FOCUSED, SHORT-TERM evaluator.

YOUR COGNITIVE LENS:
- You are impatient with long development cycles (-20 points if >12 months)
- You reward quick time-to-market (+20 points if <6 months viable)
- You weight feasibility and immediate execution heavily
- You prefer ideas that can launch fast and iterate
- You believe "shipped beats perfect"

YOUR THINKING STYLE:
- "How fast can we launch an MVP?"
- "What's the shortest path to market?"
- "Can we test this in 3 months?"
- "Is this a quick win or a long slog?"

When scoring, apply your speed lens. An idea with "huge potential in 5 years"
scores low if it can't show results in 6 months. That's your job.""",
            weight_overrides=DimensionWeights(
                novelty=0.08,
                feasibility=0.35,  # HIGHEST - can we build it fast?
                market=0.25,       # HIGH - is there immediate demand?
                complexity=0.05,   # LOW - complexity slows things
                scenario=0.05,     # LOW - worry about future later
                contrarian=0.08,
                surprise=0.07,
                cross_domain=0.07
            ),
            scoring_biases={
                'feasibility': +20,   # Reward fast execution
                'complexity': -15,    # Penalize complexity
                'scenario': -10,      # Don't care about long-term
            }
        )
        super().__init__(config)


class LongTermEvaluator(CognitiveEvaluator):
    """
    Defensibility-focused evaluator that prioritizes sustainable advantage.

    Cognitive style:
    - Patient with long development if moat is strong
    - Penalizes easily copied ideas
    - Heavy weight on scenario resilience and complexity
    - Prefers ideas with 5+ year competitive advantage
    """

    def __init__(self):
        config = EvaluatorConfig(
            persona=EvaluatorPersona.LONG_TERM,
            agent_type="visionary-futurist",
            system_prompt="""You are a DEFENSIBILITY-FOCUSED, LONG-TERM evaluator.

YOUR COGNITIVE LENS:
- You are patient with long development if the moat is strong
- You penalize easily copied ideas (-15 points if no barrier)
- You reward moats and barriers to entry (+20 points for defensibility)
- You weight scenario resilience and network effects heavily
- You prefer ideas with 5+ year competitive advantage

YOUR THINKING STYLE:
- "What's the moat?"
- "Can competitors copy this in 2 years?"
- "Does this get stronger over time?"
- "Will this survive major market shifts?"

When scoring, apply your long-term lens. A "quick win" that anyone can copy
scores low. A slow-build with strong network effects scores high. That's your job.""",
            weight_overrides=DimensionWeights(
                novelty=0.10,
                feasibility=0.08,  # LOW - patient with execution
                market=0.12,
                complexity=0.20,   # HIGH - network effects = moat
                scenario=0.25,     # HIGHEST - future resilience
                contrarian=0.10,
                surprise=0.05,
                cross_domain=0.10  # HIGH - cross-domain = unique
            ),
            scoring_biases={
                'complexity': +20,   # Reward network effects
                'scenario': +15,     # Reward future resilience
                'feasibility': -10,  # Don't care about speed
                'cross_domain': +10, # Reward unique combinations
            }
        )
        super().__init__(config)


class EvaluatorPanel:
    """
    Panel of 4 cognitively diverse evaluators.

    Manages parallel evaluation and disagreement detection.
    """

    def __init__(self):
        self.evaluators = {
            EvaluatorPersona.CONSERVATIVE: ConservativeEvaluator(),
            EvaluatorPersona.RADICAL: RadicalEvaluator(),
            EvaluatorPersona.SHORT_TERM: ShortTermEvaluator(),
            EvaluatorPersona.LONG_TERM: LongTermEvaluator(),
        }
        self.evaluation_history: List[Dict] = []

    def get_all_prompts(self, idea: Dict, domain: str) -> List[Dict]:
        """
        Get evaluation prompts for all 4 evaluators.

        Returns list of dicts ready for parallel Task tool calls.
        """
        prompts = []
        for persona, evaluator in self.evaluators.items():
            invocation = evaluator.get_agent_invocation(idea, domain)
            prompts.append(invocation)
        return prompts

    def aggregate_scores(
        self,
        results: Dict[EvaluatorPersona, EvaluationResult]
    ) -> Dict:
        """
        Aggregate scores from all evaluators.

        Returns:
            - consensus_score: Average of all evaluators
            - disagreement_level: Max difference between evaluators
            - needs_debate: Whether structured debate should be triggered
            - dimension_consensus: Per-dimension analysis
        """
        if not results:
            return {"error": "No results to aggregate"}

        scores = [r.weighted_score for r in results.values()]

        consensus_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        disagreement = max_score - min_score

        # Find which evaluators disagree most
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].weighted_score,
            reverse=True
        )
        highest = sorted_results[0]
        lowest = sorted_results[-1]

        # Analyze dimension-level consensus
        dimension_scores = {}
        for dim in ['novelty', 'feasibility', 'market', 'complexity',
                    'scenario', 'contrarian', 'surprise', 'cross_domain']:
            dim_values = [r.dimension_scores.get(dim, 0) for r in results.values()]
            dimension_scores[dim] = {
                'mean': sum(dim_values) / len(dim_values),
                'range': max(dim_values) - min(dim_values),
                'high_disagreement': (max(dim_values) - min(dim_values)) > 25
            }

        return {
            "consensus_score": consensus_score,
            "max_score": max_score,
            "min_score": min_score,
            "disagreement_level": disagreement,
            "needs_debate": disagreement > 20,
            "highest_scorer": {
                "persona": highest[0].value,
                "score": highest[1].weighted_score,
                "recommendation": highest[1].recommendation
            },
            "lowest_scorer": {
                "persona": lowest[0].value,
                "score": lowest[1].weighted_score,
                "recommendation": lowest[1].recommendation
            },
            "dimension_consensus": dimension_scores,
            "recommendations": {
                r.persona.value: r.recommendation
                for r in results.values()
            }
        }

    def detect_debate_triggers(
        self,
        results: Dict[EvaluatorPersona, EvaluationResult]
    ) -> List[str]:
        """
        Detect specific triggers that warrant structured debate.

        Returns list of debate trigger reasons.
        """
        triggers = []

        scores = [r.weighted_score for r in results.values()]
        disagreement = max(scores) - min(scores)

        # Trigger 1: High overall disagreement
        if disagreement > 20:
            triggers.append(f"High disagreement: {disagreement:.1f} points spread")

        # Trigger 2: Opposite recommendations
        recommendations = [r.recommendation for r in results.values()]
        if 'strong_accept' in recommendations and 'strong_reject' in recommendations:
            triggers.append("Polar opposite recommendations (strong_accept vs strong_reject)")

        # Trigger 3: Radical vs Conservative conflict
        if EvaluatorPersona.RADICAL in results and EvaluatorPersona.CONSERVATIVE in results:
            radical_score = results[EvaluatorPersona.RADICAL].weighted_score
            conservative_score = results[EvaluatorPersona.CONSERVATIVE].weighted_score
            if abs(radical_score - conservative_score) > 25:
                triggers.append(f"Radical-Conservative conflict: {abs(radical_score - conservative_score):.1f} points")

        # Trigger 4: Time horizon conflict
        if EvaluatorPersona.SHORT_TERM in results and EvaluatorPersona.LONG_TERM in results:
            short_score = results[EvaluatorPersona.SHORT_TERM].weighted_score
            long_score = results[EvaluatorPersona.LONG_TERM].weighted_score
            if abs(short_score - long_score) > 25:
                triggers.append(f"Time horizon conflict: {abs(short_score - long_score):.1f} points")

        return triggers

    def get_panel_statistics(self) -> Dict:
        """Get statistics about panel evaluations."""
        return {
            "total_evaluations": len(self.evaluation_history),
            "evaluator_count": len(self.evaluators),
            "personas": [p.value for p in self.evaluators.keys()]
        }


def create_evaluation_result(
    persona: EvaluatorPersona,
    dimension_scores: Dict[str, float],
    reasoning: str,
    confidence: float = 0.8,
    key_strengths: Optional[List[str]] = None,
    key_weaknesses: Optional[List[str]] = None,
    recommendation: str = "neutral"
) -> EvaluationResult:
    """Helper to create EvaluationResult from raw data."""

    # Get evaluator to calculate weighted score
    evaluator_map = {
        EvaluatorPersona.CONSERVATIVE: ConservativeEvaluator(),
        EvaluatorPersona.RADICAL: RadicalEvaluator(),
        EvaluatorPersona.SHORT_TERM: ShortTermEvaluator(),
        EvaluatorPersona.LONG_TERM: LongTermEvaluator(),
    }

    evaluator = evaluator_map[persona]
    weighted_score = evaluator.calculate_weighted_score(dimension_scores)

    return EvaluationResult(
        persona=persona,
        dimension_scores=dimension_scores,
        weighted_score=weighted_score,
        reasoning=reasoning,
        confidence=confidence,
        key_strengths=key_strengths or [],
        key_weaknesses=key_weaknesses or [],
        recommendation=recommendation
    )
