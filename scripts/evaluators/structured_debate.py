"""
Structured Debate System for Universal Ideation v3

When evaluators disagree significantly, triggers a structured debate:
1. PROPOSE: Each side states their position
2. CHALLENGE: Each side challenges the other's assumptions
3. SYNTHESIZE: First-principles analyst finds the truth

Based on dialectical reasoning and adversarial collaboration research.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import json

from .cognitive_diversity import (
    EvaluatorPersona,
    EvaluationResult,
    EvaluatorPanel
)


class DebatePhase(Enum):
    PROPOSE = "propose"
    CHALLENGE = "challenge"
    SYNTHESIZE = "synthesize"


@dataclass
class DebatePosition:
    """A position in the debate."""
    persona: EvaluatorPersona
    stance: str  # "support" or "oppose"
    score: float
    key_arguments: List[str]
    assumptions: List[str]
    evidence: List[str]


@dataclass
class DebateChallenge:
    """A challenge to a position."""
    challenger: EvaluatorPersona
    target: EvaluatorPersona
    challenged_assumptions: List[str]
    counter_arguments: List[str]
    alternative_interpretation: str


@dataclass
class DebateSynthesis:
    """The synthesis of the debate."""
    revised_score: float
    confidence: float
    key_insights: List[str]
    resolved_disagreements: List[str]
    unresolved_tensions: List[str]
    final_recommendation: str
    reasoning: str


@dataclass
class DebateResult:
    """Complete result of a structured debate."""
    idea: Dict
    trigger_reason: str
    positions: List[DebatePosition]
    challenges: List[DebateChallenge]
    synthesis: DebateSynthesis
    original_scores: Dict[str, float]
    debate_impact: float  # How much synthesis differs from simple average


class StructuredDebate:
    """
    Manages structured debate between disagreeing evaluators.

    Debate flow:
    1. Identify the highest and lowest scorers
    2. Have each state their position (PROPOSE)
    3. Have each challenge the other (CHALLENGE)
    4. Use first-principles-analyst to synthesize (SYNTHESIZE)
    """

    DEBATE_THRESHOLD = 20.0  # Minimum disagreement to trigger debate

    def __init__(self):
        self.debate_history: List[DebateResult] = []

    def should_debate(
        self,
        results: Dict[EvaluatorPersona, EvaluationResult]
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if debate should be triggered.

        Returns:
            Tuple of (should_debate, reason)
        """
        if len(results) < 2:
            return False, None

        scores = [r.weighted_score for r in results.values()]
        disagreement = max(scores) - min(scores)

        if disagreement >= self.DEBATE_THRESHOLD:
            return True, f"High disagreement: {disagreement:.1f} points"

        # Check for polar opposite recommendations
        recommendations = [r.recommendation for r in results.values()]
        if ('strong_accept' in recommendations and
            ('reject' in recommendations or 'strong_reject' in recommendations)):
            return True, "Polar opposite recommendations"

        return False, None

    def identify_debaters(
        self,
        results: Dict[EvaluatorPersona, EvaluationResult]
    ) -> Tuple[EvaluationResult, EvaluationResult]:
        """
        Identify the two most disagreeing evaluators.

        Returns highest and lowest scorers.
        """
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.weighted_score,
            reverse=True
        )
        return sorted_results[0], sorted_results[-1]

    def build_propose_prompt(
        self,
        idea: Dict,
        evaluation: EvaluationResult,
        domain: str
    ) -> str:
        """Build prompt for PROPOSE phase."""
        stance = "SUPPORT" if evaluation.weighted_score >= 70 else "OPPOSE"

        return f"""You are the {evaluation.persona.value.upper()} evaluator in a structured debate.

IDEA BEING DEBATED:
{json.dumps(idea, indent=2)}

DOMAIN: {domain}

YOUR EVALUATION:
- Score: {evaluation.weighted_score:.1f}/100
- Recommendation: {evaluation.recommendation}
- Reasoning: {evaluation.reasoning}

YOUR TASK - PROPOSE PHASE:
State your position clearly. You {stance} this idea.

Provide:
1. Your stance (support/oppose) and conviction level (1-10)
2. Your 3 strongest arguments for your position
3. The key assumptions underlying your evaluation
4. Evidence or reasoning that supports your view

OUTPUT FORMAT (JSON):
{{
    "stance": "{stance.lower()}",
    "conviction": <1-10>,
    "key_arguments": [
        "Argument 1 with specific reasoning",
        "Argument 2 with specific reasoning",
        "Argument 3 with specific reasoning"
    ],
    "assumptions": [
        "Key assumption 1",
        "Key assumption 2"
    ],
    "evidence": [
        "Evidence or reasoning 1",
        "Evidence or reasoning 2"
    ]
}}
"""

    def build_challenge_prompt(
        self,
        idea: Dict,
        my_position: DebatePosition,
        opponent_position: DebatePosition,
        domain: str
    ) -> str:
        """Build prompt for CHALLENGE phase."""
        return f"""You are the {my_position.persona.value.upper()} evaluator in a structured debate.

IDEA BEING DEBATED:
{json.dumps(idea, indent=2)}

DOMAIN: {domain}

YOUR POSITION:
- Stance: {my_position.stance}
- Score: {my_position.score:.1f}
- Arguments: {my_position.key_arguments}

OPPONENT'S POSITION ({opponent_position.persona.value.upper()}):
- Stance: {opponent_position.stance}
- Score: {opponent_position.score:.1f}
- Arguments: {opponent_position.key_arguments}
- Assumptions: {opponent_position.assumptions}

YOUR TASK - CHALLENGE PHASE:
Challenge your opponent's position. Find flaws in their reasoning.

1. Which of their assumptions are questionable or wrong?
2. What are the counter-arguments to their key points?
3. What are they missing or ignoring?
4. How would you reinterpret their evidence?

OUTPUT FORMAT (JSON):
{{
    "challenged_assumptions": [
        "Their assumption X is wrong because...",
        "They assume Y but..."
    ],
    "counter_arguments": [
        "Counter to their argument 1: ...",
        "Counter to their argument 2: ..."
    ],
    "what_they_miss": [
        "They ignore...",
        "They fail to consider..."
    ],
    "alternative_interpretation": "A different way to see this is..."
}}
"""

    def build_synthesis_prompt(
        self,
        idea: Dict,
        positions: List[DebatePosition],
        challenges: List[DebateChallenge],
        domain: str
    ) -> str:
        """Build prompt for SYNTHESIZE phase (first-principles-analyst)."""
        positions_text = ""
        for p in positions:
            positions_text += f"""
{p.persona.value.upper()} ({p.stance}, score: {p.score:.1f}):
- Arguments: {p.key_arguments}
- Assumptions: {p.assumptions}
"""

        challenges_text = ""
        for c in challenges:
            challenges_text += f"""
{c.challenger.value.upper()} challenges {c.target.value.upper()}:
- Challenged assumptions: {c.challenged_assumptions}
- Counter-arguments: {c.counter_arguments}
- Alternative view: {c.alternative_interpretation}
"""

        return f"""You are a FIRST-PRINCIPLES ANALYST synthesizing a structured debate.

IDEA BEING DEBATED:
{json.dumps(idea, indent=2)}

DOMAIN: {domain}

THE POSITIONS:
{positions_text}

THE CHALLENGES:
{challenges_text}

YOUR TASK - SYNTHESIZE:
Apply first-principles thinking to resolve this debate.

1. Strip away the cognitive biases of each evaluator
2. Identify which arguments have merit regardless of perspective
3. Determine where the truth actually lies
4. Provide a revised, balanced score

Consider:
- Which assumptions survived the challenges?
- Which arguments are objectively stronger?
- What is the TRUE potential of this idea?
- What are the evaluators missing collectively?

OUTPUT FORMAT (JSON):
{{
    "revised_score": <0-100>,
    "confidence": <0.0-1.0>,
    "key_insights": [
        "Insight 1: What the debate revealed",
        "Insight 2: The real issue at stake"
    ],
    "resolved_disagreements": [
        "The X disagreement resolves to...",
        "The Y tension resolves to..."
    ],
    "unresolved_tensions": [
        "Still uncertain: ...",
        "Genuinely ambiguous: ..."
    ],
    "final_recommendation": "<strong_accept|accept|neutral|reject|strong_reject>",
    "reasoning": "2-3 sentence synthesis of the true evaluation"
}}
"""

    def get_debate_prompts(
        self,
        idea: Dict,
        results: Dict[EvaluatorPersona, EvaluationResult],
        domain: str
    ) -> Dict[str, List[Dict]]:
        """
        Get all prompts needed for the debate.

        Returns structured prompts for each phase.
        """
        high_scorer, low_scorer = self.identify_debaters(results)

        # Phase 1: PROPOSE prompts
        propose_prompts = [
            {
                "phase": "propose",
                "persona": high_scorer.persona.value,
                "agent": "business-advisor",  # Generic for propose
                "prompt": self.build_propose_prompt(idea, high_scorer, domain)
            },
            {
                "phase": "propose",
                "persona": low_scorer.persona.value,
                "agent": "business-advisor",
                "prompt": self.build_propose_prompt(idea, low_scorer, domain)
            }
        ]

        return {
            "propose": propose_prompts,
            "high_scorer": high_scorer,
            "low_scorer": low_scorer,
            "domain": domain
        }

    def get_challenge_prompts(
        self,
        idea: Dict,
        positions: List[DebatePosition],
        domain: str
    ) -> List[Dict]:
        """Get CHALLENGE phase prompts after PROPOSE completes."""
        if len(positions) < 2:
            return []

        # Each position challenges the other
        challenges = []
        for i, pos in enumerate(positions):
            opponent = positions[1 - i]  # The other position
            challenges.append({
                "phase": "challenge",
                "persona": pos.persona.value,
                "agent": "contrarian-disruptor",  # Good at challenging
                "prompt": self.build_challenge_prompt(idea, pos, opponent, domain)
            })

        return challenges

    def get_synthesis_prompt(
        self,
        idea: Dict,
        positions: List[DebatePosition],
        challenges: List[DebateChallenge],
        domain: str
    ) -> Dict:
        """Get SYNTHESIZE phase prompt after CHALLENGE completes."""
        return {
            "phase": "synthesize",
            "persona": "first_principles",
            "agent": "first-principles-analyst",
            "prompt": self.build_synthesis_prompt(idea, positions, challenges, domain)
        }

    def calculate_debate_impact(
        self,
        original_scores: Dict[str, float],
        synthesis_score: float
    ) -> float:
        """Calculate how much the debate changed the evaluation."""
        if not original_scores:
            return 0.0

        simple_average = sum(original_scores.values()) / len(original_scores)
        return synthesis_score - simple_average

    def create_debate_result(
        self,
        idea: Dict,
        trigger_reason: str,
        positions: List[DebatePosition],
        challenges: List[DebateChallenge],
        synthesis: DebateSynthesis,
        original_results: Dict[EvaluatorPersona, EvaluationResult]
    ) -> DebateResult:
        """Create a complete debate result."""
        original_scores = {
            p.value: r.weighted_score
            for p, r in original_results.items()
        }

        result = DebateResult(
            idea=idea,
            trigger_reason=trigger_reason,
            positions=positions,
            challenges=challenges,
            synthesis=synthesis,
            original_scores=original_scores,
            debate_impact=self.calculate_debate_impact(
                original_scores,
                synthesis.revised_score
            )
        )

        self.debate_history.append(result)
        return result

    def get_statistics(self) -> Dict:
        """Get debate statistics."""
        if not self.debate_history:
            return {
                "total_debates": 0,
                "average_impact": 0.0
            }

        impacts = [d.debate_impact for d in self.debate_history]

        return {
            "total_debates": len(self.debate_history),
            "average_impact": sum(impacts) / len(impacts),
            "max_impact": max(impacts),
            "debates_raised_score": sum(1 for i in impacts if i > 0),
            "debates_lowered_score": sum(1 for i in impacts if i < 0)
        }


def parse_propose_response(response: str, persona: EvaluatorPersona, score: float) -> DebatePosition:
    """Parse the JSON response from PROPOSE phase."""
    try:
        data = json.loads(response)
        return DebatePosition(
            persona=persona,
            stance=data.get('stance', 'neutral'),
            score=score,
            key_arguments=data.get('key_arguments', []),
            assumptions=data.get('assumptions', []),
            evidence=data.get('evidence', [])
        )
    except json.JSONDecodeError:
        return DebatePosition(
            persona=persona,
            stance='neutral',
            score=score,
            key_arguments=["Unable to parse arguments"],
            assumptions=[],
            evidence=[]
        )


def parse_challenge_response(
    response: str,
    challenger: EvaluatorPersona,
    target: EvaluatorPersona
) -> DebateChallenge:
    """Parse the JSON response from CHALLENGE phase."""
    try:
        data = json.loads(response)
        return DebateChallenge(
            challenger=challenger,
            target=target,
            challenged_assumptions=data.get('challenged_assumptions', []),
            counter_arguments=data.get('counter_arguments', []),
            alternative_interpretation=data.get('alternative_interpretation', '')
        )
    except json.JSONDecodeError:
        return DebateChallenge(
            challenger=challenger,
            target=target,
            challenged_assumptions=[],
            counter_arguments=["Unable to parse challenges"],
            alternative_interpretation=""
        )


def parse_synthesis_response(response: str) -> DebateSynthesis:
    """Parse the JSON response from SYNTHESIZE phase."""
    try:
        data = json.loads(response)
        return DebateSynthesis(
            revised_score=data.get('revised_score', 70.0),
            confidence=data.get('confidence', 0.7),
            key_insights=data.get('key_insights', []),
            resolved_disagreements=data.get('resolved_disagreements', []),
            unresolved_tensions=data.get('unresolved_tensions', []),
            final_recommendation=data.get('final_recommendation', 'neutral'),
            reasoning=data.get('reasoning', '')
        )
    except json.JSONDecodeError:
        return DebateSynthesis(
            revised_score=70.0,
            confidence=0.5,
            key_insights=["Unable to parse synthesis"],
            resolved_disagreements=[],
            unresolved_tensions=["Parsing failed"],
            final_recommendation='neutral',
            reasoning="Synthesis parsing failed"
        )


class QuickDebate:
    """
    Simplified debate for when full 3-phase debate is too expensive.

    Uses a single synthesis step without full propose/challenge phases.
    """

    def __init__(self):
        self.debate = StructuredDebate()

    def build_quick_synthesis_prompt(
        self,
        idea: Dict,
        results: Dict[EvaluatorPersona, EvaluationResult],
        domain: str
    ) -> Dict:
        """Build a single prompt that synthesizes all evaluations."""
        evaluations_text = ""
        for persona, result in results.items():
            evaluations_text += f"""
{persona.value.upper()} (Score: {result.weighted_score:.1f}):
- Recommendation: {result.recommendation}
- Reasoning: {result.reasoning}
- Strengths: {result.key_strengths}
- Weaknesses: {result.key_weaknesses}
"""

        prompt = f"""You are synthesizing conflicting evaluations as a first-principles analyst.

IDEA:
{json.dumps(idea, indent=2)}

DOMAIN: {domain}

EVALUATOR ASSESSMENTS:
{evaluations_text}

The evaluators disagree significantly. Your task:
1. Identify what each evaluator is missing
2. Find the objective truth about this idea's potential
3. Provide a balanced, synthesized score

OUTPUT FORMAT (JSON):
{{
    "revised_score": <0-100>,
    "confidence": <0.0-1.0>,
    "synthesis": "2-3 sentence balanced evaluation",
    "what_conservatives_miss": "...",
    "what_radicals_miss": "...",
    "final_recommendation": "<strong_accept|accept|neutral|reject|strong_reject>"
}}
"""

        return {
            "phase": "quick_synthesis",
            "agent": "first-principles-analyst",
            "prompt": prompt
        }
