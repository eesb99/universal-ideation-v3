"""
Triple Generator System for Universal Ideation v3.1

Implements three distinct generation modes to escape mode collapse:
1. Explorer: Maximize semantic distance, ignore learnings (SCAMPER-guided)
2. Refiner: Use learnings, optimize within proven clusters (Design Thinking-guided)
3. Contrarian: Invert learnings, challenge assumptions (TRIZ-guided)

Based on Dual-Pathway Model (flexibility + persistence) from creativity science.

v3.1 Changes:
- TCRTE prompt structure (Task, Context, Role, Tone, Examples)
- Constraint-based creativity support
- Framework-guided generation (SCAMPER, Design Thinking, TRIZ)
"""

import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal
from enum import Enum


class GeneratorMode(Enum):
    EXPLORER = "explorer"
    REFINER = "refiner"
    CONTRARIAN = "contrarian"


class ConstraintTemplate(Enum):
    """Pre-defined constraint templates for focused ideation."""
    NONE = "none"
    BOOTSTRAP = "bootstrap"      # Limited budget, MVP focus, speed-to-market
    ENTERPRISE = "enterprise"    # Scalability, compliance, integration
    REGULATED = "regulated"      # FDA/regulatory pathway, safety-first
    SUSTAINABLE = "sustainable"  # Environmental, circular economy
    CUSTOM = "custom"            # User-defined constraints


@dataclass
class ConstraintConfig:
    """Configuration for constraint-based creativity."""
    template: ConstraintTemplate = ConstraintTemplate.NONE
    custom_constraints: List[str] = field(default_factory=list)
    budget_limit: Optional[str] = None        # e.g., "$50k", "minimal"
    time_to_market: Optional[str] = None      # e.g., "3 months", "fast"
    resource_constraints: Optional[str] = None # e.g., "small team", "outsourced"

    def get_constraint_text(self) -> str:
        """Generate constraint text for prompt injection."""
        if self.template == ConstraintTemplate.NONE:
            return ""

        constraints = []

        if self.template == ConstraintTemplate.BOOTSTRAP:
            constraints = [
                "LIMITED BUDGET: Solution must be achievable with minimal capital (<$50k)",
                "MVP FOCUS: Design for fastest path to first paying customer",
                "SPEED: Time-to-market under 3 months is critical",
                "LEAN: Must be executable by small team (1-3 people)"
            ]
        elif self.template == ConstraintTemplate.ENTERPRISE:
            constraints = [
                "SCALABILITY: Must handle 10x-100x growth without redesign",
                "COMPLIANCE: Consider SOC2, GDPR, industry regulations",
                "INTEGRATION: Must work with existing enterprise systems",
                "SECURITY: Enterprise-grade security requirements"
            ]
        elif self.template == ConstraintTemplate.REGULATED:
            constraints = [
                "REGULATORY: Must have clear FDA/regulatory approval pathway",
                "SAFETY: Patient/consumer safety is non-negotiable priority",
                "DOCUMENTATION: Full traceability and documentation required",
                "TIMELINE: Factor in regulatory approval timelines (12-36 months)"
            ]
        elif self.template == ConstraintTemplate.SUSTAINABLE:
            constraints = [
                "ENVIRONMENTAL: Minimize carbon footprint and waste",
                "CIRCULAR: Design for recyclability/biodegradability",
                "SOURCING: Prefer sustainable and ethical supply chains",
                "LIFECYCLE: Consider full product lifecycle impact"
            ]
        elif self.template == ConstraintTemplate.CUSTOM:
            constraints = self.custom_constraints

        # Add specific overrides
        if self.budget_limit:
            constraints.append(f"BUDGET: {self.budget_limit}")
        if self.time_to_market:
            constraints.append(f"TIMELINE: {self.time_to_market}")
        if self.resource_constraints:
            constraints.append(f"RESOURCES: {self.resource_constraints}")

        if not constraints:
            return ""

        return "\n\n## CONSTRAINTS (Ideas MUST satisfy these):\n" + "\n".join(f"- {c}" for c in constraints)


@dataclass
class GeneratedIdea:
    """Container for a generated idea with metadata."""
    title: str
    description: str
    differentiators: List[str]
    target_market: str
    mode: GeneratorMode
    iteration: int
    prompt_used: str


@dataclass
class ModeWeights:
    """Weights for mode selection."""
    explorer: float = 0.4
    refiner: float = 0.4
    contrarian: float = 0.2

    def as_list(self) -> List[float]:
        return [self.explorer, self.refiner, self.contrarian]


class TripleGenerator:
    """
    Three distinct generation modes with different objectives.

    Science-backed design:
    - Explorer: Flexibility pathway (broad exploration, SCAMPER-guided)
    - Refiner: Persistence pathway (deep refinement, Design Thinking-guided)
    - Contrarian: Assumption-challenging (invert patterns, TRIZ-guided)

    v3.1 Features:
    - TCRTE prompt structure for all modes
    - Constraint injection for focused ideation
    - Framework-guided generation
    """

    def __init__(self, constraints: Optional[ConstraintConfig] = None):
        self.mode_history: List[GeneratorMode] = []
        self.exploration_budget: float = 1.0  # Starts at 100%, decreases over time
        self.mode_stats: Dict[GeneratorMode, int] = {
            GeneratorMode.EXPLORER: 0,
            GeneratorMode.REFINER: 0,
            GeneratorMode.CONTRARIAN: 0
        }
        self.constraints = constraints or ConstraintConfig()

    def select_mode(
        self,
        iteration: int,
        max_iterations: int,
        recent_scores: List[float],
        is_plateau: bool = False
    ) -> GeneratorMode:
        """
        Adaptive mode selection based on session state.

        Args:
            iteration: Current iteration number
            max_iterations: Maximum iterations for session
            recent_scores: Last N scores for trend analysis
            is_plateau: Whether plateau has been detected

        Returns:
            Selected generation mode
        """
        # Calculate exploration budget (decreases over time)
        self.exploration_budget = 1.0 - (iteration / max_iterations)

        # Determine weights based on session state
        if is_plateau:
            # Plateau detected: force exploration
            weights = ModeWeights(explorer=0.7, refiner=0.1, contrarian=0.2)
        elif self.exploration_budget > 0.6:
            # Early session: favor exploration
            weights = ModeWeights(explorer=0.5, refiner=0.3, contrarian=0.2)
        elif self.exploration_budget > 0.3:
            # Mid session: balanced
            weights = ModeWeights(explorer=0.4, refiner=0.4, contrarian=0.2)
        else:
            # Late session: favor refinement
            weights = ModeWeights(explorer=0.3, refiner=0.5, contrarian=0.2)

        # Boost contrarian if scores are stagnating
        if len(recent_scores) >= 5:
            score_variance = self._calculate_variance(recent_scores[-5:])
            if score_variance < 2.0:  # Low variance = stagnation
                weights.contrarian = min(0.4, weights.contrarian + 0.15)
                weights.refiner = max(0.2, weights.refiner - 0.15)

        # Select mode
        modes = [GeneratorMode.EXPLORER, GeneratorMode.REFINER, GeneratorMode.CONTRARIAN]
        selected = random.choices(modes, weights=weights.as_list())[0]

        # Track history
        self.mode_history.append(selected)
        self.mode_stats[selected] += 1

        return selected

    def generate(
        self,
        mode: GeneratorMode,
        domain: str,
        learnings: List[Dict],
        prior_ideas: List[Dict],
        best_ideas: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Generate an idea using the specified mode.

        Returns prompt dict for agent invocation (not actual generation).
        Actual generation happens via Task tool in orchestrator.
        """
        if mode == GeneratorMode.EXPLORER:
            return self._build_explorer_prompt(domain, prior_ideas)
        elif mode == GeneratorMode.REFINER:
            return self._build_refiner_prompt(domain, learnings, best_ideas or [])
        elif mode == GeneratorMode.CONTRARIAN:
            return self._build_contrarian_prompt(domain, learnings)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _build_explorer_prompt(self, domain: str, prior_ideas: List[Dict]) -> Dict:
        """
        Explorer mode: Maximize semantic distance, SCAMPER-guided.

        Purpose: Escape local optima through broad exploration.
        Uses TCRTE prompt structure (Task, Context, Role, Tone, Examples).
        """
        prior_summary = self._summarize_ideas(prior_ideas, max_ideas=10)
        constraint_text = self.constraints.get_constraint_text()

        # TCRTE-structured prompt
        prompt = f"""## TASK
Generate a breakthrough innovation concept in the {domain} space that is MAXIMALLY DIFFERENT from all prior ideas. Apply SCAMPER methodology to force novel thinking.

## CONTEXT
**Domain:** {domain}
**Session Phase:** Exploration (prioritize novelty over feasibility)
**Prior Ideas to Avoid (be semantically DIFFERENT):**
{prior_summary}
{constraint_text}

## ROLE
You are a cross-industry innovation scout who specializes in bringing breakthrough ideas from unrelated fields. You think like:
- A child who doesn't know "how things are done"
- Someone from automotive, gaming, or fashion industry
- A contrarian investor who bets against the crowd

## TONE
Bold, unconventional, and provocative. Challenge assumptions. Don't play safe.

## SCAMPER FRAMEWORK (Apply at least 2)
- **S**ubstitute: What can be replaced with something unexpected?
- **C**ombine: What unrelated concepts can merge?
- **A**dapt: What can be borrowed from another industry?
- **M**odify/Magnify: What can be exaggerated or minimized?
- **P**ut to another use: What new context could this work in?
- **E**liminate: What "essential" element can be removed?
- **R**earrange/Reverse: What can be flipped or reordered?

## EXAMPLES
Good Explorer ideas:
- "Protein subscription box that changes flavor based on your Spotify listening history" (Combine: music + nutrition)
- "Gym vending machines selling single-serve protein in reusable bottles you return" (Adapt: from beer growler model)

Bad Explorer ideas (too similar to existing):
- "Another plant protein with better taste" (incremental, not breakthrough)
- "Protein bar with new packaging" (surface change only)

## OUTPUT FORMAT (JSON)
{{
    "title": "Concept name (concise, memorable)",
    "description": "2-3 sentences explaining what it is and why it matters",
    "differentiators": ["unique factor 1", "unique factor 2", "unique factor 3"],
    "target_market": "Specific target customer segment",
    "scamper_techniques_used": ["technique 1", "technique 2"],
    "semantic_distance_strategy": "How this differs from prior ideas"
}}"""

        return {
            "mode": "explorer",
            "agent": "creative-ideation-specialist",
            "prompt": prompt,
            "objective": "maximize_semantic_distance",
            "framework": "SCAMPER"
        }

    def _build_refiner_prompt(
        self,
        domain: str,
        learnings: List[Dict],
        best_ideas: List[Dict]
    ) -> Dict:
        """
        Refiner mode: Use learnings, Design Thinking-guided optimization.

        Purpose: Deep exploitation of promising directions.
        Uses TCRTE prompt structure (Task, Context, Role, Tone, Examples).
        """
        learnings_text = self._format_learnings(learnings)
        best_ideas_text = self._summarize_ideas(best_ideas, max_ideas=5)
        constraint_text = self.constraints.get_constraint_text()

        # TCRTE-structured prompt
        prompt = f"""## TASK
Refine and optimize the most promising ideas in the {domain} space. Apply Design Thinking methodology to ensure user-centered feasibility.

## CONTEXT
**Domain:** {domain}
**Session Phase:** Refinement (prioritize feasibility over novelty)
**Learnings from Prior Iterations:**
{learnings_text}

**Best-Performing Ideas to Build On:**
{best_ideas_text}
{constraint_text}

## ROLE
You are a product development expert who bridges innovation and execution. You specialize in taking promising concepts and making them market-ready through:
- User empathy and pain point understanding
- Rapid prototyping mindset
- Go-to-market execution focus

## TONE
Pragmatic, user-focused, and execution-oriented. Ground ideas in reality while preserving their innovative core.

## DESIGN THINKING FRAMEWORK (Apply all 3 phases)
1. **Empathize:** What specific user pain does this solve? Who exactly benefits?
2. **Ideate:** How can we combine strengths from multiple top ideas?
3. **Prototype:** What's the minimum viable version? How do we test it fast?

## REFINEMENT STRATEGIES
- Combine strengths from multiple top ideas
- Address specific weaknesses mentioned in learnings
- Optimize for execution and time-to-market
- Enhance differentiation while maintaining feasibility
- Remove unnecessary complexity

## EXAMPLES
Good Refiner ideas:
- Taking a "protein + AI" concept and making it specific: "AI tracks your workout intensity via Apple Watch and auto-adjusts protein delivery timing"
- Combining two winning concepts: "Gym vending + subscription = pre-paid protein credits you use at any partner gym"

Bad Refiner ideas:
- "Just make it taste better" (too vague, not actionable)
- "Add more features" (complexity without value)

## OUTPUT FORMAT (JSON)
{{
    "title": "Concept name (concise, memorable)",
    "description": "2-3 sentences explaining what it is and why it matters",
    "differentiators": ["unique factor 1", "unique factor 2", "unique factor 3"],
    "target_market": "Specific target customer segment",
    "user_pain_addressed": "The specific problem this solves",
    "mvp_description": "Minimum viable version for testing",
    "improvements_from_prior": "What this improves vs prior best ideas"
}}"""

        return {
            "mode": "refiner",
            "agent": "creative-ideation-specialist",
            "prompt": prompt,
            "objective": "optimize_within_cluster",
            "framework": "Design Thinking"
        }

    def _build_contrarian_prompt(self, domain: str, learnings: List[Dict]) -> Dict:
        """
        Contrarian mode: Invert learnings, TRIZ-guided contradiction resolution.

        Purpose: Escape local optima by questioning learned patterns.
        Uses TCRTE prompt structure (Task, Context, Role, Tone, Examples).
        """
        learnings_text = self._format_learnings(learnings)
        inverted_learnings = self._invert_learnings(learnings)
        constraint_text = self.constraints.get_constraint_text()

        # TCRTE-structured prompt
        prompt = f"""## TASK
Generate a paradigm-breaking innovation in the {domain} space that CONTRADICTS conventional wisdom. Apply TRIZ methodology to resolve contradictions without compromise.

## CONTEXT
**Domain:** {domain}
**Session Phase:** Contrarian (challenge assumptions, break patterns)
**The System Has Learned These Patterns "Work":**
{learnings_text}

**But What If The Opposite Is True?**
{inverted_learnings}
{constraint_text}

## ROLE
You are a strategic disruptor and blue ocean strategist. You specialize in:
- Finding hidden contradictions in accepted wisdom
- Identifying ignored market segments
- Creating category-defining innovations by breaking rules
- Asking "What if everyone is wrong about X?"

## TONE
Provocative, contrarian, and intellectually fearless. Challenge sacred cows. Question fundamentals.

## TRIZ FRAMEWORK (Apply at least 1 principle)
TRIZ resolves contradictions without compromise. Use these inventive principles:

**Contradiction Resolution Principles:**
1. **Segmentation:** Divide into independent parts
2. **Taking Out:** Extract the "disturbing" part
3. **Inversion:** Do the opposite action
4. **Dynamics:** Make rigid things flexible (or vice versa)
5. **Asymmetry:** Replace symmetry with asymmetry
6. **Merging:** Bring closer together or merge
7. **Universality:** Make a part perform multiple functions
8. **Nesting:** Place one inside another
9. **Prior Action:** Perform action before it's needed
10. **Copying:** Use cheap copies instead of expensive originals

**Ask:** What contradiction exists in the {domain} space that everyone accepts but shouldn't?

## CONTRARIAN STRATEGIES
- If learnings say "X works" → try the opposite of X
- If learnings say "avoid Y" → explore why Y might actually work
- Question fundamental assumptions about the domain
- Look for ignored segments, formats, or approaches
- Apply reversal: "What would make this problem WORSE?" then invert

## EXAMPLES
Good Contrarian ideas:
- "Protein powder that tastes BAD on purpose" (Inversion - bitter = perceived potency, targets hardcore athletes)
- "Protein with LESS protein per serving" (Segmentation - microdosing trend, prevents waste, appeals to casual fitness)

Bad Contrarian ideas:
- "Just disagree with everything" (contrarian for its own sake)
- "Ignore all learnings" (not principled contradiction)

## OUTPUT FORMAT (JSON)
{{
    "title": "Concept name (concise, memorable)",
    "description": "2-3 sentences explaining what it is and why it matters",
    "differentiators": ["unique factor 1", "unique factor 2", "unique factor 3"],
    "target_market": "Specific target customer segment",
    "triz_principle_used": "Which TRIZ principle resolved the contradiction",
    "contradiction_resolved": "The contradiction this idea resolves",
    "conventions_challenged": ["pattern 1 inverted", "pattern 2 inverted"]
}}"""

        return {
            "mode": "contrarian",
            "agent": "contrarian-disruptor",
            "prompt": prompt,
            "objective": "challenge_assumptions",
            "framework": "TRIZ"
        }

    def _summarize_ideas(self, ideas: List[Dict], max_ideas: int = 10) -> str:
        """Create a concise summary of prior ideas."""
        if not ideas:
            return "No prior ideas yet."

        summaries = []
        for i, idea in enumerate(ideas[:max_ideas]):
            title = idea.get('title', f'Idea {i+1}')
            desc = idea.get('description', '')[:100]
            summaries.append(f"- {title}: {desc}...")

        return "\n".join(summaries)

    def _format_learnings(self, learnings: List[Dict]) -> str:
        """Format learnings for prompt injection."""
        if not learnings:
            return "No learnings accumulated yet."

        formatted = []
        for learning in learnings[:10]:
            pattern = learning.get('pattern', '')
            observation = learning.get('observation', '')
            formatted.append(f"- {pattern}: {observation}")

        return "\n".join(formatted)

    def _invert_learnings(self, learnings: List[Dict]) -> str:
        """Invert learnings for contrarian mode."""
        if not learnings:
            return "No patterns to invert yet."

        inversions = []
        inversion_prefixes = [
            "What if",
            "Consider that",
            "Explore the possibility that",
            "Challenge the assumption that"
        ]

        for i, learning in enumerate(learnings[:5]):
            pattern = learning.get('pattern', '')
            prefix = inversion_prefixes[i % len(inversion_prefixes)]

            # Simple inversion logic
            if "high" in pattern.lower():
                inverted = pattern.replace("high", "LOW").replace("High", "LOW")
            elif "low" in pattern.lower():
                inverted = pattern.replace("low", "HIGH").replace("Low", "HIGH")
            elif "work" in pattern.lower():
                inverted = f"the opposite of '{pattern}' might work better"
            elif "avoid" in pattern.lower():
                inverted = pattern.replace("avoid", "EMBRACE").replace("Avoid", "EMBRACE")
            else:
                inverted = f"the opposite of '{pattern}'"

            inversions.append(f"- {prefix} {inverted}")

        return "\n".join(inversions)

    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores for stagnation detection."""
        if not scores:
            return 0.0
        mean = sum(scores) / len(scores)
        return sum((x - mean) ** 2 for x in scores) / len(scores)

    def get_mode_distribution(self) -> Dict[str, float]:
        """Get the distribution of modes used so far."""
        total = sum(self.mode_stats.values())
        if total == 0:
            return {mode.value: 0.0 for mode in GeneratorMode}

        return {
            mode.value: count / total
            for mode, count in self.mode_stats.items()
        }

    def get_exploration_status(self) -> Dict:
        """Get current exploration status."""
        return {
            "exploration_budget": self.exploration_budget,
            "mode_distribution": self.get_mode_distribution(),
            "total_generations": len(self.mode_history),
            "last_5_modes": [m.value for m in self.mode_history[-5:]]
        }


# Convenience function for direct mode selection
def select_generator_mode(
    iteration: int,
    max_iterations: int = 30,
    recent_scores: Optional[List[float]] = None,
    is_plateau: bool = False
) -> str:
    """
    Quick helper to select a generator mode.

    Returns mode as string for easy use in orchestrator.
    """
    generator = TripleGenerator()
    mode = generator.select_mode(
        iteration=iteration,
        max_iterations=max_iterations,
        recent_scores=recent_scores or [],
        is_plateau=is_plateau
    )
    return mode.value
