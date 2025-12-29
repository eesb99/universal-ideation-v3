"""
Plateau Escape Protocol for Universal Ideation v3

DON'T STOP at plateau - attempt escape first.

When scores plateau (±0.5 over 10 iterations):
1. Generate 5 maximally divergent ideas (ignore ALL learnings)
2. If any escape idea scores > plateau average: continue with new direction
3. If all escape ideas score lower: confirm plateau, stop

Based on curiosity-driven exploration in RL research.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
import statistics


class PlateauStatus(Enum):
    """Status of plateau detection."""
    NO_PLATEAU = "no_plateau"
    PLATEAU_DETECTED = "plateau_detected"
    ESCAPE_ATTEMPTED = "escape_attempted"
    ESCAPE_SUCCESSFUL = "escape_successful"
    ESCAPE_FAILED = "escape_failed"
    CONFIRMED_PLATEAU = "confirmed_plateau"


@dataclass
class EscapeIdea:
    """An idea generated during escape attempt."""
    idea: Dict
    score: float
    strategy: str  # Which escape strategy was used
    deviation_from_centroid: float


@dataclass
class EscapeAttempt:
    """Record of an escape attempt."""
    iteration: int
    plateau_average: float
    escape_ideas: List[EscapeIdea]
    best_escape_score: float
    escaped: bool
    new_direction: Optional[str]


@dataclass
class PlateauAnalysis:
    """Analysis of score plateau."""
    is_plateau: bool
    recent_average: float
    previous_average: float
    score_variance: float
    plateau_duration: int  # How many iterations at plateau
    recommendation: str


class PlateauDetector:
    """
    Detects when ideation has hit a local optimum.

    Plateau = scores stabilize within threshold over window.
    """

    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 0.5,
        min_iterations: int = 15
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.min_iterations = min_iterations
        self.score_history: List[float] = []
        self.plateau_checks: List[PlateauAnalysis] = []

    def add_score(self, score: float):
        """Add a new score to history."""
        self.score_history.append(score)

    def detect(self) -> PlateauAnalysis:
        """
        Check if scores have plateaued.

        Returns:
            PlateauAnalysis with detection result
        """
        # Need enough history
        if len(self.score_history) < self.min_iterations:
            return PlateauAnalysis(
                is_plateau=False,
                recent_average=0,
                previous_average=0,
                score_variance=0,
                plateau_duration=0,
                recommendation="Continue: insufficient history"
            )

        if len(self.score_history) < self.window_size * 2:
            return PlateauAnalysis(
                is_plateau=False,
                recent_average=statistics.mean(self.score_history[-self.window_size:]),
                previous_average=0,
                score_variance=statistics.stdev(self.score_history[-self.window_size:]) if len(self.score_history) >= self.window_size else 0,
                plateau_duration=0,
                recommendation="Continue: need more iterations"
            )

        # Compare recent window to previous window
        recent = self.score_history[-self.window_size:]
        previous = self.score_history[-self.window_size*2:-self.window_size]

        recent_avg = statistics.mean(recent)
        previous_avg = statistics.mean(previous)
        recent_variance = statistics.stdev(recent) if len(recent) > 1 else 0

        # Plateau if difference is below threshold
        diff = abs(recent_avg - previous_avg)
        is_plateau = diff < self.threshold

        # Calculate plateau duration
        plateau_duration = 0
        if is_plateau:
            # Count back how long we've been in plateau
            for i in range(len(self.score_history) - self.window_size, 0, -1):
                check_window = self.score_history[i:i+self.window_size]
                if len(check_window) < self.window_size:
                    break
                check_avg = statistics.mean(check_window)
                if abs(check_avg - recent_avg) < self.threshold:
                    plateau_duration += 1
                else:
                    break

        # Generate recommendation
        if not is_plateau:
            recommendation = "Continue: scores still improving"
        elif plateau_duration < 3:
            recommendation = "Monitor: possible plateau forming"
        else:
            recommendation = "Escape: confirmed plateau, attempt breakout"

        analysis = PlateauAnalysis(
            is_plateau=is_plateau,
            recent_average=recent_avg,
            previous_average=previous_avg,
            score_variance=recent_variance,
            plateau_duration=plateau_duration,
            recommendation=recommendation
        )

        self.plateau_checks.append(analysis)
        return analysis

    def get_plateau_average(self) -> float:
        """Get average score at current plateau."""
        if len(self.score_history) < self.window_size:
            return statistics.mean(self.score_history) if self.score_history else 0
        return statistics.mean(self.score_history[-self.window_size:])

    def reset_after_escape(self):
        """Reset plateau detection after successful escape."""
        # Keep history but reset plateau checks
        self.plateau_checks = []


class EscapeStrategyGenerator:
    """
    Generates maximally divergent escape ideas.

    Strategies:
    1. Complete domain inversion
    2. Random domain injection
    3. Anti-learning application
    4. Format/channel disruption
    5. Target audience flip
    """

    ESCAPE_STRATEGIES = [
        {
            "name": "domain_inversion",
            "description": "Completely invert the domain assumptions",
            "prompt_modifier": """
ESCAPE STRATEGY: Domain Inversion

Invert EVERYTHING about the conventional approach:
- If the category is about performance → make it about relaxation
- If targeting young → target elderly
- If premium → make it ultra-budget
- If functional → make it experiential

Generate an idea that is the POLAR OPPOSITE of what's expected.
"""
        },
        {
            "name": "random_domain_injection",
            "description": "Inject concepts from completely unrelated domain",
            "prompt_modifier": """
ESCAPE STRATEGY: Random Domain Injection

Pick a completely unrelated industry and force a mashup:
- Fashion + nutrition = ?
- Gaming + supplements = ?
- Hospitality + protein = ?
- Art + health = ?

Generate an idea that NO ONE in this industry would ever think of.
"""
        },
        {
            "name": "anti_learning",
            "description": "Do the exact opposite of what learnings suggest",
            "prompt_modifier": """
ESCAPE STRATEGY: Anti-Learning

The system has learned certain patterns work. BREAK THEM ALL:
- If "sensory claims boost scores" → ignore sensory
- If "familiar proteins perform better" → use the most exotic protein
- If "market fit matters" → deliberately target a non-existent market

Generate an idea that violates EVERY learned pattern.
"""
        },
        {
            "name": "format_disruption",
            "description": "Completely change the product format/delivery",
            "prompt_modifier": """
ESCAPE STRATEGY: Format Disruption

Abandon the conventional format entirely:
- Not powder, not liquid, not bar
- Think: patch, spray, gummy, infused clothing, ambient experience
- The format IS the innovation

Generate an idea where the delivery mechanism is revolutionary.
"""
        },
        {
            "name": "audience_flip",
            "description": "Target the exact opposite audience",
            "prompt_modifier": """
ESCAPE STRATEGY: Audience Flip

Target someone who would NEVER buy this product:
- Anti-fitness people
- People who hate supplements
- Demographics excluded by convention
- Non-human consumers?

Generate an idea for the LEAST expected customer.
"""
        }
    ]

    def __init__(self):
        self.strategies_used: List[str] = []

    def get_escape_prompts(
        self,
        domain: str,
        learnings: List[str],
        prior_ideas_summary: str,
        num_attempts: int = 5
    ) -> List[Dict]:
        """
        Generate escape prompts for breakout attempts.

        Args:
            domain: Target domain
            learnings: Accumulated learnings to invert
            prior_ideas_summary: Summary of prior ideas to avoid
            num_attempts: Number of escape ideas to generate

        Returns:
            List of prompt configurations
        """
        prompts = []

        for i, strategy in enumerate(self.ESCAPE_STRATEGIES[:num_attempts]):
            prompt = f"""
PLATEAU ESCAPE ATTEMPT {i+1}/{num_attempts}

The ideation session has hit a plateau. All recent ideas score similarly.
Your mission: BREAK OUT of the local optimum.

DOMAIN: {domain}

{strategy['prompt_modifier']}

IGNORE these learnings (they led to the plateau):
{chr(10).join(f'- {l}' for l in learnings[:5])}

AVOID similarity to ALL prior ideas:
{prior_ideas_summary}

CONSTRAINTS: None
FEASIBILITY: Ignore for now
CONVENTION: Break every one

Generate the MOST UNEXPECTED, SURPRISING idea possible.
Think: What would a child suggest? An alien? Someone who knows NOTHING about {domain}?

Output a complete idea with:
- Title
- Description (2-3 sentences)
- Why this breaks the mold
- Target market (even if unconventional)
"""
            prompts.append({
                "strategy": strategy["name"],
                "description": strategy["description"],
                "prompt": prompt
            })

        return prompts


@dataclass
class EscapeResult:
    """Result of escape protocol execution."""
    status: PlateauStatus
    plateau_average: float
    escape_attempts: List[EscapeAttempt]
    best_escape_idea: Optional[EscapeIdea]
    new_direction: Optional[str]
    should_continue: bool
    recommendation: str


class PlateauEscapeProtocol:
    """
    Complete plateau escape protocol.

    Orchestrates detection, escape attempts, and decision-making.
    """

    def __init__(
        self,
        window_size: int = 10,
        threshold: float = 0.5,
        min_iterations: int = 15,
        max_escape_attempts: int = 2,
        escape_ideas_per_attempt: int = 5
    ):
        self.detector = PlateauDetector(window_size, threshold, min_iterations)
        self.strategy_generator = EscapeStrategyGenerator()
        self.max_escape_attempts = max_escape_attempts
        self.escape_ideas_per_attempt = escape_ideas_per_attempt

        self.escape_attempts: List[EscapeAttempt] = []
        self.status = PlateauStatus.NO_PLATEAU
        self.consecutive_plateau_checks = 0

    def add_score(self, score: float):
        """Add score to history."""
        self.detector.add_score(score)

    def check_plateau(self) -> PlateauAnalysis:
        """Check for plateau."""
        analysis = self.detector.detect()

        if analysis.is_plateau:
            self.consecutive_plateau_checks += 1
            if self.status == PlateauStatus.NO_PLATEAU:
                self.status = PlateauStatus.PLATEAU_DETECTED
        else:
            self.consecutive_plateau_checks = 0
            self.status = PlateauStatus.NO_PLATEAU

        return analysis

    def should_attempt_escape(self) -> bool:
        """Determine if escape should be attempted."""
        if self.status != PlateauStatus.PLATEAU_DETECTED:
            return False

        if len(self.escape_attempts) >= self.max_escape_attempts:
            return False

        # Require 3 consecutive plateau checks before escape
        return self.consecutive_plateau_checks >= 3

    def get_escape_prompts(
        self,
        domain: str,
        learnings: List[str],
        prior_ideas_summary: str
    ) -> List[Dict]:
        """Get prompts for escape attempt."""
        return self.strategy_generator.get_escape_prompts(
            domain, learnings, prior_ideas_summary,
            self.escape_ideas_per_attempt
        )

    def evaluate_escape(
        self,
        escape_ideas: List[EscapeIdea],
        iteration: int
    ) -> EscapeResult:
        """
        Evaluate escape attempt results.

        Args:
            escape_ideas: Ideas generated during escape
            iteration: Current iteration number

        Returns:
            EscapeResult with decision
        """
        plateau_avg = self.detector.get_plateau_average()

        # Find best escape idea
        best_escape = max(escape_ideas, key=lambda x: x.score) if escape_ideas else None
        best_score = best_escape.score if best_escape else 0

        # Record attempt
        attempt = EscapeAttempt(
            iteration=iteration,
            plateau_average=plateau_avg,
            escape_ideas=escape_ideas,
            best_escape_score=best_score,
            escaped=best_score > plateau_avg + 5,  # Need meaningful improvement
            new_direction=best_escape.strategy if best_escape and best_score > plateau_avg + 5 else None
        )
        self.escape_attempts.append(attempt)

        # Determine outcome
        if attempt.escaped:
            self.status = PlateauStatus.ESCAPE_SUCCESSFUL
            self.detector.reset_after_escape()
            self.consecutive_plateau_checks = 0

            return EscapeResult(
                status=PlateauStatus.ESCAPE_SUCCESSFUL,
                plateau_average=plateau_avg,
                escape_attempts=self.escape_attempts,
                best_escape_idea=best_escape,
                new_direction=best_escape.strategy,
                should_continue=True,
                recommendation=f"Escape successful! Best escape idea ({best_escape.strategy}) "
                              f"scored {best_score:.1f} vs plateau avg {plateau_avg:.1f}. "
                              f"Continue with new direction."
            )
        else:
            self.status = PlateauStatus.ESCAPE_FAILED

            # Check if we should try again or give up
            if len(self.escape_attempts) >= self.max_escape_attempts:
                self.status = PlateauStatus.CONFIRMED_PLATEAU
                return EscapeResult(
                    status=PlateauStatus.CONFIRMED_PLATEAU,
                    plateau_average=plateau_avg,
                    escape_attempts=self.escape_attempts,
                    best_escape_idea=best_escape,
                    new_direction=None,
                    should_continue=False,
                    recommendation=f"Plateau confirmed after {len(self.escape_attempts)} escape attempts. "
                                  f"Best escape score {best_score:.1f} did not exceed "
                                  f"plateau avg {plateau_avg:.1f} + 5. Session complete."
                )
            else:
                return EscapeResult(
                    status=PlateauStatus.ESCAPE_FAILED,
                    plateau_average=plateau_avg,
                    escape_attempts=self.escape_attempts,
                    best_escape_idea=best_escape,
                    new_direction=None,
                    should_continue=True,
                    recommendation=f"Escape attempt {len(self.escape_attempts)} failed. "
                                  f"Best score {best_score:.1f} < plateau + 5 ({plateau_avg + 5:.1f}). "
                                  f"Will try again."
                )

    def get_statistics(self) -> Dict:
        """Get escape protocol statistics."""
        return {
            "status": self.status.value,
            "total_escape_attempts": len(self.escape_attempts),
            "successful_escapes": sum(1 for a in self.escape_attempts if a.escaped),
            "plateau_average": self.detector.get_plateau_average(),
            "score_history_length": len(self.detector.score_history),
            "consecutive_plateau_checks": self.consecutive_plateau_checks,
            "strategies_attempted": [a.new_direction for a in self.escape_attempts if a.new_direction]
        }


# Convenience functions

def create_escape_protocol(
    conservative: bool = False
) -> PlateauEscapeProtocol:
    """
    Create escape protocol with preset configuration.

    Args:
        conservative: If True, use stricter plateau detection

    Returns:
        Configured PlateauEscapeProtocol
    """
    if conservative:
        return PlateauEscapeProtocol(
            window_size=15,
            threshold=0.3,
            min_iterations=20,
            max_escape_attempts=3,
            escape_ideas_per_attempt=5
        )
    else:
        return PlateauEscapeProtocol(
            window_size=10,
            threshold=0.5,
            min_iterations=15,
            max_escape_attempts=2,
            escape_ideas_per_attempt=5
        )


def simulate_plateau_detection(scores: List[float]) -> PlateauAnalysis:
    """
    Quick helper to test plateau detection on a score list.

    Args:
        scores: List of scores to analyze

    Returns:
        PlateauAnalysis result
    """
    detector = PlateauDetector()
    for score in scores:
        detector.add_score(score)
    return detector.detect()
