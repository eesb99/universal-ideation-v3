"""
Data Models for Universal Interview
Structured context schema with typed fields and validation
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import json


class TemplateScaffold(Enum):
    """Pre-built constraint templates."""
    BOOTSTRAP = "bootstrap"       # Budget-constrained startup
    ENTERPRISE = "enterprise"     # Corporate context
    REGULATED = "regulated"       # Healthcare, finance, etc.
    SUSTAINABLE = "sustainable"   # Environmental focus
    CUSTOM = "custom"            # User-defined


@dataclass
class TemplateDefaults:
    """Default values for each template scaffold."""

    TEMPLATES = {
        TemplateScaffold.BOOTSTRAP: {
            "constraints": {
                "budget_range": "<$50k",
                "timeline": "3-6 months",
                "team_size": "1-5 people",
                "focus": "MVP, speed-to-market"
            },
            "suggested_prompts": [
                "What's the absolute minimum you need to prove the concept?",
                "If you had to launch in 30 days, what would you cut?"
            ]
        },
        TemplateScaffold.ENTERPRISE: {
            "constraints": {
                "requirements": ["scalability", "compliance", "integration", "security"],
                "timeline": "6-18 months",
                "stakeholders": "multiple departments"
            },
            "suggested_prompts": [
                "What existing systems does this need to integrate with?",
                "Who are the key stakeholders that need to sign off?"
            ]
        },
        TemplateScaffold.REGULATED: {
            "constraints": {
                "regulatory_bodies": ["FDA", "SEC", "HIPAA", "GDPR"],
                "documentation": "extensive",
                "safety": "safety-first approach"
            },
            "suggested_prompts": [
                "What regulatory pathway are you considering?",
                "Have you consulted with regulatory affairs specialists?"
            ]
        },
        TemplateScaffold.SUSTAINABLE: {
            "constraints": {
                "principles": ["circular economy", "ethical sourcing", "carbon neutral"],
                "certifications": "B-Corp, organic, fair trade",
                "stakeholders": "environmental, social"
            },
            "suggested_prompts": [
                "How do you measure environmental impact?",
                "What sustainability certifications are you targeting?"
            ]
        }
    }

    @classmethod
    def get_defaults(cls, template: TemplateScaffold) -> Dict:
        """Get default values for a template."""
        return cls.TEMPLATES.get(template, {})


@dataclass
class DimensionResponse:
    """Response data for a single interview dimension."""
    response: str = ""
    confidence: str = "medium"  # high, medium, low, unknown
    source: str = "user"        # user, injected, inferred
    response_count: int = 0
    last_updated: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'DimensionResponse':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InterviewContext:
    """Complete interview context for ideation."""

    # Initiative metadata
    initiative_id: str = ""
    initiative_name: str = ""
    status: str = "draft"
    created_at: str = ""
    last_updated: str = ""

    # Domain information
    original_domain: str = ""
    enriched_domain: str = ""

    # Template
    template_scaffold: Optional[str] = None

    # 7 Dimensions
    problem_space: DimensionResponse = field(default_factory=DimensionResponse)
    constraints: DimensionResponse = field(default_factory=DimensionResponse)
    assumptions: DimensionResponse = field(default_factory=DimensionResponse)
    intent: DimensionResponse = field(default_factory=DimensionResponse)
    preferences: DimensionResponse = field(default_factory=DimensionResponse)
    existing_solutions: DimensionResponse = field(default_factory=DimensionResponse)
    resources: DimensionResponse = field(default_factory=DimensionResponse)

    # Gaps and attributions
    gaps_flagged: List[str] = field(default_factory=list)
    source_attributions: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'initiative_id': self.initiative_id,
            'initiative_name': self.initiative_name,
            'status': self.status,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'original_domain': self.original_domain,
            'enriched_domain': self.enriched_domain,
            'template_scaffold': self.template_scaffold,
            'dimensions': {
                'problem_space': self.problem_space.to_dict(),
                'constraints': self.constraints.to_dict(),
                'assumptions': self.assumptions.to_dict(),
                'intent': self.intent.to_dict(),
                'preferences': self.preferences.to_dict(),
                'existing_solutions': self.existing_solutions.to_dict(),
                'resources': self.resources.to_dict()
            },
            'gaps_flagged': self.gaps_flagged,
            'source_attributions': self.source_attributions
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict) -> 'InterviewContext':
        """Create from dictionary."""
        dimensions = data.get('dimensions', {})

        return cls(
            initiative_id=data.get('initiative_id', ''),
            initiative_name=data.get('initiative_name', ''),
            status=data.get('status', 'draft'),
            created_at=data.get('created_at', ''),
            last_updated=data.get('last_updated', ''),
            original_domain=data.get('original_domain', ''),
            enriched_domain=data.get('enriched_domain', ''),
            template_scaffold=data.get('template_scaffold'),
            problem_space=DimensionResponse.from_dict(dimensions.get('problem_space', {})),
            constraints=DimensionResponse.from_dict(dimensions.get('constraints', {})),
            assumptions=DimensionResponse.from_dict(dimensions.get('assumptions', {})),
            intent=DimensionResponse.from_dict(dimensions.get('intent', {})),
            preferences=DimensionResponse.from_dict(dimensions.get('preferences', {})),
            existing_solutions=DimensionResponse.from_dict(dimensions.get('existing_solutions', {})),
            resources=DimensionResponse.from_dict(dimensions.get('resources', {})),
            gaps_flagged=data.get('gaps_flagged', []),
            source_attributions=data.get('source_attributions', {})
        )

    def get_covered_dimensions(self) -> List[str]:
        """Get list of dimensions with responses."""
        covered = []
        for dim_name in ['problem_space', 'constraints', 'assumptions', 'intent',
                         'preferences', 'existing_solutions', 'resources']:
            dim = getattr(self, dim_name)
            if dim.response:
                covered.append(dim_name)
        return covered

    def get_coverage_percentage(self) -> float:
        """Get percentage of dimensions covered."""
        covered = len(self.get_covered_dimensions())
        return (covered / 7) * 100

    def get_high_confidence_dimensions(self) -> List[str]:
        """Get dimensions with high confidence responses."""
        high_conf = []
        for dim_name in ['problem_space', 'constraints', 'assumptions', 'intent',
                         'preferences', 'existing_solutions', 'resources']:
            dim = getattr(self, dim_name)
            if dim.response and dim.confidence == 'high':
                high_conf.append(dim_name)
        return high_conf

    def get_prompt_context(self) -> str:
        """Generate context string for LLM prompts."""
        lines = []

        if self.enriched_domain:
            lines.append(f"Domain: {self.enriched_domain}")
        elif self.original_domain:
            lines.append(f"Domain: {self.original_domain}")

        lines.append("")
        lines.append("Context from interview:")

        for dim_name, dim_title in [
            ('problem_space', 'Problem Space'),
            ('constraints', 'Constraints'),
            ('assumptions', 'Assumptions'),
            ('intent', 'Strategic Intent'),
            ('preferences', 'Preferences'),
            ('existing_solutions', 'Existing Solutions'),
            ('resources', 'Resources')
        ]:
            dim = getattr(self, dim_name)
            if dim.response:
                conf_marker = f" [{dim.confidence}]" if dim.confidence != 'high' else ""
                lines.append(f"- {dim_title}{conf_marker}: {dim.response}")

        if self.gaps_flagged:
            lines.append("")
            lines.append("Knowledge gaps to explore:")
            for gap in self.gaps_flagged:
                lines.append(f"- {gap}")

        return '\n'.join(lines)


@dataclass
class EngagementMetrics:
    """Metrics for tracking user engagement during interview."""
    response_lengths: List[int] = field(default_factory=list)
    response_times: List[float] = field(default_factory=list)  # seconds
    elaboration_count: int = 0  # times user gave detailed answer
    vague_response_count: int = 0  # "I don't know", short answers
    fatigue_signals: int = 0  # "let's move on", etc.
    dimension_switch_count: int = 0  # how many times topic changed

    def add_response(self, response: str, time_taken: float = 0):
        """Record a response for engagement tracking."""
        self.response_lengths.append(len(response))
        if time_taken > 0:
            self.response_times.append(time_taken)

        # Classify response
        if len(response) < 20 or response.lower() in ['i don\'t know', 'not sure', 'no idea', 'skip']:
            self.vague_response_count += 1
        elif len(response) > 100:
            self.elaboration_count += 1

        # Check for fatigue signals
        fatigue_phrases = ['let\'s move on', 'just generate', 'skip', 'that\'s enough', 'i\'m ready']
        if any(phrase in response.lower() for phrase in fatigue_phrases):
            self.fatigue_signals += 1

    def get_average_response_length(self) -> float:
        """Get average response length."""
        if not self.response_lengths:
            return 0
        return sum(self.response_lengths) / len(self.response_lengths)

    def get_response_length_trend(self) -> str:
        """Detect if response lengths are declining (fatigue indicator)."""
        if len(self.response_lengths) < 3:
            return "insufficient_data"

        # Compare first half to second half
        mid = len(self.response_lengths) // 2
        first_half_avg = sum(self.response_lengths[:mid]) / mid
        second_half_avg = sum(self.response_lengths[mid:]) / (len(self.response_lengths) - mid)

        if second_half_avg < first_half_avg * 0.7:
            return "declining"
        elif second_half_avg > first_half_avg * 1.3:
            return "increasing"
        else:
            return "stable"

    def should_wrap_up(self) -> bool:
        """Determine if interview should wrap up based on engagement."""
        # Strong signals
        if self.fatigue_signals >= 2:
            return True

        # Multiple weak signals
        weak_signals = 0
        if self.vague_response_count >= 3:
            weak_signals += 1
        if self.get_response_length_trend() == "declining":
            weak_signals += 1
        if len(self.response_lengths) > 10 and self.get_average_response_length() < 30:
            weak_signals += 1

        return weak_signals >= 2

    def to_dict(self) -> Dict:
        return {
            'response_count': len(self.response_lengths),
            'avg_response_length': self.get_average_response_length(),
            'response_length_trend': self.get_response_length_trend(),
            'elaboration_count': self.elaboration_count,
            'vague_response_count': self.vague_response_count,
            'fatigue_signals': self.fatigue_signals,
            'should_wrap_up': self.should_wrap_up()
        }


@dataclass
class InterviewProgress:
    """Track progress through the interview."""
    total_dimensions: int = 7
    dimensions_touched: List[str] = field(default_factory=list)
    current_dimension: Optional[str] = None
    questions_asked: int = 0
    follow_ups_asked: int = 0

    def touch_dimension(self, dimension: str):
        """Mark a dimension as touched."""
        if dimension not in self.dimensions_touched:
            self.dimensions_touched.append(dimension)

    def get_progress_percentage(self) -> float:
        """Get progress as percentage."""
        return (len(self.dimensions_touched) / self.total_dimensions) * 100

    def get_remaining_dimensions(self) -> List[str]:
        """Get dimensions not yet touched."""
        all_dims = ['problem_space', 'constraints', 'assumptions', 'intent',
                    'preferences', 'existing_solutions', 'resources']
        return [d for d in all_dims if d not in self.dimensions_touched]

    def get_progress_message(self) -> str:
        """Get human-readable progress message."""
        touched = len(self.dimensions_touched)
        remaining = self.total_dimensions - touched

        if touched == 0:
            return "We're just getting started."
        elif remaining == 0:
            return "We've covered all the key areas."
        elif remaining == 1:
            return "Almost there - just one more area to explore."
        elif remaining <= 3:
            return f"Good progress - {remaining} more areas to touch on."
        else:
            return f"We've covered {touched} of {self.total_dimensions} areas."

    def to_dict(self) -> Dict:
        return {
            'dimensions_touched': self.dimensions_touched,
            'current_dimension': self.current_dimension,
            'questions_asked': self.questions_asked,
            'follow_ups_asked': self.follow_ups_asked,
            'progress_percentage': self.get_progress_percentage(),
            'remaining_dimensions': self.get_remaining_dimensions()
        }


@dataclass
class InterviewState:
    """Complete state of an in-progress interview."""
    session_id: str = ""
    initiative_id: str = ""
    context: InterviewContext = field(default_factory=InterviewContext)
    progress: InterviewProgress = field(default_factory=InterviewProgress)
    engagement: EngagementMetrics = field(default_factory=EngagementMetrics)
    is_complete: bool = False
    interruption_point: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'session_id': self.session_id,
            'initiative_id': self.initiative_id,
            'context': self.context.to_dict(),
            'progress': self.progress.to_dict(),
            'engagement': self.engagement.to_dict(),
            'is_complete': self.is_complete,
            'interruption_point': self.interruption_point
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def should_complete(self) -> bool:
        """Determine if interview should complete."""
        # All dimensions covered
        if self.progress.get_progress_percentage() >= 100:
            return True

        # Engagement signals wrap-up
        if self.engagement.should_wrap_up():
            return True

        return False

    def get_next_dimension_suggestion(self) -> Optional[str]:
        """Suggest next dimension to explore."""
        remaining = self.progress.get_remaining_dimensions()
        if not remaining:
            return None

        # Priority order (most important first)
        priority_order = ['problem_space', 'intent', 'constraints',
                         'existing_solutions', 'preferences', 'resources', 'assumptions']

        for dim in priority_order:
            if dim in remaining:
                return dim

        return remaining[0] if remaining else None
