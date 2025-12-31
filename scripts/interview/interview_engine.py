"""
Interview Engine for Universal Interview
Core interview logic with 7-dimension question flow and adaptive behavior
"""

from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import random

from .interview_storage import (
    InterviewStorage,
    InterviewDimension,
    ConfidenceLevel,
    ResponseSource,
    InitiativeStatus
)
from .models import (
    InterviewContext,
    InterviewState,
    InterviewProgress,
    EngagementMetrics,
    TemplateScaffold,
    TemplateDefaults,
    DimensionResponse
)


@dataclass
class Question:
    """A question to ask during the interview."""
    text: str
    dimension: InterviewDimension
    is_base: bool = True  # Base question vs follow-up
    follow_up_depth: int = 0
    metadata: Dict = field(default_factory=dict)


class QuestionBank:
    """Repository of base questions for each dimension."""

    # Curious collaborator tone, open-ended questions
    BASE_QUESTIONS = {
        InterviewDimension.PROBLEM_SPACE: [
            "I'm curious - what frustration or observation sparked this exploration?",
            "Who experiences this problem most acutely? Paint me a picture of their day.",
            "If this problem disappeared tomorrow, what would change for people?",
            "What makes this problem interesting to you personally?",
        ],
        InterviewDimension.CONSTRAINTS: [
            "What boundaries are you working within? Budget, timeline, resources?",
            "Are there any non-negotiables or hard limits I should know about?",
            "What regulatory or compliance considerations come into play here?",
            "If you had unlimited resources, what would still be a constraint?",
        ],
        InterviewDimension.ASSUMPTIONS: [
            "What do you take for granted about this space that might be worth questioning?",
            "What does 'everyone know' about this problem that might not actually be true?",
            "If you were wrong about one thing in your mental model, what would it be?",
            "What surprised you when you first dug into this domain?",
        ],
        InterviewDimension.INTENT: [
            "What does success look like for you in 2 years? 5 years?",
            "Are you looking to disrupt, defend, optimize, or explore entirely new territory?",
            "How does this fit into your broader vision or strategy?",
            "What would make you say 'this was worth it' at the end?",
        ],
        InterviewDimension.PREFERENCES: [
            "What kind of ideas energize you versus bore you?",
            "Are there approaches or types of solutions you'd rather avoid?",
            "Do you prefer incremental improvements or breakthrough disruptions?",
            "What's an example of innovation you admire in any field?",
        ],
        InterviewDimension.EXISTING_SOLUTIONS: [
            "What's already been tried in this space, by you or others?",
            "Who are the key players or competitors addressing this?",
            "What works well about existing solutions? What's frustrating about them?",
            "Where do current approaches fall short?",
        ],
        InterviewDimension.RESOURCES: [
            "What unfair advantages or unique assets do you bring to this?",
            "What capabilities, relationships, or infrastructure do you already have?",
            "Who could you partner with or leverage?",
            "What's in your toolkit that others might not have?",
        ],
    }

    # Follow-up question templates (used with LLM to generate context-aware follow-ups)
    FOLLOW_UP_TEMPLATES = {
        InterviewDimension.PROBLEM_SPACE: [
            "You mentioned {key_point}. Can you tell me more about that?",
            "That's interesting about {key_point}. How widespread is this?",
            "When you say {key_point}, what's the underlying cause?",
        ],
        InterviewDimension.CONSTRAINTS: [
            "The {key_point} constraint is notable. How firm is that?",
            "If {key_point} changed, what would become possible?",
            "What's driving the {key_point} constraint?",
        ],
        InterviewDimension.ASSUMPTIONS: [
            "You assume {key_point}. What if that weren't true?",
            "Interesting assumption about {key_point}. Where does that come from?",
            "If {key_point} was wrong, what would change?",
        ],
        InterviewDimension.INTENT: [
            "When you say {key_point}, what would that look like concretely?",
            "How would you measure progress toward {key_point}?",
            "What's the biggest obstacle to achieving {key_point}?",
        ],
        InterviewDimension.PREFERENCES: [
            "You prefer {key_point}. What draws you to that?",
            "When you say {key_point} bores you, what specifically?",
            "Can you give an example of {key_point} done well?",
        ],
        InterviewDimension.EXISTING_SOLUTIONS: [
            "You mentioned {key_point}. Why do you think they approached it that way?",
            "What would it take to do better than {key_point}?",
            "Are there lessons from {key_point} to apply here?",
        ],
        InterviewDimension.RESOURCES: [
            "The {key_point} capability is interesting. How could that be leveraged?",
            "Who else knows about your {key_point}?",
            "How defensible is the {key_point} advantage?",
        ],
    }

    @classmethod
    def get_base_question(cls, dimension: InterviewDimension, exclude: List[str] = None) -> Question:
        """Get a base question for a dimension, avoiding already-asked ones."""
        questions = cls.BASE_QUESTIONS.get(dimension, [])
        exclude = exclude or []

        available = [q for q in questions if q not in exclude]
        if not available:
            available = questions  # Reset if all used

        text = random.choice(available) if available else questions[0]
        return Question(text=text, dimension=dimension, is_base=True)

    @classmethod
    def get_follow_up_template(cls, dimension: InterviewDimension) -> str:
        """Get a follow-up template for a dimension."""
        templates = cls.FOLLOW_UP_TEMPLATES.get(dimension, [])
        return random.choice(templates) if templates else "Tell me more about {key_point}."


class InterviewEngine:
    """Core engine that conducts adaptive interviews."""

    def __init__(
        self,
        storage: InterviewStorage = None,
        llm_callback: Callable[[str], str] = None
    ):
        """
        Initialize the interview engine.

        Args:
            storage: InterviewStorage instance for persistence
            llm_callback: Optional callback for LLM-generated follow-ups
                         Signature: (prompt: str) -> response: str
        """
        self.storage = storage or InterviewStorage()
        self.llm_callback = llm_callback
        self.question_bank = QuestionBank()

        # Track asked questions per session
        self._asked_questions: Dict[str, List[str]] = {}

    # ===================
    # Session Management
    # ===================

    def start_interview(
        self,
        domain: str,
        initiative_name: str = None,
        template: TemplateScaffold = None
    ) -> InterviewState:
        """Start a new interview session."""
        # Create initiative
        name = initiative_name or f"Initiative: {domain[:50]}"
        template_name = template.value if template else None

        initiative_id = self.storage.create_initiative(
            name=name,
            original_domain=domain,
            template_scaffold=template_name
        )

        # Create session
        session_id = self.storage.create_session(initiative_id)

        # Initialize state
        state = InterviewState(
            session_id=session_id,
            initiative_id=initiative_id,
            context=InterviewContext(
                initiative_id=initiative_id,
                initiative_name=name,
                original_domain=domain,
                template_scaffold=template_name
            ),
            progress=InterviewProgress(),
            engagement=EngagementMetrics()
        )

        # Apply template defaults if specified
        if template:
            self._apply_template_defaults(state, template)

        self._asked_questions[session_id] = []
        return state

    def resume_interview(self, initiative_id: str) -> Optional[InterviewState]:
        """Resume an incomplete interview session."""
        # Get initiative
        initiative = self.storage.get_initiative(initiative_id)
        if not initiative:
            return None

        # Get latest incomplete session
        sessions = self.storage.get_incomplete_sessions(initiative_id)
        if not sessions:
            # No incomplete session - create new one
            session_id = self.storage.create_session(initiative_id)
            session = self.storage.get_session(session_id)
        else:
            session = sessions[0]
            session_id = session['id']

        # Build context from stored data
        context_dict = self.storage.build_context(initiative_id)
        context = InterviewContext.from_dict(context_dict)

        # Rebuild progress
        dimensions_covered = self.storage.get_dimensions_covered(initiative_id)
        responses = self.storage.get_all_responses(initiative_id)

        progress = InterviewProgress(
            dimensions_touched=dimensions_covered,
            questions_asked=sum(len(r) for r in responses.values())
        )

        # Create state (engagement metrics start fresh on resume)
        state = InterviewState(
            session_id=session_id,
            initiative_id=initiative_id,
            context=context,
            progress=progress,
            engagement=EngagementMetrics(),
            interruption_point=session.get('interruption_point')
        )

        self._asked_questions[session_id] = []
        return state

    def save_interruption(self, state: InterviewState, reason: str = None):
        """Save interview state when interrupted."""
        # Update session
        self.storage.update_session(
            state.session_id,
            interruption_point=state.progress.current_dimension,
            dimensions_covered=state.progress.dimensions_touched
        )

        # Update initiative status
        self.storage.update_initiative_status(
            state.initiative_id,
            InitiativeStatus.DRAFT
        )

    def complete_interview(self, state: InterviewState) -> InterviewContext:
        """Mark interview as complete and finalize context."""
        # Generate enriched domain
        enriched_domain = self._synthesize_enriched_domain(state)

        # Update initiative
        self.storage.update_initiative(
            state.initiative_id,
            enriched_domain=enriched_domain,
            status=InitiativeStatus.READY.value
        )

        # Mark session complete
        self.storage.update_session(
            state.session_id,
            is_complete=1,
            dimensions_covered=state.progress.dimensions_touched
        )

        # Update context
        state.context.enriched_domain = enriched_domain
        state.context.status = InitiativeStatus.READY.value
        state.is_complete = True

        return state.context

    # ===================
    # Question Flow
    # ===================

    def get_next_question(self, state: InterviewState) -> Optional[Question]:
        """Get the next question to ask based on current state."""
        # Check if should wrap up
        if state.engagement.should_wrap_up():
            return None

        # Check if all dimensions covered
        if state.progress.get_progress_percentage() >= 100:
            return None

        # Get next dimension to explore
        next_dim = self._select_next_dimension(state)
        if not next_dim:
            return None

        # Get question for dimension
        asked = self._asked_questions.get(state.session_id, [])
        question = QuestionBank.get_base_question(next_dim, exclude=asked)

        # Update tracking
        state.progress.current_dimension = next_dim.value
        asked.append(question.text)
        self._asked_questions[state.session_id] = asked

        return question

    def get_follow_up_question(
        self,
        state: InterviewState,
        last_response: str,
        dimension: InterviewDimension
    ) -> Optional[Question]:
        """Generate a follow-up question based on response."""
        # Check if should continue probing this dimension
        dim_responses = self.storage.get_responses_by_dimension(
            state.initiative_id, dimension
        )
        if len(dim_responses) >= 3:  # Max 3 follow-ups per dimension
            return None

        # Check engagement
        if state.engagement.should_wrap_up():
            return None

        # Generate follow-up
        if self.llm_callback:
            follow_up_text = self._generate_llm_follow_up(
                state, last_response, dimension
            )
        else:
            # Use template with key point extraction (simple heuristic)
            key_point = self._extract_key_point(last_response)
            template = QuestionBank.get_follow_up_template(dimension)
            follow_up_text = template.format(key_point=key_point)

        return Question(
            text=follow_up_text,
            dimension=dimension,
            is_base=False,
            follow_up_depth=len(dim_responses)
        )

    def _select_next_dimension(self, state: InterviewState) -> Optional[InterviewDimension]:
        """Select the next dimension to explore."""
        remaining = state.progress.get_remaining_dimensions()
        if not remaining:
            return None

        # Priority order based on importance and energy-following
        priority = [
            'problem_space',  # Start with problem understanding
            'intent',         # Then strategic direction
            'constraints',    # Practical boundaries
            'existing_solutions',  # Competitive landscape
            'preferences',    # Personal fit
            'resources',      # Available assets
            'assumptions',    # Hidden beliefs (often need more rapport)
        ]

        # Find first priority dimension that's remaining
        for dim_name in priority:
            if dim_name in remaining:
                return InterviewDimension(dim_name)

        # Fallback: first remaining
        return InterviewDimension(remaining[0])

    def _extract_key_point(self, response: str) -> str:
        """Extract a key point from response for follow-up."""
        # Simple heuristic: first clause or first 50 chars
        if '.' in response:
            first_sentence = response.split('.')[0]
            if len(first_sentence) > 10:
                return first_sentence[:50] + ('...' if len(first_sentence) > 50 else '')

        words = response.split()[:8]
        return ' '.join(words) + ('...' if len(words) >= 8 else '')

    def _generate_llm_follow_up(
        self,
        state: InterviewState,
        last_response: str,
        dimension: InterviewDimension
    ) -> str:
        """Generate follow-up using LLM."""
        if not self.llm_callback:
            return QuestionBank.get_follow_up_template(dimension).format(
                key_point=self._extract_key_point(last_response)
            )

        prompt = f"""You are a curious collaborator conducting an interview about: {state.context.original_domain}

The user just said (regarding {dimension.value.replace('_', ' ')}):
"{last_response}"

Generate ONE follow-up question that:
1. Builds on what they said
2. Uses curious, collaborative tone (e.g., "I'm curious...", "That's interesting...")
3. Is open-ended, not yes/no
4. Digs deeper into their specific point
5. Is concise (under 25 words)

Respond with just the question, nothing else."""

        try:
            follow_up = self.llm_callback(prompt)
            return follow_up.strip()
        except Exception:
            # Fallback to template
            return QuestionBank.get_follow_up_template(dimension).format(
                key_point=self._extract_key_point(last_response)
            )

    # ===================
    # Response Processing
    # ===================

    def process_response(
        self,
        state: InterviewState,
        question: Question,
        response: str,
        response_time: float = 0
    ) -> Tuple[bool, Optional[str]]:
        """
        Process a user response.

        Returns:
            Tuple of (should_continue, optional_system_message)
        """
        # Update engagement metrics
        state.engagement.add_response(response, response_time)

        # Determine confidence from response
        confidence = self._assess_confidence(response)

        # Store response
        self.storage.store_response(
            session_id=state.session_id,
            initiative_id=state.initiative_id,
            dimension=question.dimension,
            question=question.text,
            response=response,
            confidence=confidence,
            source=ResponseSource.USER,
            follow_up_depth=question.follow_up_depth
        )

        # Update progress
        state.progress.touch_dimension(question.dimension.value)
        state.progress.questions_asked += 1

        # Update context dimension
        self._update_context_dimension(state, question.dimension, response, confidence)

        # Check for knowledge gaps
        if confidence == ConfidenceLevel.LOW or self._is_uncertain_response(response):
            gap_desc = f"User uncertain about {question.dimension.value.replace('_', ' ')}: {self._extract_key_point(response)}"
            self.storage.flag_gap(
                state.initiative_id,
                question.dimension,
                gap_desc
            )

        # Add attribution
        self.storage.add_attribution(
            state.initiative_id,
            question.dimension,
            ResponseSource.USER
        )

        # Check if should continue
        should_continue = not state.should_complete()

        # Generate progress message
        progress_msg = None
        if should_continue:
            progress_msg = state.progress.get_progress_message()

        return should_continue, progress_msg

    def _assess_confidence(self, response: str) -> ConfidenceLevel:
        """Assess confidence level from response text."""
        low_confidence_signals = [
            'not sure', 'don\'t know', 'maybe', 'perhaps', 'i think',
            'probably', 'might', 'uncertain', 'guess', 'assume'
        ]

        high_confidence_signals = [
            'definitely', 'absolutely', 'certain', 'know for sure',
            'without doubt', 'clearly', 'obviously'
        ]

        response_lower = response.lower()

        for signal in high_confidence_signals:
            if signal in response_lower:
                return ConfidenceLevel.HIGH

        for signal in low_confidence_signals:
            if signal in response_lower:
                return ConfidenceLevel.LOW

        # Default based on response length and specificity
        if len(response) > 100:
            return ConfidenceLevel.HIGH
        elif len(response) < 30:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.MEDIUM

    def _is_uncertain_response(self, response: str) -> bool:
        """Check if response indicates uncertainty."""
        uncertain_phrases = [
            'i don\'t know', 'not sure', 'no idea', 'haven\'t thought',
            'good question', 'need to research', 'unclear'
        ]
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in uncertain_phrases)

    def _update_context_dimension(
        self,
        state: InterviewState,
        dimension: InterviewDimension,
        response: str,
        confidence: ConfidenceLevel
    ):
        """Update the context with dimension response."""
        dim_attr = dimension.value
        existing = getattr(state.context, dim_attr)

        # Combine with existing response if any
        if existing.response:
            combined = f"{existing.response} {response}"
        else:
            combined = response

        new_dim = DimensionResponse(
            response=combined,
            confidence=confidence.value,
            source='user',
            response_count=existing.response_count + 1
        )

        setattr(state.context, dim_attr, new_dim)

    # ===================
    # Template & Scaffolding
    # ===================

    def _apply_template_defaults(self, state: InterviewState, template: TemplateScaffold):
        """Apply template defaults to state."""
        defaults = TemplateDefaults.get_defaults(template)
        if not defaults:
            return

        # Pre-fill constraints dimension with template defaults
        constraints_defaults = defaults.get('constraints', {})
        if constraints_defaults:
            constraint_text = ', '.join(
                f"{k}: {v}" for k, v in constraints_defaults.items()
            )
            state.context.constraints = DimensionResponse(
                response=f"Template defaults ({template.value}): {constraint_text}",
                confidence='medium',
                source='inferred',
                response_count=0
            )

    def detect_template(self, domain: str, responses: Dict[str, str] = None) -> Optional[TemplateScaffold]:
        """Detect which template scaffold might apply."""
        domain_lower = domain.lower()
        responses_text = ' '.join((responses or {}).values()).lower()
        combined = f"{domain_lower} {responses_text}"

        # Simple keyword matching
        if any(kw in combined for kw in ['startup', 'bootstrap', 'mvp', 'lean', 'limited budget']):
            return TemplateScaffold.BOOTSTRAP

        if any(kw in combined for kw in ['enterprise', 'corporate', 'b2b', 'integration', 'compliance']):
            return TemplateScaffold.ENTERPRISE

        if any(kw in combined for kw in ['healthcare', 'medical', 'fda', 'regulated', 'pharma', 'finance', 'fintech']):
            return TemplateScaffold.REGULATED

        if any(kw in combined for kw in ['sustainable', 'green', 'eco', 'circular', 'environmental', 'climate']):
            return TemplateScaffold.SUSTAINABLE

        return None

    # ===================
    # Context Synthesis
    # ===================

    def _synthesize_enriched_domain(self, state: InterviewState) -> str:
        """Synthesize enriched domain statement from interview responses."""
        if self.llm_callback:
            return self._synthesize_with_llm(state)
        else:
            return self._synthesize_basic(state)

    def _synthesize_basic(self, state: InterviewState) -> str:
        """Basic synthesis without LLM."""
        parts = [state.context.original_domain]

        if state.context.problem_space.response:
            parts.append(f"Problem: {state.context.problem_space.response[:100]}")

        if state.context.intent.response:
            parts.append(f"Intent: {state.context.intent.response[:100]}")

        if state.context.constraints.response:
            parts.append(f"Constraints: {state.context.constraints.response[:100]}")

        return ' | '.join(parts)

    def _synthesize_with_llm(self, state: InterviewState) -> str:
        """Synthesize using LLM for natural language."""
        if not self.llm_callback:
            return self._synthesize_basic(state)

        context_text = state.context.get_prompt_context()

        prompt = f"""Based on this interview context, synthesize a rich, coherent domain description in 2-3 sentences.
This will be used to guide idea generation.

{context_text}

Write a clear, actionable domain description that captures:
1. The core problem/opportunity
2. Key constraints
3. Strategic intent
4. Target context

Keep it under 100 words. Be specific, not generic."""

        try:
            enriched = self.llm_callback(prompt)
            return enriched.strip()
        except Exception:
            return self._synthesize_basic(state)

    # ===================
    # Similarity Detection
    # ===================

    def find_similar_initiatives(self, domain: str, threshold: float = 0.7) -> List[Dict]:
        """Find initiatives similar to the given domain."""
        return self.storage.find_similar_initiatives(domain, threshold)

    def check_continuation(self, domain: str) -> Tuple[bool, Optional[Dict]]:
        """
        Check if domain matches an existing initiative.

        Returns:
            Tuple of (found_similar, similar_initiative_or_none)
        """
        similar = self.find_similar_initiatives(domain, threshold=0.75)
        if similar:
            return True, similar[0]
        return False, None

    # ===================
    # Progress Messaging
    # ===================

    def get_welcome_message(self, state: InterviewState) -> str:
        """Get welcome message for interview start."""
        template_note = ""
        if state.context.template_scaffold:
            template_note = f" I've applied the {state.context.template_scaffold.upper()} template as a starting scaffold."

        return f"""Let's explore your domain: "{state.context.original_domain}"{template_note}

I'll ask questions across several areas to understand your context deeply. This helps generate better, more relevant ideas.

Feel free to elaborate - the richer your answers, the better the ideas. If something isn't relevant, just say so and we'll move on.

{state.progress.get_progress_message()}"""

    def get_completion_message(self, state: InterviewState) -> str:
        """Get completion message for interview end."""
        coverage = state.progress.get_progress_percentage()
        gaps = self.storage.get_gaps(state.initiative_id)

        gap_note = ""
        if gaps:
            gap_note = f"\n\nI've noted {len(gaps)} areas for exploration during ideation: {', '.join(g['gap_description'][:50] for g in gaps[:3])}"

        return f"""Great! We've covered {coverage:.0f}% of the key dimensions.

Here's your synthesized context:

---
{state.context.enriched_domain}
---
{gap_note}

Does this capture your situation accurately? Feel free to correct anything before we proceed."""

    def get_dimension_transition_message(
        self,
        from_dim: InterviewDimension,
        to_dim: InterviewDimension
    ) -> str:
        """Get message for transitioning between dimensions."""
        dimension_names = {
            InterviewDimension.PROBLEM_SPACE: "the problem space",
            InterviewDimension.CONSTRAINTS: "constraints and boundaries",
            InterviewDimension.ASSUMPTIONS: "assumptions and beliefs",
            InterviewDimension.INTENT: "strategic intent",
            InterviewDimension.PREFERENCES: "your preferences",
            InterviewDimension.EXISTING_SOLUTIONS: "existing solutions",
            InterviewDimension.RESOURCES: "your resources and advantages",
        }

        from_name = dimension_names.get(from_dim, from_dim.value)
        to_name = dimension_names.get(to_dim, to_dim.value)

        return f"Thanks for that insight on {from_name}. Now let's explore {to_name}."
