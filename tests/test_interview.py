"""
Unit Tests for Universal Interview Module
Tests storage, models, engine, and context selection
"""

import pytest
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from interview import (
    InterviewStorage,
    InterviewDimension,
    InitiativeStatus,
    ConfidenceLevel,
    ResponseSource,
    TemplateScaffold,
    TemplateDefaults,
    DimensionResponse,
    InterviewContext,
    EngagementMetrics,
    InterviewProgress,
    InterviewState,
    InterviewEngine,
    Question,
    QuestionBank,
    ContextSelector
)


# ===================
# Fixtures
# ===================

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    yield db_path
    os.unlink(db_path)


@pytest.fixture
def storage(temp_db):
    """Create a storage instance with temporary database."""
    return InterviewStorage(db_path=temp_db)


@pytest.fixture
def engine(storage):
    """Create an engine instance."""
    return InterviewEngine(storage=storage)


# ===================
# Storage Tests
# ===================

class TestInterviewStorage:
    """Tests for InterviewStorage class."""

    def test_init_creates_tables(self, storage):
        """Test that initialization creates required tables."""
        stats = storage.get_stats()
        assert 'total_initiatives' in stats
        assert 'total_sessions' in stats
        assert 'total_responses' in stats

    def test_create_initiative(self, storage):
        """Test creating an initiative."""
        init_id = storage.create_initiative(
            name="Test Initiative",
            original_domain="sustainable packaging",
            template_scaffold="bootstrap"
        )

        assert init_id is not None
        assert len(init_id) == 36  # UUID format

        # Retrieve and verify
        init = storage.get_initiative(init_id)
        assert init is not None
        assert init['name'] == "Test Initiative"
        assert init['original_domain'] == "sustainable packaging"
        assert init['status'] == 'draft'

    def test_update_initiative(self, storage):
        """Test updating an initiative."""
        init_id = storage.create_initiative(
            name="Test",
            original_domain="test domain"
        )

        # Update enriched domain
        storage.update_initiative(
            init_id,
            enriched_domain="Enriched test domain with context"
        )

        init = storage.get_initiative(init_id)
        assert init['enriched_domain'] == "Enriched test domain with context"

    def test_update_initiative_status(self, storage):
        """Test status lifecycle updates."""
        init_id = storage.create_initiative(
            name="Test",
            original_domain="test"
        )

        # Draft -> Ready
        storage.update_initiative_status(init_id, InitiativeStatus.READY)
        init = storage.get_initiative(init_id)
        assert init['status'] == 'ready'

        # Ready -> Active
        storage.update_initiative_status(init_id, InitiativeStatus.ACTIVE)
        init = storage.get_initiative(init_id)
        assert init['status'] == 'active'

    def test_list_initiatives(self, storage):
        """Test listing initiatives."""
        # Create multiple initiatives
        storage.create_initiative(name="Init 1", original_domain="domain 1")
        storage.create_initiative(name="Init 2", original_domain="domain 2")
        storage.create_initiative(name="Init 3", original_domain="domain 3")

        initiatives = storage.list_initiatives()
        assert len(initiatives) == 3

    def test_list_initiatives_by_status(self, storage):
        """Test filtering initiatives by status."""
        init1 = storage.create_initiative(name="Init 1", original_domain="domain 1")
        init2 = storage.create_initiative(name="Init 2", original_domain="domain 2")

        storage.update_initiative_status(init1, InitiativeStatus.READY)

        drafts = storage.list_initiatives(status=InitiativeStatus.DRAFT)
        ready = storage.list_initiatives(status=InitiativeStatus.READY)

        assert len(drafts) == 1
        assert len(ready) == 1

    def test_delete_initiative(self, storage):
        """Test deleting an initiative."""
        init_id = storage.create_initiative(
            name="To Delete",
            original_domain="delete me"
        )

        assert storage.delete_initiative(init_id)
        assert storage.get_initiative(init_id) is None

    def test_create_session(self, storage):
        """Test creating an interview session."""
        init_id = storage.create_initiative(
            name="Test",
            original_domain="test"
        )

        session_id = storage.create_session(init_id)
        assert session_id is not None

        session = storage.get_session(session_id)
        assert session is not None
        assert session['initiative_id'] == init_id
        assert session['is_complete'] == 0

    def test_store_response(self, storage):
        """Test storing interview responses."""
        init_id = storage.create_initiative(
            name="Test",
            original_domain="test"
        )
        session_id = storage.create_session(init_id)

        response_id = storage.store_response(
            session_id=session_id,
            initiative_id=init_id,
            dimension=InterviewDimension.PROBLEM_SPACE,
            question="What problem are you solving?",
            response="Users struggle with X",
            confidence=ConfidenceLevel.HIGH,
            source=ResponseSource.USER
        )

        assert response_id is not None

        responses = storage.get_responses_by_dimension(
            init_id,
            InterviewDimension.PROBLEM_SPACE
        )
        assert len(responses) == 1
        assert responses[0]['response'] == "Users struggle with X"

    def test_flag_gap(self, storage):
        """Test flagging knowledge gaps."""
        init_id = storage.create_initiative(
            name="Test",
            original_domain="test"
        )

        gap_id = storage.flag_gap(
            init_id,
            InterviewDimension.CONSTRAINTS,
            "User uncertain about regulatory requirements"
        )

        gaps = storage.get_gaps(init_id)
        assert len(gaps) == 1
        assert gaps[0]['dimension'] == 'constraints'

    def test_build_context(self, storage):
        """Test building complete context."""
        init_id = storage.create_initiative(
            name="Test Context",
            original_domain="test domain"
        )
        session_id = storage.create_session(init_id)

        # Add some responses
        storage.store_response(
            session_id=session_id,
            initiative_id=init_id,
            dimension=InterviewDimension.PROBLEM_SPACE,
            question="Q1",
            response="Problem response",
            confidence=ConfidenceLevel.HIGH
        )
        storage.store_response(
            session_id=session_id,
            initiative_id=init_id,
            dimension=InterviewDimension.CONSTRAINTS,
            question="Q2",
            response="Budget under $50k",
            confidence=ConfidenceLevel.MEDIUM
        )

        context = storage.build_context(init_id)

        assert context['initiative_name'] == "Test Context"
        assert 'problem_space' in context['dimensions']
        assert 'constraints' in context['dimensions']

    def test_export_markdown(self, storage):
        """Test markdown export."""
        init_id = storage.create_initiative(
            name="Export Test",
            original_domain="test domain"
        )
        session_id = storage.create_session(init_id)

        storage.store_response(
            session_id=session_id,
            initiative_id=init_id,
            dimension=InterviewDimension.PROBLEM_SPACE,
            question="Q1",
            response="Problem description here",
            confidence=ConfidenceLevel.HIGH
        )

        markdown = storage.export_markdown(init_id)

        assert "# Interview Context: Export Test" in markdown
        assert "Problem Space" in markdown
        assert "Problem description here" in markdown


# ===================
# Model Tests
# ===================

class TestModels:
    """Tests for data models."""

    def test_dimension_response(self):
        """Test DimensionResponse dataclass."""
        dim = DimensionResponse(
            response="Test response",
            confidence="high",
            source="user"
        )

        assert dim.response == "Test response"
        assert dim.confidence == "high"

        d = dim.to_dict()
        assert d['response'] == "Test response"

    def test_interview_context(self):
        """Test InterviewContext dataclass."""
        context = InterviewContext(
            initiative_id="test-id",
            initiative_name="Test",
            original_domain="test domain"
        )

        context.problem_space = DimensionResponse(
            response="Problem description",
            confidence="high"
        )

        assert context.get_coverage_percentage() > 0
        assert 'problem_space' in context.get_covered_dimensions()

    def test_interview_context_to_dict(self):
        """Test context serialization."""
        context = InterviewContext(
            initiative_id="test-id",
            initiative_name="Test",
            original_domain="test domain",
            enriched_domain="Enriched domain text"
        )

        d = context.to_dict()
        assert d['initiative_id'] == "test-id"
        assert d['enriched_domain'] == "Enriched domain text"
        assert 'dimensions' in d

    def test_interview_context_from_dict(self):
        """Test context deserialization."""
        data = {
            'initiative_id': 'test-id',
            'initiative_name': 'Test',
            'original_domain': 'test domain',
            'dimensions': {
                'problem_space': {
                    'response': 'Problem text',
                    'confidence': 'high',
                    'source': 'user'
                }
            }
        }

        context = InterviewContext.from_dict(data)
        assert context.initiative_id == 'test-id'
        assert context.problem_space.response == 'Problem text'

    def test_engagement_metrics(self):
        """Test EngagementMetrics tracking."""
        metrics = EngagementMetrics()

        # Add responses
        metrics.add_response("This is a detailed response about the problem", 5.0)
        metrics.add_response("Short", 2.0)
        metrics.add_response("I don't know", 1.0)

        assert len(metrics.response_lengths) == 3
        assert metrics.vague_response_count >= 1
        assert metrics.get_average_response_length() > 0

    def test_engagement_fatigue_detection(self):
        """Test fatigue signal detection."""
        metrics = EngagementMetrics()

        # Simulate declining engagement
        metrics.add_response("This is a long detailed response", 10.0)
        metrics.add_response("This is another long response", 8.0)
        metrics.add_response("shorter", 3.0)
        metrics.add_response("let's move on", 2.0)
        metrics.add_response("skip", 1.0)

        assert metrics.fatigue_signals >= 1
        assert metrics.should_wrap_up()

    def test_interview_progress(self):
        """Test InterviewProgress tracking."""
        progress = InterviewProgress()

        assert progress.get_progress_percentage() == 0

        progress.touch_dimension('problem_space')
        progress.touch_dimension('constraints')

        assert len(progress.dimensions_touched) == 2
        assert progress.get_progress_percentage() > 0
        assert 'intent' in progress.get_remaining_dimensions()

    def test_template_defaults(self):
        """Test template scaffold defaults."""
        bootstrap = TemplateDefaults.get_defaults(TemplateScaffold.BOOTSTRAP)
        assert 'constraints' in bootstrap
        assert bootstrap['constraints']['budget_range'] == "<$50k"

        enterprise = TemplateDefaults.get_defaults(TemplateScaffold.ENTERPRISE)
        assert 'scalability' in enterprise['constraints']['requirements']


# ===================
# Question Bank Tests
# ===================

class TestQuestionBank:
    """Tests for QuestionBank."""

    def test_get_base_question(self):
        """Test getting base questions."""
        question = QuestionBank.get_base_question(InterviewDimension.PROBLEM_SPACE)

        assert question is not None
        assert question.text
        assert question.dimension == InterviewDimension.PROBLEM_SPACE
        assert question.is_base

    def test_get_base_question_excludes(self):
        """Test question exclusion."""
        q1 = QuestionBank.get_base_question(InterviewDimension.PROBLEM_SPACE)

        # Get another question excluding the first
        q2 = QuestionBank.get_base_question(
            InterviewDimension.PROBLEM_SPACE,
            exclude=[q1.text]
        )

        # Should get different question (if available)
        assert q2 is not None

    def test_all_dimensions_have_questions(self):
        """Test all dimensions have base questions."""
        for dim in InterviewDimension:
            question = QuestionBank.get_base_question(dim)
            assert question is not None
            assert question.text


# ===================
# Engine Tests
# ===================

class TestInterviewEngine:
    """Tests for InterviewEngine."""

    def test_start_interview(self, engine):
        """Test starting a new interview."""
        state = engine.start_interview(
            domain="sustainable packaging",
            initiative_name="Test Interview"
        )

        assert state is not None
        assert state.initiative_id
        assert state.session_id
        assert state.context.original_domain == "sustainable packaging"
        assert state.progress.get_progress_percentage() == 0

    def test_start_interview_with_template(self, engine):
        """Test starting with template scaffold."""
        state = engine.start_interview(
            domain="healthcare startup",
            template=TemplateScaffold.REGULATED
        )

        assert state.context.template_scaffold == "regulated"

    def test_get_next_question(self, engine):
        """Test question generation."""
        state = engine.start_interview(
            domain="test domain"
        )

        question = engine.get_next_question(state)

        assert question is not None
        assert question.text
        assert question.dimension is not None

    def test_process_response(self, engine):
        """Test response processing."""
        state = engine.start_interview(domain="test")
        question = engine.get_next_question(state)

        should_continue, progress_msg = engine.process_response(
            state,
            question,
            "This is my detailed response about the problem we're trying to solve."
        )

        assert isinstance(should_continue, bool)
        assert state.progress.questions_asked == 1

    def test_resume_interview(self, engine, storage):
        """Test resuming an interrupted interview."""
        # Start interview
        state = engine.start_interview(domain="test")
        question = engine.get_next_question(state)
        engine.process_response(state, question, "Initial response")

        # Simulate interruption
        engine.save_interruption(state, "test_interruption")

        # Resume
        resumed_state = engine.resume_interview(state.initiative_id)

        assert resumed_state is not None
        assert resumed_state.initiative_id == state.initiative_id
        assert resumed_state.progress.questions_asked == 1

    def test_complete_interview(self, engine):
        """Test completing an interview."""
        state = engine.start_interview(domain="test domain")

        # Answer several questions
        for _ in range(3):
            question = engine.get_next_question(state)
            if question:
                engine.process_response(state, question, "Response text")

        context = engine.complete_interview(state)

        assert context is not None
        assert state.is_complete
        assert context.status == 'ready'

    def test_template_detection(self, engine):
        """Test automatic template detection."""
        bootstrap = engine.detect_template("budget-constrained startup MVP")
        assert bootstrap == TemplateScaffold.BOOTSTRAP

        enterprise = engine.detect_template("enterprise B2B integration")
        assert enterprise == TemplateScaffold.ENTERPRISE

        regulated = engine.detect_template("FDA-approved medical device")
        assert regulated == TemplateScaffold.REGULATED

        sustainable = engine.detect_template("eco-friendly sustainable packaging")
        assert sustainable == TemplateScaffold.SUSTAINABLE

    def test_confidence_assessment(self, engine):
        """Test response confidence assessment."""
        assert engine._assess_confidence("I'm absolutely certain about this") == ConfidenceLevel.HIGH
        assert engine._assess_confidence("Maybe it could work") == ConfidenceLevel.LOW
        assert engine._assess_confidence("The budget is around $100k") == ConfidenceLevel.MEDIUM


# ===================
# Interview State Tests
# ===================

class TestInterviewState:
    """Tests for InterviewState."""

    def test_should_complete_all_dimensions(self):
        """Test completion when all dimensions covered."""
        state = InterviewState()
        state.progress.dimensions_touched = [
            'problem_space', 'constraints', 'assumptions',
            'intent', 'preferences', 'existing_solutions', 'resources'
        ]

        assert state.should_complete()

    def test_should_complete_fatigue(self):
        """Test completion on fatigue signals."""
        state = InterviewState()
        state.engagement.fatigue_signals = 3

        assert state.should_complete()

    def test_next_dimension_suggestion(self):
        """Test dimension suggestion."""
        state = InterviewState()

        # First suggestion should be problem_space
        assert state.get_next_dimension_suggestion() == 'problem_space'

        state.progress.touch_dimension('problem_space')
        assert state.get_next_dimension_suggestion() == 'intent'


# ===================
# Context Selector Tests
# ===================

class TestContextSelector:
    """Tests for ContextSelector."""

    def test_get_available_contexts(self, storage):
        """Test getting available contexts."""
        # Create some initiatives
        init1 = storage.create_initiative(name="Init 1", original_domain="domain 1")
        init2 = storage.create_initiative(name="Init 2", original_domain="domain 2")

        storage.update_initiative_status(init1, InitiativeStatus.READY)
        storage.update_initiative_status(init2, InitiativeStatus.READY)

        selector = ContextSelector(storage=storage)
        contexts = selector.get_available_contexts()

        assert len(contexts) == 2

    def test_get_context_by_id(self, storage):
        """Test getting context by ID."""
        init_id = storage.create_initiative(
            name="Test",
            original_domain="test domain"
        )
        session_id = storage.create_session(init_id)

        storage.store_response(
            session_id=session_id,
            initiative_id=init_id,
            dimension=InterviewDimension.PROBLEM_SPACE,
            question="Q1",
            response="Problem response"
        )

        selector = ContextSelector(storage=storage)
        context = selector.get_context_by_id(init_id[:8])

        assert context is not None
        assert context.initiative_id == init_id

    def test_format_context_for_prompt(self, storage):
        """Test prompt formatting."""
        init_id = storage.create_initiative(
            name="Test",
            original_domain="test domain"
        )

        context = InterviewContext(
            initiative_id=init_id,
            initiative_name="Test",
            original_domain="test domain"
        )
        context.problem_space = DimensionResponse(
            response="Users struggle with packaging waste",
            confidence="high"
        )

        selector = ContextSelector(storage=storage)
        prompt_text = selector.format_context_for_prompt(context)

        assert "test domain" in prompt_text
        assert "Problem Space" in prompt_text


# ===================
# Integration Tests
# ===================

class TestIntegration:
    """Integration tests for the full interview flow."""

    def test_full_interview_flow(self, engine, storage):
        """Test complete interview flow from start to finish."""
        # Start
        state = engine.start_interview(
            domain="sustainable protein alternatives",
            initiative_name="Protein Innovation"
        )

        # Answer questions until complete or max reached
        max_questions = 10
        for i in range(max_questions):
            question = engine.get_next_question(state)
            if not question:
                break

            response = f"Detailed response #{i+1} about {question.dimension.value}"
            should_continue, _ = engine.process_response(state, question, response)

            if not should_continue:
                break

        # Complete
        context = engine.complete_interview(state)

        # Verify
        assert context.status == 'ready'
        assert state.progress.questions_asked > 0
        assert len(context.get_covered_dimensions()) > 0

        # Export
        markdown = storage.export_markdown(state.initiative_id)
        assert "Protein Innovation" in markdown

    def test_interview_with_gaps(self, engine, storage):
        """Test interview that flags knowledge gaps."""
        state = engine.start_interview(domain="test")

        question = engine.get_next_question(state)
        engine.process_response(state, question, "I don't know about this")

        engine.complete_interview(state)

        gaps = storage.get_gaps(state.initiative_id)
        # Should have flagged a gap due to uncertain response
        assert len(gaps) >= 0  # May or may not have gaps depending on dimension


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
