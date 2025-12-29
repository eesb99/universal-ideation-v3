"""
Gates module for Universal Ideation v3.2

Provides quality gates that filter and validate generated ideas:
- SemanticDistanceGate: Prevents idea clustering (v3.0)
- VerificationGate: Validates idea completeness and feasibility (v3.2)
"""

from .semantic_distance_gate import (
    SemanticDistanceGate,
    GateDecision,
    GateResult,
    IdeaEmbedding,
    check_semantic_distance,
    create_mock_embedding
)

from .verification_gate import (
    VerificationGate,
    VerificationLevel,
    VerificationResult,
    VerifierOutput,
    GateVerificationResult,
    RetryContext,
    RetryController,
    HardVerifiers,
    SoftVerifiers,
    LLMVerifiers
)

from .verification_diagnostics import (
    VerificationDiagnostics,
    DiagnosticsAggregator,
    FailureRecord,
    PatternAnalysis,
    SessionHealth
)

__all__ = [
    # Semantic Distance Gate
    "SemanticDistanceGate",
    "GateDecision",
    "GateResult",
    "IdeaEmbedding",
    "check_semantic_distance",
    "create_mock_embedding",
    # Verification Gate
    "VerificationGate",
    "VerificationLevel",
    "VerificationResult",
    "VerifierOutput",
    "GateVerificationResult",
    "RetryContext",
    "RetryController",
    "HardVerifiers",
    "SoftVerifiers",
    "LLMVerifiers",
    # Diagnostics
    "VerificationDiagnostics",
    "DiagnosticsAggregator",
    "FailureRecord",
    "PatternAnalysis",
    "SessionHealth"
]
