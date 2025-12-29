"""Escape module for plateau detection and breakout."""

from .plateau_escape import (
    PlateauEscapeProtocol,
    PlateauDetector,
    EscapeStrategyGenerator,
    PlateauStatus,
    PlateauAnalysis,
    EscapeResult,
    EscapeIdea,
    EscapeAttempt,
    create_escape_protocol,
    simulate_plateau_detection
)

__all__ = [
    'PlateauEscapeProtocol',
    'PlateauDetector',
    'EscapeStrategyGenerator',
    'PlateauStatus',
    'PlateauAnalysis',
    'EscapeResult',
    'EscapeIdea',
    'EscapeAttempt',
    'create_escape_protocol',
    'simulate_plateau_detection'
]
