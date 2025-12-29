"""
Verification Diagnostics for Universal Ideation v3.2

Provides detailed analysis of verification failures and patterns:
- Failure frequency tracking by verifier type
- Common failure patterns detection
- Session health monitoring
- Retry effectiveness analysis
- Diagnostic reports for debugging
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import json

from .verification_gate import (
    VerifierOutput,
    VerificationResult,
    VerificationLevel,
    GateVerificationResult,
    RetryContext
)


@dataclass
class FailureRecord:
    """Record of a single verification failure."""
    timestamp: str
    verifier_name: str
    level: str
    message: str
    idea_title: str
    fix_hint: Optional[str]
    was_retried: bool
    retry_succeeded: bool


@dataclass
class PatternAnalysis:
    """Analysis of failure patterns."""
    most_common_failures: List[Tuple[str, int]]
    failure_by_level: Dict[str, int]
    retry_success_rate: float
    average_retries_needed: float
    recommendations: List[str]


@dataclass
class SessionHealth:
    """Overall health metrics for ideation session."""
    total_ideas: int
    passed_first_try: int
    passed_after_retry: int
    failed_permanently: int
    pass_rate: float
    first_try_rate: float
    health_score: float  # 0.0 to 1.0
    status: str  # "healthy", "degraded", "critical"


class VerificationDiagnostics:
    """
    Comprehensive diagnostics for verification system.

    Tracks failures, identifies patterns, and provides recommendations.
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize diagnostics tracker.

        Args:
            session_id: Optional identifier for this session
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = datetime.now()

        # Failure tracking
        self.failure_records: List[FailureRecord] = []
        self.verifier_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"pass": 0, "fail": 0, "warn": 0, "skip": 0}
        )

        # Pattern detection
        self.failure_sequences: List[List[str]] = []
        self.current_sequence: List[str] = []

        # Retry tracking
        self.retry_attempts: List[Tuple[int, bool]] = []  # (attempts, succeeded)

        # Idea tracking
        self.ideas_processed = 0
        self.ideas_passed = 0
        self.ideas_passed_first_try = 0

    def record_verification(
        self,
        result: GateVerificationResult,
        idea: Dict,
        retry_context: Optional[RetryContext] = None
    ) -> None:
        """
        Record a verification result for diagnostics.

        Args:
            result: The verification result
            idea: The idea that was verified
            retry_context: Optional retry context if this was a retry
        """
        self.ideas_processed += 1
        idea_title = idea.get("title", f"Idea_{self.ideas_processed}")

        # Track all verifier outputs
        for output in result.all_outputs:
            self.verifier_stats[output.name][output.result.value] += 1

        if result.passed:
            self.ideas_passed += 1
            if retry_context is None or retry_context.attempt == 0:
                self.ideas_passed_first_try += 1

            # End failure sequence on success
            if self.current_sequence:
                self.failure_sequences.append(self.current_sequence.copy())
                self.current_sequence = []

        else:
            # Record failures
            for failure in result.hard_failures + result.soft_warnings:
                record = FailureRecord(
                    timestamp=datetime.now().isoformat(),
                    verifier_name=failure.name,
                    level=failure.level.value,
                    message=failure.message,
                    idea_title=idea_title,
                    fix_hint=failure.fix_hint,
                    was_retried=retry_context is not None,
                    retry_succeeded=False
                )
                self.failure_records.append(record)
                self.current_sequence.append(failure.name)

    def record_retry_outcome(self, attempts: int, succeeded: bool) -> None:
        """Record the outcome of a retry sequence."""
        self.retry_attempts.append((attempts, succeeded))

        # Update success flags on recent failure records
        if succeeded:
            for record in reversed(self.failure_records[-attempts:]):
                record.retry_succeeded = True

    def get_failure_frequency(self) -> Dict[str, int]:
        """Get failure count by verifier name."""
        return {
            name: stats["fail"]
            for name, stats in self.verifier_stats.items()
            if stats["fail"] > 0
        }

    def get_verifier_pass_rates(self) -> Dict[str, float]:
        """Get pass rate for each verifier."""
        rates = {}
        for name, stats in self.verifier_stats.items():
            total = stats["pass"] + stats["fail"]
            if total > 0:
                rates[name] = stats["pass"] / total
        return rates

    def analyze_patterns(self) -> PatternAnalysis:
        """Analyze failure patterns and generate insights."""
        # Count most common failures
        failure_counts: Dict[str, int] = defaultdict(int)
        for record in self.failure_records:
            failure_counts[record.verifier_name] += 1

        most_common = sorted(
            failure_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # Count by level
        level_counts: Dict[str, int] = defaultdict(int)
        for record in self.failure_records:
            level_counts[record.level] += 1

        # Retry analysis
        if self.retry_attempts:
            successful_retries = sum(1 for _, s in self.retry_attempts if s)
            retry_success_rate = successful_retries / len(self.retry_attempts)
            total_attempts = sum(a for a, _ in self.retry_attempts)
            avg_retries = total_attempts / len(self.retry_attempts)
        else:
            retry_success_rate = 0.0
            avg_retries = 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            most_common, level_counts, retry_success_rate
        )

        return PatternAnalysis(
            most_common_failures=most_common,
            failure_by_level=dict(level_counts),
            retry_success_rate=retry_success_rate,
            average_retries_needed=avg_retries,
            recommendations=recommendations
        )

    def _generate_recommendations(
        self,
        most_common: List[Tuple[str, int]],
        level_counts: Dict[str, int],
        retry_success_rate: float
    ) -> List[str]:
        """Generate actionable recommendations based on patterns."""
        recommendations = []

        if not most_common:
            return ["No failures recorded - verification is working well"]

        # Analyze top failure
        top_failure, top_count = most_common[0]
        total_failures = sum(count for _, count in most_common)

        if top_count / total_failures > 0.5:
            recommendations.append(
                f"Focus on '{top_failure}' - accounts for {top_count/total_failures:.0%} of failures. "
                "Consider improving generation prompts to address this specific issue."
            )

        # Check hard vs soft failures
        hard_count = level_counts.get("hard", 0)
        soft_count = level_counts.get("soft", 0)

        if hard_count > soft_count * 2:
            recommendations.append(
                "High hard failure rate indicates structural issues with generated ideas. "
                "Check that generation prompts specify required format correctly."
            )

        if soft_count > hard_count * 3:
            recommendations.append(
                "Many soft warnings suggest ideas are valid but could be improved. "
                "Consider enabling strict mode or improving generation guidance."
            )

        # Retry effectiveness
        if retry_success_rate < 0.3 and len(self.retry_attempts) > 5:
            recommendations.append(
                f"Low retry success rate ({retry_success_rate:.0%}). "
                "Retry hints may not be effective - consider more specific guidance."
            )
        elif retry_success_rate > 0.7:
            recommendations.append(
                f"High retry success rate ({retry_success_rate:.0%}). "
                "Retry mechanism is working well - failures are recoverable."
            )

        # Specific verifier recommendations
        verifier_recommendations = {
            "required_fields": "Ensure generation prompt specifies all required fields in output format",
            "description_length": "Adjust description length guidance in generation prompts",
            "differentiators_count": "Specify exact number of differentiators expected",
            "market_specificity": "Add examples of specific target markets in prompts",
            "buzzword_density": "Instruct generator to use concrete language instead of buzzwords"
        }

        for failure, count in most_common[:3]:
            if failure in verifier_recommendations and count >= 3:
                recommendations.append(verifier_recommendations[failure])

        return recommendations[:5]  # Limit to top 5 recommendations

    def get_session_health(self) -> SessionHealth:
        """Calculate overall session health metrics."""
        if self.ideas_processed == 0:
            return SessionHealth(
                total_ideas=0,
                passed_first_try=0,
                passed_after_retry=0,
                failed_permanently=0,
                pass_rate=0.0,
                first_try_rate=0.0,
                health_score=1.0,
                status="healthy"
            )

        passed_after_retry = self.ideas_passed - self.ideas_passed_first_try
        failed_permanently = self.ideas_processed - self.ideas_passed

        pass_rate = self.ideas_passed / self.ideas_processed
        first_try_rate = self.ideas_passed_first_try / self.ideas_processed

        # Calculate health score
        # Weighted: 60% pass rate, 30% first-try rate, 10% retry effectiveness
        health_score = (
            0.6 * pass_rate +
            0.3 * first_try_rate +
            0.1 * (1.0 if failed_permanently == 0 else
                   max(0, 1.0 - failed_permanently / self.ideas_processed))
        )

        # Determine status
        if health_score >= 0.8:
            status = "healthy"
        elif health_score >= 0.5:
            status = "degraded"
        else:
            status = "critical"

        return SessionHealth(
            total_ideas=self.ideas_processed,
            passed_first_try=self.ideas_passed_first_try,
            passed_after_retry=passed_after_retry,
            failed_permanently=failed_permanently,
            pass_rate=pass_rate,
            first_try_rate=first_try_rate,
            health_score=health_score,
            status=status
        )

    def get_recent_failures(self, limit: int = 10) -> List[FailureRecord]:
        """Get most recent failure records."""
        return self.failure_records[-limit:]

    def get_diagnostic_report(self) -> Dict:
        """Generate comprehensive diagnostic report."""
        health = self.get_session_health()
        patterns = self.analyze_patterns()

        return {
            "session_id": self.session_id,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "health": {
                "score": health.health_score,
                "status": health.status,
                "pass_rate": health.pass_rate,
                "first_try_rate": health.first_try_rate
            },
            "totals": {
                "processed": self.ideas_processed,
                "passed": self.ideas_passed,
                "passed_first_try": self.ideas_passed_first_try,
                "failed": health.failed_permanently
            },
            "verifier_pass_rates": self.get_verifier_pass_rates(),
            "failure_frequency": self.get_failure_frequency(),
            "patterns": {
                "most_common_failures": patterns.most_common_failures,
                "failure_by_level": patterns.failure_by_level,
                "retry_success_rate": patterns.retry_success_rate,
                "average_retries_needed": patterns.average_retries_needed
            },
            "recommendations": patterns.recommendations,
            "recent_failures": [
                {
                    "verifier": r.verifier_name,
                    "message": r.message,
                    "idea": r.idea_title,
                    "recovered": r.retry_succeeded
                }
                for r in self.get_recent_failures(5)
            ]
        }

    def format_summary(self) -> str:
        """Format a human-readable summary."""
        health = self.get_session_health()
        patterns = self.analyze_patterns()

        lines = [
            f"VERIFICATION DIAGNOSTICS - Session {self.session_id}",
            "=" * 50,
            "",
            f"Session Health: {health.status.upper()} ({health.health_score:.0%})",
            "",
            "TOTALS:",
            f"  Processed: {self.ideas_processed}",
            f"  Passed (first try): {self.ideas_passed_first_try}",
            f"  Passed (after retry): {health.passed_after_retry}",
            f"  Failed permanently: {health.failed_permanently}",
            "",
            f"Pass Rate: {health.pass_rate:.0%}",
            f"First-Try Rate: {health.first_try_rate:.0%}",
            "",
        ]

        if patterns.most_common_failures:
            lines.append("TOP FAILURES:")
            for name, count in patterns.most_common_failures[:5]:
                lines.append(f"  {name}: {count} occurrences")
            lines.append("")

        if patterns.recommendations:
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(patterns.recommendations, 1):
                lines.append(f"  {i}. {rec}")
            lines.append("")

        return "\n".join(lines)

    def export_json(self, filepath: str) -> None:
        """Export diagnostics to JSON file."""
        report = self.get_diagnostic_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

    def reset(self) -> None:
        """Reset all diagnostic data for new session."""
        self.__init__(session_id=None)


class DiagnosticsAggregator:
    """
    Aggregates diagnostics across multiple sessions.

    Useful for tracking long-term patterns and system improvements.
    """

    def __init__(self):
        self.session_reports: List[Dict] = []

    def add_session(self, diagnostics: VerificationDiagnostics) -> None:
        """Add a completed session's diagnostics."""
        self.session_reports.append(diagnostics.get_diagnostic_report())

    def get_trend_analysis(self) -> Dict:
        """Analyze trends across sessions."""
        if not self.session_reports:
            return {"error": "No sessions recorded"}

        # Track key metrics over time
        pass_rates = [r["health"]["pass_rate"] for r in self.session_reports]
        health_scores = [r["health"]["score"] for r in self.session_reports]

        # Identify improving/degrading verifiers
        verifier_trends: Dict[str, List[float]] = defaultdict(list)
        for report in self.session_reports:
            for verifier, rate in report.get("verifier_pass_rates", {}).items():
                verifier_trends[verifier].append(rate)

        improving = []
        degrading = []
        for verifier, rates in verifier_trends.items():
            if len(rates) >= 3:
                trend = rates[-1] - rates[0]
                if trend > 0.1:
                    improving.append((verifier, trend))
                elif trend < -0.1:
                    degrading.append((verifier, trend))

        return {
            "session_count": len(self.session_reports),
            "pass_rate_trend": {
                "first": pass_rates[0] if pass_rates else 0,
                "last": pass_rates[-1] if pass_rates else 0,
                "average": sum(pass_rates) / len(pass_rates) if pass_rates else 0
            },
            "health_score_trend": {
                "first": health_scores[0] if health_scores else 0,
                "last": health_scores[-1] if health_scores else 0,
                "average": sum(health_scores) / len(health_scores) if health_scores else 0
            },
            "improving_verifiers": improving,
            "degrading_verifiers": degrading
        }
