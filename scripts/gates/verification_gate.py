"""
Verification Gate System for Universal Ideation v3.2

Implements VeriMAP-style verification with fail-fast pattern:
1. Hard Verifiers: Programmatic completeness checks (required fields, format)
2. Soft Verifiers: Heuristic feasibility checks (bounds, plausibility)
3. LLM Verifiers: Semantic validation (coherence, market fit, feasibility)

Ideas failing verification are rejected with diagnostic feedback for retry.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from enum import Enum
import re
import json


class VerificationLevel(Enum):
    """Verification strictness levels."""
    HARD = "hard"      # Must pass - reject if fails
    SOFT = "soft"      # Should pass - warn if fails, allow borderline
    LLM = "llm"        # Semantic check - requires LLM call


class VerificationResult(Enum):
    """Result of a single verification check."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"      # Verifier not applicable


@dataclass
class VerifierOutput:
    """Output from a single verifier."""
    name: str
    level: VerificationLevel
    result: VerificationResult
    message: str
    details: Optional[Dict] = None
    fix_hint: Optional[str] = None


@dataclass
class GateVerificationResult:
    """Aggregate result from verification gate."""
    passed: bool
    hard_failures: List[VerifierOutput]
    soft_warnings: List[VerifierOutput]
    llm_results: List[VerifierOutput]
    all_outputs: List[VerifierOutput]
    retry_allowed: bool
    retry_hints: List[str]
    confidence_score: float  # 0.0 to 1.0

    @property
    def failure_count(self) -> int:
        return len(self.hard_failures)

    @property
    def warning_count(self) -> int:
        return len(self.soft_warnings)

    def get_summary(self) -> str:
        """Get human-readable summary."""
        if self.passed:
            return f"PASSED (confidence: {self.confidence_score:.0%}, warnings: {self.warning_count})"
        else:
            return f"FAILED ({self.failure_count} hard failures, {self.warning_count} warnings)"


@dataclass
class RetryContext:
    """Context for retry attempts."""
    attempt: int
    max_attempts: int
    prior_failures: List[str]
    accumulated_hints: List[str]

    @property
    def should_retry(self) -> bool:
        return self.attempt < self.max_attempts

    def get_retry_prompt_injection(self) -> str:
        """Generate text to inject into retry prompt."""
        if not self.accumulated_hints:
            return ""

        hints_text = "\n".join(f"- {hint}" for hint in self.accumulated_hints)
        return f"""
## RETRY GUIDANCE (Attempt {self.attempt + 1}/{self.max_attempts})
Previous attempt failed verification. Address these issues:
{hints_text}

Be specific and concrete in your response to avoid these issues again.
"""


class HardVerifiers:
    """
    Programmatic completeness checks - ideas MUST pass these.

    These are fast, deterministic checks for structural requirements.
    """

    @staticmethod
    def check_required_fields(idea: Dict, required: List[str]) -> VerifierOutput:
        """Check that all required fields are present and non-empty."""
        missing = []
        empty = []

        for field_name in required:
            if field_name not in idea:
                missing.append(field_name)
            elif not idea[field_name] or (isinstance(idea[field_name], str) and not idea[field_name].strip()):
                empty.append(field_name)

        if missing or empty:
            issues = []
            if missing:
                issues.append(f"missing: {missing}")
            if empty:
                issues.append(f"empty: {empty}")

            return VerifierOutput(
                name="required_fields",
                level=VerificationLevel.HARD,
                result=VerificationResult.FAIL,
                message=f"Required field issues: {'; '.join(issues)}",
                details={"missing": missing, "empty": empty},
                fix_hint=f"Ensure these fields are provided: {missing + empty}"
            )

        return VerifierOutput(
            name="required_fields",
            level=VerificationLevel.HARD,
            result=VerificationResult.PASS,
            message="All required fields present"
        )

    @staticmethod
    def check_description_length(idea: Dict, min_words: int = 10, max_words: int = 200) -> VerifierOutput:
        """Check description is within acceptable length."""
        description = idea.get("description", "")
        if not description:
            return VerifierOutput(
                name="description_length",
                level=VerificationLevel.HARD,
                result=VerificationResult.FAIL,
                message="Description is missing",
                fix_hint="Provide a description of 2-3 sentences"
            )

        word_count = len(description.split())

        if word_count < min_words:
            return VerifierOutput(
                name="description_length",
                level=VerificationLevel.HARD,
                result=VerificationResult.FAIL,
                message=f"Description too short ({word_count} words, minimum {min_words})",
                details={"word_count": word_count, "min": min_words},
                fix_hint=f"Expand description to at least {min_words} words with more detail"
            )

        if word_count > max_words:
            return VerifierOutput(
                name="description_length",
                level=VerificationLevel.HARD,
                result=VerificationResult.FAIL,
                message=f"Description too long ({word_count} words, maximum {max_words})",
                details={"word_count": word_count, "max": max_words},
                fix_hint=f"Condense description to under {max_words} words"
            )

        return VerifierOutput(
            name="description_length",
            level=VerificationLevel.HARD,
            result=VerificationResult.PASS,
            message=f"Description length OK ({word_count} words)"
        )

    @staticmethod
    def check_differentiators_count(idea: Dict, min_count: int = 2, max_count: int = 5) -> VerifierOutput:
        """Check differentiators list has appropriate count."""
        differentiators = idea.get("differentiators", [])

        if not isinstance(differentiators, list):
            return VerifierOutput(
                name="differentiators_count",
                level=VerificationLevel.HARD,
                result=VerificationResult.FAIL,
                message="Differentiators must be a list",
                fix_hint="Provide differentiators as a list of strings"
            )

        count = len(differentiators)

        if count < min_count:
            return VerifierOutput(
                name="differentiators_count",
                level=VerificationLevel.HARD,
                result=VerificationResult.FAIL,
                message=f"Too few differentiators ({count}, minimum {min_count})",
                details={"count": count, "min": min_count},
                fix_hint=f"Add at least {min_count - count} more unique differentiators"
            )

        if count > max_count:
            return VerifierOutput(
                name="differentiators_count",
                level=VerificationLevel.HARD,
                result=VerificationResult.FAIL,
                message=f"Too many differentiators ({count}, maximum {max_count})",
                details={"count": count, "max": max_count},
                fix_hint=f"Focus on the {max_count} most important differentiators"
            )

        return VerifierOutput(
            name="differentiators_count",
            level=VerificationLevel.HARD,
            result=VerificationResult.PASS,
            message=f"Differentiator count OK ({count})"
        )

    @staticmethod
    def check_title_format(idea: Dict, max_length: int = 80) -> VerifierOutput:
        """Check title is properly formatted."""
        title = idea.get("title", "")

        if not title or not title.strip():
            return VerifierOutput(
                name="title_format",
                level=VerificationLevel.HARD,
                result=VerificationResult.FAIL,
                message="Title is missing or empty",
                fix_hint="Provide a concise, memorable title"
            )

        if len(title) > max_length:
            return VerifierOutput(
                name="title_format",
                level=VerificationLevel.HARD,
                result=VerificationResult.FAIL,
                message=f"Title too long ({len(title)} chars, max {max_length})",
                fix_hint=f"Shorten title to under {max_length} characters"
            )

        # Check for placeholder patterns
        placeholder_patterns = [
            r'\[.*\]',  # [placeholder]
            r'<.*>',    # <placeholder>
            r'TODO',
            r'TBD',
            r'XXX'
        ]
        for pattern in placeholder_patterns:
            if re.search(pattern, title, re.IGNORECASE):
                return VerifierOutput(
                    name="title_format",
                    level=VerificationLevel.HARD,
                    result=VerificationResult.FAIL,
                    message="Title contains placeholder text",
                    fix_hint="Replace placeholder with actual content"
                )

        return VerifierOutput(
            name="title_format",
            level=VerificationLevel.HARD,
            result=VerificationResult.PASS,
            message="Title format OK"
        )

    @staticmethod
    def check_json_parseable(text: str) -> VerifierOutput:
        """Check that text can be parsed as JSON."""
        try:
            # Try to extract JSON from text (may be wrapped in markdown)
            json_match = re.search(r'\{[\s\S]*\}', text)
            if not json_match:
                return VerifierOutput(
                    name="json_parseable",
                    level=VerificationLevel.HARD,
                    result=VerificationResult.FAIL,
                    message="No JSON object found in response",
                    fix_hint="Ensure response contains valid JSON object"
                )

            json.loads(json_match.group())
            return VerifierOutput(
                name="json_parseable",
                level=VerificationLevel.HARD,
                result=VerificationResult.PASS,
                message="JSON is valid"
            )
        except json.JSONDecodeError as e:
            return VerifierOutput(
                name="json_parseable",
                level=VerificationLevel.HARD,
                result=VerificationResult.FAIL,
                message=f"Invalid JSON: {str(e)}",
                fix_hint="Fix JSON syntax errors and ensure valid format"
            )


class SoftVerifiers:
    """
    Heuristic feasibility checks - ideas SHOULD pass these.

    Warnings are issued but don't block acceptance.
    """

    @staticmethod
    def check_market_specificity(idea: Dict) -> VerifierOutput:
        """Check that target market is specific enough."""
        target = idea.get("target_market", "")

        if not target:
            return VerifierOutput(
                name="market_specificity",
                level=VerificationLevel.SOFT,
                result=VerificationResult.WARN,
                message="Target market not specified",
                fix_hint="Define a specific target customer segment"
            )

        # Check for vague terms
        vague_terms = ["everyone", "all", "general", "broad", "anyone", "people"]
        target_lower = target.lower()

        if any(term in target_lower for term in vague_terms):
            return VerifierOutput(
                name="market_specificity",
                level=VerificationLevel.SOFT,
                result=VerificationResult.WARN,
                message=f"Target market too vague: '{target}'",
                details={"target": target},
                fix_hint="Narrow down to specific demographics, psychographics, or use cases"
            )

        # Check minimum specificity (at least 3 words)
        if len(target.split()) < 3:
            return VerifierOutput(
                name="market_specificity",
                level=VerificationLevel.SOFT,
                result=VerificationResult.WARN,
                message=f"Target market could be more specific: '{target}'",
                fix_hint="Add more detail about who exactly will use this"
            )

        return VerifierOutput(
            name="market_specificity",
            level=VerificationLevel.SOFT,
            result=VerificationResult.PASS,
            message="Target market is sufficiently specific"
        )

    @staticmethod
    def check_differentiator_uniqueness(idea: Dict) -> VerifierOutput:
        """Check that differentiators are actually different from each other."""
        differentiators = idea.get("differentiators", [])

        if len(differentiators) < 2:
            return VerifierOutput(
                name="differentiator_uniqueness",
                level=VerificationLevel.SOFT,
                result=VerificationResult.SKIP,
                message="Not enough differentiators to check uniqueness"
            )

        # Check for semantic overlap (simple keyword matching)
        normalized = [d.lower().split() for d in differentiators]

        overlap_pairs = []
        for i in range(len(normalized)):
            for j in range(i + 1, len(normalized)):
                common = set(normalized[i]) & set(normalized[j])
                # Remove common filler words
                common -= {"the", "a", "an", "and", "or", "is", "are", "for", "to", "with"}
                if len(common) >= 3:  # Significant overlap
                    overlap_pairs.append((differentiators[i], differentiators[j]))

        if overlap_pairs:
            return VerifierOutput(
                name="differentiator_uniqueness",
                level=VerificationLevel.SOFT,
                result=VerificationResult.WARN,
                message=f"Some differentiators overlap: {len(overlap_pairs)} pairs",
                details={"overlapping_pairs": overlap_pairs[:2]},  # Limit to 2 examples
                fix_hint="Make each differentiator address a distinct aspect"
            )

        return VerifierOutput(
            name="differentiator_uniqueness",
            level=VerificationLevel.SOFT,
            result=VerificationResult.PASS,
            message="Differentiators are sufficiently unique"
        )

    @staticmethod
    def check_buzzword_density(idea: Dict, max_density: float = 0.15) -> VerifierOutput:
        """Check that description isn't overloaded with buzzwords."""
        description = idea.get("description", "")

        buzzwords = [
            "revolutionary", "disruptive", "game-changing", "paradigm-shift",
            "synergy", "leverage", "innovative", "cutting-edge", "best-in-class",
            "next-generation", "breakthrough", "transformative", "holistic",
            "ecosystem", "platform", "scalable", "agile", "seamless"
        ]

        words = description.lower().split()
        if not words:
            return VerifierOutput(
                name="buzzword_density",
                level=VerificationLevel.SOFT,
                result=VerificationResult.SKIP,
                message="No description to analyze"
            )

        buzzword_count = sum(1 for word in words if any(bw in word for bw in buzzwords))
        density = buzzword_count / len(words)

        if density > max_density:
            return VerifierOutput(
                name="buzzword_density",
                level=VerificationLevel.SOFT,
                result=VerificationResult.WARN,
                message=f"High buzzword density ({density:.0%})",
                details={"density": density, "buzzword_count": buzzword_count},
                fix_hint="Replace buzzwords with specific, concrete descriptions"
            )

        return VerifierOutput(
            name="buzzword_density",
            level=VerificationLevel.SOFT,
            result=VerificationResult.PASS,
            message=f"Buzzword density OK ({density:.0%})"
        )

    @staticmethod
    def check_actionability(idea: Dict) -> VerifierOutput:
        """Check that the idea suggests a clear path to action."""
        description = idea.get("description", "")
        mvp = idea.get("mvp_description", "")

        action_indicators = [
            "by", "through", "using", "with", "via",
            "customers can", "users can", "allows", "enables",
            "provides", "delivers", "offers"
        ]

        text = (description + " " + mvp).lower()

        has_action = any(indicator in text for indicator in action_indicators)

        if not has_action:
            return VerifierOutput(
                name="actionability",
                level=VerificationLevel.SOFT,
                result=VerificationResult.WARN,
                message="Idea lacks clear action mechanism",
                fix_hint="Explain HOW this works, not just WHAT it is"
            )

        return VerifierOutput(
            name="actionability",
            level=VerificationLevel.SOFT,
            result=VerificationResult.PASS,
            message="Idea has clear action mechanism"
        )


class LLMVerifiers:
    """
    Semantic validation checks requiring LLM inference.

    These are expensive but catch issues programmatic checks miss.
    """

    @staticmethod
    def get_coherence_prompt(idea: Dict) -> str:
        """Generate prompt for coherence check."""
        return f"""Evaluate if this idea is internally coherent and logically consistent.

IDEA:
Title: {idea.get('title', 'N/A')}
Description: {idea.get('description', 'N/A')}
Differentiators: {idea.get('differentiators', [])}
Target Market: {idea.get('target_market', 'N/A')}

EVALUATE:
1. Do the differentiators support the core concept?
2. Is the target market aligned with the offering?
3. Does the description make logical sense?

Respond with JSON:
{{
    "coherent": true/false,
    "issues": ["issue 1", "issue 2"] or [],
    "confidence": 0.0-1.0
}}"""

    @staticmethod
    def get_feasibility_prompt(idea: Dict, domain: str) -> str:
        """Generate prompt for feasibility check."""
        return f"""Evaluate if this idea is feasible to implement in the {domain} industry.

IDEA:
Title: {idea.get('title', 'N/A')}
Description: {idea.get('description', 'N/A')}
MVP: {idea.get('mvp_description', 'Not specified')}

EVALUATE:
1. Are there obvious technical barriers?
2. Are there regulatory/legal concerns?
3. Is the business model plausible?
4. Can this be built within reasonable resources?

Respond with JSON:
{{
    "feasible": true/false,
    "barriers": ["barrier 1", "barrier 2"] or [],
    "risk_level": "low"/"medium"/"high",
    "confidence": 0.0-1.0
}}"""

    @staticmethod
    def get_novelty_prompt(idea: Dict, prior_ideas: List[Dict]) -> str:
        """Generate prompt for novelty check vs prior ideas."""
        prior_titles = [i.get('title', 'Untitled') for i in prior_ideas[:10]]
        prior_list = "\n".join(f"- {t}" for t in prior_titles)

        return f"""Evaluate if this idea is sufficiently novel compared to prior ideas.

NEW IDEA:
Title: {idea.get('title', 'N/A')}
Description: {idea.get('description', 'N/A')}

PRIOR IDEAS (for comparison):
{prior_list if prior_list else "No prior ideas"}

EVALUATE:
1. Is this conceptually different from prior ideas?
2. Does it offer a genuinely new approach?
3. Or is it just a minor variation?

Respond with JSON:
{{
    "novel": true/false,
    "similarity_to": "most similar prior idea title or null",
    "novelty_score": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""


class VerificationGate:
    """
    Main verification gate orchestrating all verifiers.

    Implements fail-fast pattern: hard failures stop immediately.
    """

    def __init__(
        self,
        max_retries: int = 3,
        required_fields: Optional[List[str]] = None,
        enable_llm_verification: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize verification gate.

        Args:
            max_retries: Maximum retry attempts for failed verification
            required_fields: Fields that must be present in ideas
            enable_llm_verification: Whether to run LLM-based checks
            strict_mode: If True, soft warnings also cause rejection
        """
        self.max_retries = max_retries
        self.required_fields = required_fields or [
            "title", "description", "differentiators", "target_market"
        ]
        self.enable_llm_verification = enable_llm_verification
        self.strict_mode = strict_mode

        # Statistics
        self.total_checked = 0
        self.passed_count = 0
        self.failed_count = 0
        self.retry_success_count = 0

    def verify(
        self,
        idea: Dict,
        domain: str = "general",
        prior_ideas: Optional[List[Dict]] = None,
        llm_callback: Optional[Callable[[str], Dict]] = None
    ) -> GateVerificationResult:
        """
        Run all verification checks on an idea.

        Args:
            idea: The idea dictionary to verify
            domain: Domain context for feasibility checks
            prior_ideas: Prior ideas for novelty comparison
            llm_callback: Function to call LLM for semantic checks

        Returns:
            GateVerificationResult with all outputs
        """
        self.total_checked += 1

        hard_failures: List[VerifierOutput] = []
        soft_warnings: List[VerifierOutput] = []
        llm_results: List[VerifierOutput] = []
        all_outputs: List[VerifierOutput] = []

        # Phase 1: Hard Verifiers (fail-fast)
        hard_checks = [
            HardVerifiers.check_required_fields(idea, self.required_fields),
            HardVerifiers.check_title_format(idea),
            HardVerifiers.check_description_length(idea),
            HardVerifiers.check_differentiators_count(idea),
        ]

        for output in hard_checks:
            all_outputs.append(output)
            if output.result == VerificationResult.FAIL:
                hard_failures.append(output)

        # Fail fast on hard failures
        if hard_failures:
            self.failed_count += 1
            return GateVerificationResult(
                passed=False,
                hard_failures=hard_failures,
                soft_warnings=[],
                llm_results=[],
                all_outputs=all_outputs,
                retry_allowed=True,
                retry_hints=[f.fix_hint for f in hard_failures if f.fix_hint],
                confidence_score=0.0
            )

        # Phase 2: Soft Verifiers
        soft_checks = [
            SoftVerifiers.check_market_specificity(idea),
            SoftVerifiers.check_differentiator_uniqueness(idea),
            SoftVerifiers.check_buzzword_density(idea),
            SoftVerifiers.check_actionability(idea),
        ]

        for output in soft_checks:
            all_outputs.append(output)
            if output.result == VerificationResult.WARN:
                soft_warnings.append(output)

        # In strict mode, soft warnings cause failure
        if self.strict_mode and soft_warnings:
            self.failed_count += 1
            return GateVerificationResult(
                passed=False,
                hard_failures=[],
                soft_warnings=soft_warnings,
                llm_results=[],
                all_outputs=all_outputs,
                retry_allowed=True,
                retry_hints=[w.fix_hint for w in soft_warnings if w.fix_hint],
                confidence_score=0.3
            )

        # Phase 3: LLM Verifiers (if enabled and callback provided)
        if self.enable_llm_verification and llm_callback:
            llm_results = self._run_llm_verification(
                idea, domain, prior_ideas or [], llm_callback
            )
            all_outputs.extend(llm_results)

            # Check for LLM failures
            llm_failures = [r for r in llm_results if r.result == VerificationResult.FAIL]
            if llm_failures:
                self.failed_count += 1
                return GateVerificationResult(
                    passed=False,
                    hard_failures=llm_failures,  # Treat as hard failures
                    soft_warnings=soft_warnings,
                    llm_results=llm_results,
                    all_outputs=all_outputs,
                    retry_allowed=True,
                    retry_hints=[f.fix_hint for f in llm_failures if f.fix_hint],
                    confidence_score=0.2
                )

        # Calculate confidence score
        warning_penalty = len(soft_warnings) * 0.1
        confidence = max(0.5, 1.0 - warning_penalty)

        self.passed_count += 1
        return GateVerificationResult(
            passed=True,
            hard_failures=[],
            soft_warnings=soft_warnings,
            llm_results=llm_results,
            all_outputs=all_outputs,
            retry_allowed=False,
            retry_hints=[],
            confidence_score=confidence
        )

    def _run_llm_verification(
        self,
        idea: Dict,
        domain: str,
        prior_ideas: List[Dict],
        llm_callback: Callable[[str], Dict]
    ) -> List[VerifierOutput]:
        """Run LLM-based verification checks."""
        results = []

        # Coherence check
        try:
            coherence_prompt = LLMVerifiers.get_coherence_prompt(idea)
            coherence_response = llm_callback(coherence_prompt)

            if coherence_response.get("coherent", True):
                results.append(VerifierOutput(
                    name="llm_coherence",
                    level=VerificationLevel.LLM,
                    result=VerificationResult.PASS,
                    message="Idea is coherent",
                    details=coherence_response
                ))
            else:
                issues = coherence_response.get("issues", [])
                results.append(VerifierOutput(
                    name="llm_coherence",
                    level=VerificationLevel.LLM,
                    result=VerificationResult.FAIL,
                    message=f"Coherence issues: {issues}",
                    details=coherence_response,
                    fix_hint="Ensure description, differentiators, and target market align logically"
                ))
        except Exception as e:
            results.append(VerifierOutput(
                name="llm_coherence",
                level=VerificationLevel.LLM,
                result=VerificationResult.SKIP,
                message=f"Coherence check failed: {str(e)}"
            ))

        # Feasibility check
        try:
            feasibility_prompt = LLMVerifiers.get_feasibility_prompt(idea, domain)
            feasibility_response = llm_callback(feasibility_prompt)

            if feasibility_response.get("feasible", True):
                results.append(VerifierOutput(
                    name="llm_feasibility",
                    level=VerificationLevel.LLM,
                    result=VerificationResult.PASS,
                    message=f"Idea is feasible (risk: {feasibility_response.get('risk_level', 'unknown')})",
                    details=feasibility_response
                ))
            else:
                barriers = feasibility_response.get("barriers", [])
                results.append(VerifierOutput(
                    name="llm_feasibility",
                    level=VerificationLevel.LLM,
                    result=VerificationResult.WARN,  # Warn, don't fail
                    message=f"Feasibility concerns: {barriers}",
                    details=feasibility_response,
                    fix_hint="Address identified barriers or propose workarounds"
                ))
        except Exception as e:
            results.append(VerifierOutput(
                name="llm_feasibility",
                level=VerificationLevel.LLM,
                result=VerificationResult.SKIP,
                message=f"Feasibility check failed: {str(e)}"
            ))

        return results

    def create_retry_context(
        self,
        result: GateVerificationResult,
        prior_context: Optional[RetryContext] = None
    ) -> RetryContext:
        """Create retry context from verification result."""
        attempt = 1 if prior_context is None else prior_context.attempt + 1
        prior_failures = prior_context.prior_failures if prior_context else []

        # Collect failure messages
        new_failures = [f.message for f in result.hard_failures]
        all_failures = prior_failures + new_failures

        return RetryContext(
            attempt=attempt,
            max_attempts=self.max_retries,
            prior_failures=all_failures,
            accumulated_hints=result.retry_hints
        )

    def get_statistics(self) -> Dict:
        """Get verification statistics."""
        return {
            "total_checked": self.total_checked,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "pass_rate": self.passed_count / max(self.total_checked, 1),
            "retry_success": self.retry_success_count
        }


class RetryController:
    """
    Manages verification retry logic with exponential backoff hints.
    """

    def __init__(self, gate: VerificationGate):
        self.gate = gate
        self.retry_history: List[RetryContext] = []

    def attempt_verification(
        self,
        idea: Dict,
        domain: str = "general",
        prior_ideas: Optional[List[Dict]] = None,
        llm_callback: Optional[Callable[[str], Dict]] = None,
        regenerate_callback: Optional[Callable[[str, RetryContext], Dict]] = None
    ) -> tuple[bool, GateVerificationResult, Optional[Dict]]:
        """
        Attempt verification with automatic retries.

        Args:
            idea: Initial idea to verify
            domain: Domain context
            prior_ideas: Prior ideas for comparison
            llm_callback: LLM for semantic checks
            regenerate_callback: Function to regenerate idea based on hints

        Returns:
            Tuple of (success, final_result, final_idea)
        """
        current_idea = idea
        context: Optional[RetryContext] = None

        for attempt in range(self.gate.max_retries + 1):
            result = self.gate.verify(
                current_idea,
                domain=domain,
                prior_ideas=prior_ideas,
                llm_callback=llm_callback
            )

            if result.passed:
                if attempt > 0:
                    self.gate.retry_success_count += 1
                return (True, result, current_idea)

            # Check if retry is allowed and we have more attempts
            if not result.retry_allowed or attempt >= self.gate.max_retries:
                return (False, result, current_idea)

            # Create retry context
            context = self.gate.create_retry_context(result, context)
            self.retry_history.append(context)

            # Regenerate if callback provided
            if regenerate_callback:
                retry_prompt = context.get_retry_prompt_injection()
                current_idea = regenerate_callback(retry_prompt, context)
            else:
                # No regeneration possible, fail
                return (False, result, current_idea)

        return (False, result, current_idea)
