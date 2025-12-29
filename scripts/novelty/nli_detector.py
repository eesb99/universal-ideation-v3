"""
NLI-Based Novelty Detector for Universal Ideation v3.2

Uses Natural Language Inference to detect novelty of atomic claims:
- Compares each claim against prior claims corpus
- Classifies relationships: entailment (not novel), contradiction, neutral (novel)
- Supports both lightweight rule-based and LLM-based NLI
- Achieves 0.94 accuracy vs 0.83 for cosine similarity (per research)

Based on NovAScore: Atomic novelty evaluation with NLI detection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple, Set
from enum import Enum
from datetime import datetime
import re
from collections import defaultdict

from .acu_decomposer import AtomicClaim, ClaimType


class NLIRelation(Enum):
    """NLI relationship classification."""
    ENTAILMENT = "entailment"     # Prior claim implies this claim (NOT novel)
    CONTRADICTION = "contradiction"  # Prior claim contradicts this claim
    NEUTRAL = "neutral"           # No relationship (potentially novel)


class NoveltyLevel(Enum):
    """Novelty classification levels."""
    HIGHLY_NOVEL = "highly_novel"       # No similar claims, unique concept
    MODERATELY_NOVEL = "moderately_novel"  # Some overlap but distinct
    INCREMENTAL = "incremental"         # Small variation on existing
    NOT_NOVEL = "not_novel"             # Essentially restates prior claim


@dataclass
class NLIResult:
    """Result of NLI comparison for a single claim pair."""
    claim_id: str
    prior_claim_id: str
    relation: NLIRelation
    confidence: float  # 0-1
    explanation: str


@dataclass
class ClaimNoveltyResult:
    """Novelty assessment for a single claim."""
    claim: AtomicClaim
    novelty_level: NoveltyLevel
    novelty_score: float  # 0-100
    nli_results: List[NLIResult]
    most_similar_prior: Optional[str]  # ID of most similar prior claim
    similarity_to_prior: float  # 0-1
    is_contradictory: bool  # Contradicts any prior claim


@dataclass
class NoveltyDetectionResult:
    """Complete novelty detection result for an idea."""
    idea_id: str
    claim_results: List[ClaimNoveltyResult]
    total_claims: int
    highly_novel_count: int
    moderately_novel_count: int
    incremental_count: int
    not_novel_count: int
    average_novelty_score: float
    contradictions_found: int
    processing_time_ms: float

    def get_novel_claims(self, min_level: NoveltyLevel = NoveltyLevel.MODERATELY_NOVEL) -> List[ClaimNoveltyResult]:
        """Get claims meeting minimum novelty level."""
        level_order = {
            NoveltyLevel.NOT_NOVEL: 0,
            NoveltyLevel.INCREMENTAL: 1,
            NoveltyLevel.MODERATELY_NOVEL: 2,
            NoveltyLevel.HIGHLY_NOVEL: 3
        }
        min_order = level_order[min_level]
        return [r for r in self.claim_results if level_order[r.novelty_level] >= min_order]

    def to_dict(self) -> Dict:
        return {
            "idea_id": self.idea_id,
            "total_claims": self.total_claims,
            "highly_novel_count": self.highly_novel_count,
            "moderately_novel_count": self.moderately_novel_count,
            "incremental_count": self.incremental_count,
            "not_novel_count": self.not_novel_count,
            "average_novelty_score": self.average_novelty_score,
            "contradictions_found": self.contradictions_found,
            "processing_time_ms": self.processing_time_ms
        }


class NLINoveltyDetector:
    """
    Detects novelty using Natural Language Inference.

    Compares atomic claims against a corpus of prior claims to determine
    if each claim is truly novel or just a restatement of existing ideas.
    """

    # Keywords indicating semantic similarity
    SYNONYM_GROUPS = {
        "personalize": {"customize", "tailor", "individualize", "adapt", "personal"},
        "optimize": {"improve", "enhance", "maximize", "better", "upgrade"},
        "ai": {"artificial intelligence", "machine learning", "ml", "neural", "algorithm"},
        "protein": {"whey", "casein", "plant-based", "amino", "powder"},
        "health": {"wellness", "fitness", "nutrition", "healthy", "nutritious"},
        "fast": {"quick", "rapid", "instant", "speedy", "swift"},
        "cheap": {"affordable", "low-cost", "budget", "economical", "inexpensive"},
        "new": {"novel", "innovative", "revolutionary", "breakthrough", "first"},
    }

    # Entailment indicators
    ENTAILMENT_PATTERNS = [
        (r"(.+) is a type of (.+)", "hypernym"),
        (r"(.+) includes (.+)", "inclusion"),
        (r"(.+) such as (.+)", "example"),
        (r"(.+) means (.+)", "definition"),
    ]

    # Contradiction indicators
    CONTRADICTION_INDICATORS = [
        ("not", "eliminates"),
        ("without", "with"),
        ("unlike", "like"),
        ("instead of", "using"),
        ("rather than", "and"),
        ("never", "always"),
        ("none", "all"),
    ]

    def __init__(
        self,
        prior_claims: Optional[List[AtomicClaim]] = None,
        llm_callback: Optional[Callable[[str], Dict]] = None,
        use_llm: bool = False,
        similarity_threshold: float = 0.7,
        novelty_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize NLI novelty detector.

        Args:
            prior_claims: Existing claims to compare against
            llm_callback: Optional LLM for NLI classification
            use_llm: Whether to use LLM for NLI
            similarity_threshold: Threshold for considering claims similar
            novelty_thresholds: Custom thresholds for novelty levels
        """
        self.prior_claims: List[AtomicClaim] = prior_claims or []
        self.llm_callback = llm_callback
        self.use_llm = use_llm and llm_callback is not None
        self.similarity_threshold = similarity_threshold

        # Novelty score thresholds
        self.novelty_thresholds = novelty_thresholds or {
            "highly_novel": 85,
            "moderately_novel": 65,
            "incremental": 40,
        }

        # Build prior claims index for fast lookup
        self._build_index()

    def _build_index(self):
        """Build keyword index for prior claims."""
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self.claim_keywords: Dict[str, Set[str]] = {}

        for claim in self.prior_claims:
            keywords = set(kw.lower() for kw in claim.keywords)
            self.claim_keywords[claim.id] = keywords

            for kw in keywords:
                self.keyword_index[kw].add(claim.id)

    def add_prior_claims(self, claims: List[AtomicClaim]):
        """Add claims to prior claims corpus."""
        for claim in claims:
            self.prior_claims.append(claim)

            keywords = set(kw.lower() for kw in claim.keywords)
            self.claim_keywords[claim.id] = keywords

            for kw in keywords:
                self.keyword_index[kw].add(claim.id)

    def detect_novelty(
        self,
        claims: List[AtomicClaim],
        idea_id: str = "unknown"
    ) -> NoveltyDetectionResult:
        """
        Detect novelty for a list of claims.

        Args:
            claims: Claims to assess
            idea_id: ID of the source idea

        Returns:
            NoveltyDetectionResult with all assessments
        """
        start_time = datetime.now()

        claim_results = []
        for claim in claims:
            result = self._assess_claim_novelty(claim)
            claim_results.append(result)

        # Calculate statistics
        highly_novel = sum(1 for r in claim_results if r.novelty_level == NoveltyLevel.HIGHLY_NOVEL)
        moderately_novel = sum(1 for r in claim_results if r.novelty_level == NoveltyLevel.MODERATELY_NOVEL)
        incremental = sum(1 for r in claim_results if r.novelty_level == NoveltyLevel.INCREMENTAL)
        not_novel = sum(1 for r in claim_results if r.novelty_level == NoveltyLevel.NOT_NOVEL)
        contradictions = sum(1 for r in claim_results if r.is_contradictory)

        avg_novelty = (
            sum(r.novelty_score for r in claim_results) / len(claim_results)
            if claim_results else 0
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return NoveltyDetectionResult(
            idea_id=idea_id,
            claim_results=claim_results,
            total_claims=len(claims),
            highly_novel_count=highly_novel,
            moderately_novel_count=moderately_novel,
            incremental_count=incremental,
            not_novel_count=not_novel,
            average_novelty_score=avg_novelty,
            contradictions_found=contradictions,
            processing_time_ms=processing_time
        )

    def _assess_claim_novelty(self, claim: AtomicClaim) -> ClaimNoveltyResult:
        """Assess novelty of a single claim."""
        nli_results = []
        max_similarity = 0.0
        most_similar_id = None
        is_contradictory = False

        # Get candidate prior claims (those sharing keywords)
        candidates = self._get_candidate_priors(claim)

        for prior in candidates:
            # Perform NLI comparison
            nli_result = self._compare_claims(claim, prior)
            nli_results.append(nli_result)

            # Track highest similarity (entailment)
            if nli_result.relation == NLIRelation.ENTAILMENT:
                if nli_result.confidence > max_similarity:
                    max_similarity = nli_result.confidence
                    most_similar_id = prior.id

            # Track contradictions
            if nli_result.relation == NLIRelation.CONTRADICTION:
                is_contradictory = True

        # Calculate novelty score
        novelty_score = self._calculate_novelty_score(max_similarity, is_contradictory, claim)
        novelty_level = self._classify_novelty_level(novelty_score)

        return ClaimNoveltyResult(
            claim=claim,
            novelty_level=novelty_level,
            novelty_score=novelty_score,
            nli_results=nli_results,
            most_similar_prior=most_similar_id,
            similarity_to_prior=max_similarity,
            is_contradictory=is_contradictory
        )

    def _get_candidate_priors(self, claim: AtomicClaim) -> List[AtomicClaim]:
        """Get prior claims that might be related (keyword overlap)."""
        claim_keywords = set(kw.lower() for kw in claim.keywords)

        # Find claims sharing at least one keyword
        candidate_ids = set()
        for kw in claim_keywords:
            # Also check synonyms
            all_keywords = {kw}
            for base, synonyms in self.SYNONYM_GROUPS.items():
                if kw == base or kw in synonyms:
                    all_keywords.update(synonyms)
                    all_keywords.add(base)

            for k in all_keywords:
                candidate_ids.update(self.keyword_index.get(k, set()))

        # Return actual claims
        candidates = [c for c in self.prior_claims if c.id in candidate_ids]

        # If no keyword matches, sample some claims for comparison
        if not candidates and self.prior_claims:
            candidates = self.prior_claims[:10]  # Sample first 10

        return candidates

    def _compare_claims(self, claim: AtomicClaim, prior: AtomicClaim) -> NLIResult:
        """Compare two claims using NLI."""
        if self.use_llm and self.llm_callback:
            return self._llm_nli(claim, prior)
        else:
            return self._rule_based_nli(claim, prior)

    def _rule_based_nli(self, claim: AtomicClaim, prior: AtomicClaim) -> NLIResult:
        """Rule-based NLI classification."""
        claim_text = claim.text.lower()
        prior_text = prior.text.lower()

        # Calculate word overlap
        claim_words = set(claim_text.split())
        prior_words = set(prior_text.split())
        overlap = len(claim_words & prior_words) / max(len(claim_words | prior_words), 1)

        # Check for contradiction indicators
        for neg, pos in self.CONTRADICTION_INDICATORS:
            if neg in claim_text and pos in prior_text:
                return NLIResult(
                    claim_id=claim.id,
                    prior_claim_id=prior.id,
                    relation=NLIRelation.CONTRADICTION,
                    confidence=0.75,
                    explanation=f"Contradiction detected: '{neg}' vs '{pos}'"
                )
            if pos in claim_text and neg in prior_text:
                return NLIResult(
                    claim_id=claim.id,
                    prior_claim_id=prior.id,
                    relation=NLIRelation.CONTRADICTION,
                    confidence=0.75,
                    explanation=f"Contradiction detected: '{pos}' vs '{neg}'"
                )

        # Check for entailment (high overlap + similar claim type)
        if overlap > self.similarity_threshold:
            # Same type and high overlap suggests entailment
            if claim.claim_type == prior.claim_type:
                return NLIResult(
                    claim_id=claim.id,
                    prior_claim_id=prior.id,
                    relation=NLIRelation.ENTAILMENT,
                    confidence=min(0.95, overlap),
                    explanation=f"High word overlap ({overlap:.0%}) with same claim type"
                )
            else:
                return NLIResult(
                    claim_id=claim.id,
                    prior_claim_id=prior.id,
                    relation=NLIRelation.ENTAILMENT,
                    confidence=min(0.8, overlap * 0.9),
                    explanation=f"High word overlap ({overlap:.0%}) with different claim type"
                )

        # Moderate overlap - check for synonym matches
        synonym_boost = self._calculate_synonym_similarity(claim_text, prior_text)
        adjusted_overlap = overlap + synonym_boost

        if adjusted_overlap > self.similarity_threshold:
            return NLIResult(
                claim_id=claim.id,
                prior_claim_id=prior.id,
                relation=NLIRelation.ENTAILMENT,
                confidence=min(0.85, adjusted_overlap),
                explanation=f"Semantic similarity ({adjusted_overlap:.0%}) via synonyms"
            )

        # Low overlap - neutral
        return NLIResult(
            claim_id=claim.id,
            prior_claim_id=prior.id,
            relation=NLIRelation.NEUTRAL,
            confidence=1.0 - adjusted_overlap,
            explanation=f"Low overlap ({overlap:.0%}), considered distinct"
        )

    def _calculate_synonym_similarity(self, text1: str, text2: str) -> float:
        """Calculate additional similarity from synonym matches."""
        boost = 0.0

        for base, synonyms in self.SYNONYM_GROUPS.items():
            all_forms = synonyms | {base}

            # Check if text1 has one form and text2 has another
            text1_has = any(form in text1 for form in all_forms)
            text2_has = any(form in text2 for form in all_forms)

            if text1_has and text2_has:
                boost += 0.1  # Each synonym match adds 10%

        return min(0.3, boost)  # Cap at 30% boost

    def _llm_nli(self, claim: AtomicClaim, prior: AtomicClaim) -> NLIResult:
        """Use LLM for NLI classification."""
        if not self.llm_callback:
            return self._rule_based_nli(claim, prior)

        prompt = f"""Classify the relationship between these two claims:

Claim A (new): {claim.text}
Claim B (prior): {prior.text}

Relationship types:
- ENTAILMENT: Claim B implies or covers Claim A (A is not novel)
- CONTRADICTION: Claims conflict or are incompatible
- NEUTRAL: Claims are unrelated or A adds new information (A is novel)

Respond with: RELATIONSHIP, CONFIDENCE (0-1), BRIEF_EXPLANATION"""

        try:
            result = self.llm_callback(prompt)
            if isinstance(result, dict):
                relation_str = result.get("relationship", "neutral").lower()
                relation = NLIRelation.NEUTRAL
                if "entail" in relation_str:
                    relation = NLIRelation.ENTAILMENT
                elif "contradict" in relation_str:
                    relation = NLIRelation.CONTRADICTION

                return NLIResult(
                    claim_id=claim.id,
                    prior_claim_id=prior.id,
                    relation=relation,
                    confidence=result.get("confidence", 0.8),
                    explanation=result.get("explanation", "LLM classification")
                )
        except Exception:
            pass

        return self._rule_based_nli(claim, prior)

    def _calculate_novelty_score(
        self,
        max_similarity: float,
        is_contradictory: bool,
        claim: AtomicClaim
    ) -> float:
        """Calculate novelty score 0-100."""
        # Base score inversely related to similarity
        base_score = (1.0 - max_similarity) * 100

        # Bonus for explicit novelty claims
        if claim.claim_type == ClaimType.NOVELTY:
            base_score = min(100, base_score + 10)

        # Bonus for contradiction (indicates challenging assumptions)
        if is_contradictory:
            base_score = min(100, base_score + 5)

        # Penalty for low confidence extraction
        if claim.confidence < 0.7:
            base_score *= 0.9

        return round(base_score, 1)

    def _classify_novelty_level(self, score: float) -> NoveltyLevel:
        """Classify novelty score into level."""
        if score >= self.novelty_thresholds["highly_novel"]:
            return NoveltyLevel.HIGHLY_NOVEL
        elif score >= self.novelty_thresholds["moderately_novel"]:
            return NoveltyLevel.MODERATELY_NOVEL
        elif score >= self.novelty_thresholds["incremental"]:
            return NoveltyLevel.INCREMENTAL
        else:
            return NoveltyLevel.NOT_NOVEL

    def get_statistics(self) -> Dict:
        """Get detector statistics."""
        return {
            "prior_claims_count": len(self.prior_claims),
            "indexed_keywords": len(self.keyword_index),
            "use_llm": self.use_llm,
            "similarity_threshold": self.similarity_threshold,
            "novelty_thresholds": self.novelty_thresholds
        }


def create_test_detector() -> NLINoveltyDetector:
    """Create detector with sample prior claims."""
    prior_claims = [
        AtomicClaim(
            id="prior_1",
            text="AI-powered protein recommendations",
            claim_type=ClaimType.FEATURE,
            source=ClaimType.FEATURE,
            source_text="",
            confidence=0.9,
            keywords=["ai", "protein", "recommendations"]
        ),
        AtomicClaim(
            id="prior_2",
            text="Personalized nutrition based on user data",
            claim_type=ClaimType.BENEFIT,
            source=ClaimType.BENEFIT,
            source_text="",
            confidence=0.9,
            keywords=["personalized", "nutrition", "user", "data"]
        ),
        AtomicClaim(
            id="prior_3",
            text="Integration with fitness trackers",
            claim_type=ClaimType.FEATURE,
            source=ClaimType.FEATURE,
            source_text="",
            confidence=0.9,
            keywords=["integration", "fitness", "trackers"]
        ),
    ]

    return NLINoveltyDetector(prior_claims=prior_claims)
