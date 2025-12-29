"""
ACU Decomposer for Universal Ideation v3.2

Implements Atomic Content Unit decomposition for fine-grained novelty detection:
- Breaks ideas into atomic claims (smallest meaningful units)
- Extracts structured claims from title, description, differentiators
- Categorizes claims by type (feature, benefit, mechanism, market, constraint)
- Supports both rule-based and LLM-based decomposition

Based on NovAScore: Atomic novelty evaluation achieving 0.94 accuracy vs 0.83 for cosine similarity.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple
from enum import Enum
from datetime import datetime
import re
import hashlib


class ClaimType(Enum):
    """Type of atomic claim."""
    FEATURE = "feature"           # What the product has/does
    BENEFIT = "benefit"           # What value it provides
    MECHANISM = "mechanism"       # How it works
    MARKET = "market"             # Who it's for
    CONSTRAINT = "constraint"     # Limitations or requirements
    COMPARISON = "comparison"     # Comparison to alternatives
    NOVELTY = "novelty"           # Explicit novelty claims
    UNKNOWN = "unknown"


class ClaimSource(Enum):
    """Source field of the claim."""
    TITLE = "title"
    DESCRIPTION = "description"
    DIFFERENTIATOR = "differentiator"
    TARGET_MARKET = "target_market"
    MECHANISM = "mechanism"
    CUSTOM = "custom"


@dataclass
class AtomicClaim:
    """Single atomic content unit."""
    id: str
    text: str
    claim_type: ClaimType
    source: ClaimSource
    source_text: str  # Original text this was extracted from
    confidence: float  # 0-1, how confident we are in extraction
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)  # Named entities
    negated: bool = False  # Whether claim is negated

    def __hash__(self):
        return hash(self.id)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "text": self.text,
            "claim_type": self.claim_type.value,
            "source": self.source.value,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "keywords": self.keywords,
            "entities": self.entities,
            "negated": self.negated
        }


@dataclass
class DecompositionResult:
    """Result of ACU decomposition."""
    idea_id: str
    claims: List[AtomicClaim]
    total_claims: int
    claims_by_type: Dict[str, int]
    claims_by_source: Dict[str, int]
    decomposition_method: str  # "rule_based" or "llm"
    processing_time_ms: float

    def get_claims_of_type(self, claim_type: ClaimType) -> List[AtomicClaim]:
        """Get claims of a specific type."""
        return [c for c in self.claims if c.claim_type == claim_type]

    def get_claims_from_source(self, source: ClaimSource) -> List[AtomicClaim]:
        """Get claims from a specific source."""
        return [c for c in self.claims if c.source == source]

    def to_dict(self) -> Dict:
        return {
            "idea_id": self.idea_id,
            "claims": [c.to_dict() for c in self.claims],
            "total_claims": self.total_claims,
            "claims_by_type": self.claims_by_type,
            "claims_by_source": self.claims_by_source,
            "decomposition_method": self.decomposition_method,
            "processing_time_ms": self.processing_time_ms
        }


class ACUDecomposer:
    """
    Decomposes ideas into Atomic Content Units (ACUs).

    Uses a combination of rule-based extraction and optional LLM enhancement
    to break ideas into their smallest meaningful claims.
    """

    # Claim type detection patterns
    FEATURE_PATTERNS = [
        r"(?:has|includes?|features?|offers?|provides?|contains?)\s+(.+)",
        r"(?:with|using|via)\s+(.+)",
        r"(.+)\s+(?:technology|system|platform|solution|approach)",
    ]

    BENEFIT_PATTERNS = [
        r"(?:enables?|allows?|helps?|improves?|increases?|reduces?|saves?)\s+(.+)",
        r"(?:for|to)\s+(?:better|easier|faster|cheaper)\s+(.+)",
        r"(?:benefit|advantage|value)(?:s)?(?:\s+of|\s+is|\:)\s+(.+)",
    ]

    MECHANISM_PATTERNS = [
        r"(?:works?\s+by|through|via|using)\s+(.+)",
        r"(?:process|method|technique|approach)(?:\s+of|\s+is|\:)\s+(.+)",
        r"(.+)\s+(?:mechanism|process|method)",
    ]

    MARKET_PATTERNS = [
        r"(?:for|targeting|aimed\s+at|designed\s+for)\s+(.+)",
        r"(.+)\s+(?:users?|customers?|consumers?|audience|market|segment)",
        r"(?:ideal|perfect|suited)\s+for\s+(.+)",
    ]

    COMPARISON_PATTERNS = [
        r"(?:unlike|compared\s+to|versus|vs\.?|better\s+than)\s+(.+)",
        r"(?:first|only|unique|novel)\s+(.+)",
        r"(.+)\s+(?:alternative|competitor|replacement)",
    ]

    NOVELTY_PATTERNS = [
        r"(?:new|novel|innovative|revolutionary|breakthrough)\s+(.+)",
        r"(?:first|only|unique)\s+(.+)",
        r"(?:never\s+before|unprecedented)\s+(.+)",
    ]

    # Sentence splitters
    SENTENCE_DELIMITERS = r'[.!?;]|\n|(?:,\s+(?:and|but|or|which|that))'

    # Common stop phrases to filter
    STOP_PHRASES = {
        "the", "a", "an", "this", "that", "these", "those",
        "is", "are", "was", "were", "be", "been", "being",
        "it", "its", "it's"
    }

    def __init__(
        self,
        min_claim_length: int = 5,
        max_claim_length: int = 200,
        llm_callback: Optional[Callable[[str], Dict]] = None,
        use_llm: bool = False
    ):
        """
        Initialize ACU decomposer.

        Args:
            min_claim_length: Minimum characters for valid claim
            max_claim_length: Maximum characters for claim (truncate if longer)
            llm_callback: Optional LLM function for enhanced decomposition
            use_llm: Whether to use LLM for decomposition
        """
        self.min_claim_length = min_claim_length
        self.max_claim_length = max_claim_length
        self.llm_callback = llm_callback
        self.use_llm = use_llm and llm_callback is not None

        # Compile patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.feature_patterns = [re.compile(p, re.IGNORECASE) for p in self.FEATURE_PATTERNS]
        self.benefit_patterns = [re.compile(p, re.IGNORECASE) for p in self.BENEFIT_PATTERNS]
        self.mechanism_patterns = [re.compile(p, re.IGNORECASE) for p in self.MECHANISM_PATTERNS]
        self.market_patterns = [re.compile(p, re.IGNORECASE) for p in self.MARKET_PATTERNS]
        self.comparison_patterns = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
        self.novelty_patterns = [re.compile(p, re.IGNORECASE) for p in self.NOVELTY_PATTERNS]

    def decompose(self, idea: Dict, idea_id: Optional[str] = None) -> DecompositionResult:
        """
        Decompose an idea into atomic claims.

        Args:
            idea: Idea dictionary with title, description, etc.
            idea_id: Optional ID for the idea

        Returns:
            DecompositionResult with all extracted claims
        """
        start_time = datetime.now()

        if idea_id is None:
            idea_id = self._generate_id(idea)

        claims = []

        # Extract from title
        title = idea.get("title", "")
        if title:
            title_claims = self._extract_claims_from_text(
                title, ClaimSource.TITLE
            )
            claims.extend(title_claims)

        # Extract from description
        description = idea.get("description", "")
        if description:
            desc_claims = self._extract_claims_from_text(
                description, ClaimSource.DESCRIPTION
            )
            claims.extend(desc_claims)

        # Extract from differentiators
        differentiators = idea.get("differentiators", [])
        if isinstance(differentiators, list):
            for diff in differentiators:
                diff_claims = self._extract_claims_from_text(
                    diff, ClaimSource.DIFFERENTIATOR
                )
                claims.extend(diff_claims)

        # Extract from target market
        target_market = idea.get("target_market", "")
        if target_market:
            market_claims = self._extract_claims_from_text(
                target_market, ClaimSource.TARGET_MARKET
            )
            claims.extend(market_claims)

        # Extract from mechanism if present
        mechanism = idea.get("mechanism", idea.get("how_it_works", ""))
        if mechanism:
            mech_claims = self._extract_claims_from_text(
                mechanism, ClaimSource.MECHANISM
            )
            claims.extend(mech_claims)

        # Optional LLM enhancement
        if self.use_llm and self.llm_callback:
            llm_claims = self._llm_decompose(idea)
            claims = self._merge_claims(claims, llm_claims)

        # Deduplicate claims
        claims = self._deduplicate_claims(claims)

        # Calculate statistics
        claims_by_type = {}
        for claim in claims:
            type_name = claim.claim_type.value
            claims_by_type[type_name] = claims_by_type.get(type_name, 0) + 1

        claims_by_source = {}
        for claim in claims:
            source_name = claim.source.value
            claims_by_source[source_name] = claims_by_source.get(source_name, 0) + 1

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return DecompositionResult(
            idea_id=idea_id,
            claims=claims,
            total_claims=len(claims),
            claims_by_type=claims_by_type,
            claims_by_source=claims_by_source,
            decomposition_method="llm" if self.use_llm else "rule_based",
            processing_time_ms=processing_time
        )

    def _extract_claims_from_text(
        self,
        text: str,
        source: ClaimSource
    ) -> List[AtomicClaim]:
        """Extract atomic claims from a text segment."""
        claims = []

        # Split into sentences/clauses
        segments = re.split(self.SENTENCE_DELIMITERS, text)
        segments = [s.strip() for s in segments if s.strip()]

        for segment in segments:
            # Skip too short or too long
            if len(segment) < self.min_claim_length:
                continue

            if len(segment) > self.max_claim_length:
                segment = segment[:self.max_claim_length] + "..."

            # Detect claim type
            claim_type, confidence = self._detect_claim_type(segment)

            # Check for negation
            negated = self._is_negated(segment)

            # Extract keywords and entities
            keywords = self._extract_keywords(segment)
            entities = self._extract_entities(segment)

            # Create claim
            claim = AtomicClaim(
                id=self._generate_claim_id(segment, source),
                text=segment,
                claim_type=claim_type,
                source=source,
                source_text=text,
                confidence=confidence,
                keywords=keywords,
                entities=entities,
                negated=negated
            )
            claims.append(claim)

        return claims

    def _detect_claim_type(self, text: str) -> Tuple[ClaimType, float]:
        """Detect the type of claim with confidence score."""
        text_lower = text.lower()

        # Check novelty patterns first (highest priority)
        for pattern in self.novelty_patterns:
            if pattern.search(text_lower):
                return ClaimType.NOVELTY, 0.9

        # Check comparison patterns
        for pattern in self.comparison_patterns:
            if pattern.search(text_lower):
                return ClaimType.COMPARISON, 0.85

        # Check benefit patterns
        for pattern in self.benefit_patterns:
            if pattern.search(text_lower):
                return ClaimType.BENEFIT, 0.8

        # Check mechanism patterns
        for pattern in self.mechanism_patterns:
            if pattern.search(text_lower):
                return ClaimType.MECHANISM, 0.8

        # Check market patterns
        for pattern in self.market_patterns:
            if pattern.search(text_lower):
                return ClaimType.MARKET, 0.8

        # Check feature patterns
        for pattern in self.feature_patterns:
            if pattern.search(text_lower):
                return ClaimType.FEATURE, 0.75

        # Check for constraint indicators
        constraint_indicators = ["must", "require", "need", "only", "limited", "cannot", "won't"]
        if any(ind in text_lower for ind in constraint_indicators):
            return ClaimType.CONSTRAINT, 0.7

        # Default to unknown with low confidence
        return ClaimType.UNKNOWN, 0.5

    def _is_negated(self, text: str) -> bool:
        """Check if claim contains negation."""
        negation_words = [
            "not", "no", "never", "none", "neither", "nor",
            "cannot", "can't", "won't", "wouldn't", "shouldn't",
            "doesn't", "don't", "didn't", "isn't", "aren't",
            "wasn't", "weren't", "without", "lack", "absent"
        ]
        text_lower = text.lower()
        return any(neg in text_lower for neg in negation_words)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Simple keyword extraction - remove stop words, keep nouns/verbs
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in self.STOP_PHRASES]

        # Keep unique keywords, preserve order
        seen = set()
        unique = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique.append(kw)

        return unique[:10]  # Top 10 keywords

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (capitalized phrases)."""
        # Simple entity extraction - capitalized words/phrases
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)

        # Filter out sentence starters
        entities = [e for e in entities if len(e) > 2]

        return list(set(entities))[:5]  # Top 5 unique entities

    def _generate_id(self, idea: Dict) -> str:
        """Generate unique ID for idea."""
        content = str(idea.get("title", "")) + str(idea.get("description", ""))
        return hashlib.md5(content.encode()).hexdigest()[:8]

    def _generate_claim_id(self, text: str, source: ClaimSource) -> str:
        """Generate unique ID for claim."""
        content = f"{source.value}:{text}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _deduplicate_claims(self, claims: List[AtomicClaim]) -> List[AtomicClaim]:
        """Remove duplicate or near-duplicate claims."""
        seen_texts = set()
        unique_claims = []

        for claim in claims:
            # Normalize text for comparison
            normalized = claim.text.lower().strip()
            normalized = re.sub(r'\s+', ' ', normalized)

            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_claims.append(claim)

        return unique_claims

    def _llm_decompose(self, idea: Dict) -> List[AtomicClaim]:
        """Use LLM to decompose idea into claims."""
        if not self.llm_callback:
            return []

        prompt = f"""Decompose this idea into atomic claims (smallest meaningful units).
For each claim, identify:
1. The claim text (one specific assertion)
2. Type: feature, benefit, mechanism, market, constraint, comparison, novelty
3. Confidence (0-1)

Idea:
Title: {idea.get('title', '')}
Description: {idea.get('description', '')}
Differentiators: {idea.get('differentiators', [])}
Target Market: {idea.get('target_market', '')}

Return as JSON array of claims."""

        try:
            result = self.llm_callback(prompt)
            if isinstance(result, dict) and "claims" in result:
                claims = []
                for c in result["claims"]:
                    claim = AtomicClaim(
                        id=self._generate_claim_id(c.get("text", ""), ClaimSource.CUSTOM),
                        text=c.get("text", ""),
                        claim_type=ClaimType(c.get("type", "unknown")),
                        source=ClaimSource.CUSTOM,
                        source_text="LLM extraction",
                        confidence=c.get("confidence", 0.8),
                        keywords=c.get("keywords", []),
                        entities=c.get("entities", []),
                        negated=c.get("negated", False)
                    )
                    claims.append(claim)
                return claims
        except Exception:
            pass

        return []

    def _merge_claims(
        self,
        rule_claims: List[AtomicClaim],
        llm_claims: List[AtomicClaim]
    ) -> List[AtomicClaim]:
        """Merge rule-based and LLM claims, preferring LLM when overlap."""
        # Start with LLM claims (higher quality)
        merged = list(llm_claims)

        # Add rule-based claims that don't overlap
        llm_texts = {c.text.lower().strip() for c in llm_claims}

        for claim in rule_claims:
            normalized = claim.text.lower().strip()
            # Check for significant overlap
            if not any(self._text_overlap(normalized, lt) > 0.7 for lt in llm_texts):
                merged.append(claim)

        return merged

    def _text_overlap(self, text1: str, text2: str) -> float:
        """Calculate word overlap ratio between two texts."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)


def decompose_idea(idea: Dict, use_llm: bool = False) -> DecompositionResult:
    """Convenience function to decompose a single idea."""
    decomposer = ACUDecomposer(use_llm=use_llm)
    return decomposer.decompose(idea)


def create_test_idea() -> Dict:
    """Create a test idea for decomposition."""
    return {
        "title": "AI-Powered Protein Optimization Platform",
        "description": "Uses machine learning to analyze individual nutrition needs and "
                      "optimize protein blend formulations. The system provides personalized "
                      "recommendations based on workout patterns, dietary restrictions, and goals.",
        "differentiators": [
            "First AI-driven personalization in protein supplements",
            "Real-time adaptation based on user feedback",
            "Integration with fitness wearables"
        ],
        "target_market": "Health-conscious fitness enthusiasts aged 25-45 who track their workouts"
    }
