"""
Rule-based classification functions for the EU AI Act Compliance Scanner.

Matches AI system descriptions against Annex III categories, determines
provider/deployer roles, detects Article 25 triggers, and checks GPAI
provider terms.
"""
import re
from typing import Optional

from src.shared.types import (
    AnnexIIIMatch,
    Article25Trigger,
    ClassificationResult,
    GPAITermsCheck,
    RiskLevel,
    Role,
)

# ---------------------------------------------------------------------------
# GPAI provider detection patterns
# ---------------------------------------------------------------------------
GPAI_PATTERNS: dict[str, list[str]] = {
    "openai": [
        r"\bgpt[-\s]?4\b", r"\bgpt[-\s]?3\.?5\b", r"\bchatgpt\b",
        r"\bopenai\b", r"\bgpt[-\s]?4o\b", r"\bgpt[-\s]?4\s*turbo\b",
        r"\bdall[-\s]?e\b", r"\bwhisper\b",
    ],
    "anthropic": [
        r"\bclaude\b", r"\banthropic\b",
    ],
    "google": [
        r"\bgemini\b", r"\bgemini\s*pro\b", r"\bgemini\s*ultra\b",
        r"\bgemini\s*nano\b",
    ],
    "microsoft": [
        r"\bazure\s*openai\b", r"\bcopilot\b", r"\bbing\s*chat\b",
    ],
    "meta": [
        r"\bllama\b", r"\bmeta\s*ai\b",
    ],
}

# ---------------------------------------------------------------------------
# Fraud detection exclusion keywords (Annex III 5a explicitly excludes fraud)
# ---------------------------------------------------------------------------
FRAUD_KEYWORDS: list[str] = [
    "fraud", "fraudulent", "anti-fraud", "fraud detection",
    "fraud prevention", "suspicious transaction", "transaction monitoring",
    "money laundering", "aml",
]

# ---------------------------------------------------------------------------
# Keywords signaling the org is NOT modifying / building, just deploying
# ---------------------------------------------------------------------------
PURE_DEPLOYER_CUES: list[str] = [
    "purchased", "bought", "use it exactly as provided",
    "no modifications", "off-the-shelf", "as-is", "out of the box",
    "vendor-provided", "no changes", "without modification",
    "use as provided", "use it as provided",
]

# ---------------------------------------------------------------------------
# Unacceptable risk keywords (Article 5)
# ---------------------------------------------------------------------------
UNACCEPTABLE_KEYWORDS: list[str] = [
    "social scoring", "social credit", "real-time biometric",
    "mass surveillance", "subliminal manipulation",
    "exploit vulnerabilities", "emotion recognition in workplace",
    "predictive policing based on profiling",
]


def detect_gpai_provider(
    description: str, gpai_data: dict
) -> Optional[str]:
    """Detect which GPAI provider's product is being used, if any.

    Args:
        description: Plain-English AI system description.
        gpai_data: Parsed gpai_providers.json data.

    Returns:
        Provider key (e.g. "openai") or None.
    """
    desc_lower = description.lower()

    # First try pattern-based detection from our constants
    for provider, patterns in GPAI_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, desc_lower):
                return provider

    # Also try matching against product names in the data file
    providers: dict = gpai_data.get("providers", {})
    for provider_key, provider_info in providers.items():
        products = provider_info.get("products", [])
        for product in products:
            if product.lower() in desc_lower:
                return provider_key

    return None


def match_annex_iii(
    description: str, annex_data: dict
) -> Optional[AnnexIIIMatch]:
    """Match a description against Annex III high-risk categories.

    Uses keyword matching with scoring. Returns the best match above a
    confidence threshold, or None if no match.

    Handles the fraud detection exclusion: Annex III Category 5a explicitly
    excludes AI systems used for detecting financial fraud.

    Args:
        description: Plain-English AI system description.
        annex_data: Parsed annex_iii.json data.

    Returns:
        Best AnnexIIIMatch or None.
    """
    desc_lower = description.lower()

    # Check fraud exclusion early — if this is clearly a fraud detection
    # system, it is excluded from 5a (credit scoring).
    is_fraud_system = _is_fraud_detection(desc_lower)

    best_match: Optional[AnnexIIIMatch] = None
    best_score: float = 0.0

    categories: list[dict] = annex_data.get("categories", [])
    for category in categories:
        cat_id: int = category["id"]
        cat_name: str = category["name"]
        subcategories: list[dict] = category.get("subcategories", [])

        for sub in subcategories:
            sub_id: str = sub["id"]
            sub_desc: str = sub.get("description", "")

            # Skip 5a for fraud detection systems
            if sub_id == "5a" and is_fraud_system:
                continue

            keywords: list[str] = sub.get("keywords", [])
            examples: list[str] = sub.get("examples", [])

            score, matched_terms = _score_match(
                desc_lower, keywords, examples, sub_desc, sub_id
            )

            if score > best_score:
                best_score = score
                reasoning = (
                    f"Matched Annex III {sub_id} ({cat_name}) "
                    f"based on: {', '.join(matched_terms[:5])}"
                )
                best_match = AnnexIIIMatch(
                    category_id=cat_id,
                    subcategory_id=sub_id,
                    category_name=cat_name,
                    description=sub_desc,
                    confidence=min(score, 1.0),
                    reasoning=reasoning,
                )

    # Require minimum confidence
    if best_match and best_match.confidence < 0.2:
        return None

    return best_match


def detect_article_25_triggers(
    description: str,
    article_25_data: dict,
    annex_match: Optional[AnnexIIIMatch],
    gpai_provider: Optional[str] = None,
) -> Optional[Article25Trigger]:
    """Detect Article 25 deployer-to-provider role-shift triggers.

    Checks three scenarios:
      (a) Rebranding — org puts their name on someone else's high-risk AI
      (b) Substantial modification — org significantly changes the AI system
      (c) Modified intended purpose — org uses GPAI for a high-risk purpose

    Args:
        description: Plain-English AI system description.
        article_25_data: Parsed article_25.json data.
        annex_match: The Annex III match result (or None).
        gpai_provider: Detected GPAI provider name (or None).

    Returns:
        Best Article25Trigger or None.
    """
    desc_lower = description.lower()
    scenarios: list[dict] = article_25_data.get("scenarios", [])

    # ------------------------------------------------------------------
    # Scenario (c): Modified intended purpose — check first because it
    # is the most common and dangerous.
    # A GPAI is used for a high-risk (Annex III) purpose.
    # ------------------------------------------------------------------
    if gpai_provider and annex_match:
        scenario_c = _find_scenario(scenarios, "25_1_c")
        if scenario_c:
            return Article25Trigger(
                scenario_id="25_1_c",
                scenario_number=3,
                short_name="Modified Intended Purpose",
                confidence=0.95,
                reasoning=(
                    f"GPAI product from {gpai_provider} is being used for "
                    f"Annex III category {annex_match.subcategory_id} "
                    f"({annex_match.category_name}). This modifies the "
                    f"intended purpose of a general-purpose AI system to a "
                    f"high-risk use, triggering Article 25(1)(c)."
                ),
            )

    # ------------------------------------------------------------------
    # Scenario (a): Rebranding
    # ------------------------------------------------------------------
    scenario_a = _find_scenario(scenarios, "25_1_a")
    if scenario_a:
        rebrand_keywords = scenario_a.get("detection_keywords", [])
        rebrand_cues = [
            "our brand", "our company brand", "under our brand",
            "our name", "offer it to", "offer under",
            "offer it under", "market under",
        ]
        all_rebrand = rebrand_keywords + rebrand_cues
        matched = [kw for kw in all_rebrand if kw.lower() in desc_lower]
        if matched and annex_match:
            return Article25Trigger(
                scenario_id="25_1_a",
                scenario_number=1,
                short_name="Rebranding",
                confidence=0.9,
                reasoning=(
                    f"Description indicates rebranding of a high-risk AI "
                    f"system (matched cues: {', '.join(matched[:3])}). "
                    f"Placing your name/trademark on a high-risk system "
                    f"triggers Article 25(1)(a)."
                ),
            )

    # ------------------------------------------------------------------
    # Scenario (b): Substantial modification
    # ------------------------------------------------------------------
    scenario_b = _find_scenario(scenarios, "25_1_b")
    if scenario_b:
        mod_keywords = scenario_b.get("detection_keywords", [])
        mod_cues = [
            "fine-tuned", "fine tuned", "retrained", "modified",
            "customized", "adapted", "changed architecture",
            "transfer learning", "our own data", "our data",
            "took an", "took a",
        ]
        all_mod = mod_keywords + mod_cues
        matched = [kw for kw in all_mod if kw.lower() in desc_lower]
        if matched:
            return Article25Trigger(
                scenario_id="25_1_b",
                scenario_number=2,
                short_name="Substantial Modification",
                confidence=0.85,
                reasoning=(
                    f"Description indicates substantial modification of an "
                    f"AI system (matched cues: {', '.join(matched[:3])}). "
                    f"Fine-tuning, retraining, or modifying model weights "
                    f"triggers Article 25(1)(b)."
                ),
            )

    return None


def check_gpai_terms(
    description: str,
    provider_name: str,
    annex_match: Optional[AnnexIIIMatch],
    gpai_data: dict,
) -> GPAITermsCheck:
    """Check if the use likely violates the GPAI provider's terms.

    High-risk use (Annex III match) of a GPAI system typically violates
    the provider's acceptable use policy.

    Args:
        description: Plain-English AI system description.
        provider_name: GPAI provider key (e.g. "openai").
        annex_match: The Annex III match result (or None).
        gpai_data: Parsed gpai_providers.json data.

    Returns:
        GPAITermsCheck with violation details.
    """
    providers: dict = gpai_data.get("providers", {})
    provider_info: dict = providers.get(provider_name, {})

    if not provider_info:
        return GPAITermsCheck(
            provider_name=provider_name,
            product_used=None,
            terms_violated=False,
            reasoning=f"No terms data available for provider '{provider_name}'.",
        )

    # Detect which product is mentioned
    product_used = _detect_product(description, provider_info)

    # If there is an Annex III match, the use is high-risk → almost
    # certainly violates the GPAI provider's terms.
    if annex_match:
        prohibited_uses = provider_info.get(
            "prohibited_uses_relevant_to_annex_iii", []
        )
        high_risk_restrictions = provider_info.get(
            "high_risk_restrictions", []
        )
        violated_terms = prohibited_uses + high_risk_restrictions

        return GPAITermsCheck(
            provider_name=provider_info.get("provider_name", provider_name),
            product_used=product_used,
            terms_violated=True,
            violated_terms=violated_terms,
            reasoning=(
                f"Using {provider_info.get('provider_name', provider_name)}'s "
                f"GPAI for Annex III category "
                f"{annex_match.subcategory_id} ({annex_match.category_name}) "
                f"likely violates the provider's acceptable use policy, which "
                f"restricts automated high-stakes decision-making."
            ),
        )

    return GPAITermsCheck(
        provider_name=provider_info.get("provider_name", provider_name),
        product_used=product_used,
        terms_violated=False,
        reasoning=(
            f"Use case does not appear to fall under Annex III high-risk "
            f"categories. No apparent terms violation detected for "
            f"{provider_info.get('provider_name', provider_name)}."
        ),
    )


def detect_transparency_obligations(
    description: str, obligations_data: dict
) -> list[str]:
    """Detect Article 50 transparency obligation triggers.

    Args:
        description: Plain-English AI system description.
        obligations_data: Parsed obligations.json data.

    Returns:
        List of applicable transparency obligation descriptions.
    """
    desc_lower = description.lower()
    triggered: list[str] = []

    transparency = obligations_data.get("transparency_obligations", {})
    requirements: list[dict] = transparency.get("requirements", [])

    for req in requirements:
        trigger: str = req.get("trigger", "").lower()
        obligation: str = req.get("obligation", "")
        req_id: str = req.get("id", "")

        # T1: Direct interaction with people
        if req_id == "T1":
            interaction_cues = [
                "chatbot", "chat bot", "customer service",
                "interact", "users", "patients", "students",
                "candidates", "applicants", "customers",
            ]
            if any(cue in desc_lower for cue in interaction_cues):
                triggered.append(obligation)

        # T2: Synthetic content generation
        elif req_id == "T2":
            generation_cues = [
                "generate", "draft", "write", "create content",
                "email campaigns", "social media posts", "copy",
                "content creation", "text generation",
            ]
            if any(cue in desc_lower for cue in generation_cues):
                triggered.append(obligation)

        # T3: Emotion recognition / biometric categorisation
        elif req_id == "T3":
            biometric_cues = [
                "emotion recognition", "facial expression",
                "biometric categorisation", "sentiment analysis from face",
                "demographic inference",
            ]
            if any(cue in desc_lower for cue in biometric_cues):
                triggered.append(obligation)

    return triggered


def determine_risk_level(
    annex_match: Optional[AnnexIIIMatch], description: str
) -> RiskLevel:
    """Determine the overall risk level for an AI system.

    Args:
        annex_match: The Annex III match result (or None).
        description: Plain-English AI system description.

    Returns:
        The determined RiskLevel.
    """
    desc_lower = description.lower()

    # Check unacceptable risk first
    if any(kw in desc_lower for kw in UNACCEPTABLE_KEYWORDS):
        return RiskLevel.UNACCEPTABLE

    # If there is an Annex III match, it is high-risk
    if annex_match:
        return RiskLevel.HIGH_RISK

    # Check for limited risk indicators (AI interaction, content generation)
    limited_cues = [
        "chatbot", "chat bot", "customer service",
        "content generation", "draft", "email campaign",
        "social media", "creative", "writing assistant",
        "text generation", "copywriting", "marketing",
        "generate", "assistant",
    ]
    if any(cue in desc_lower for cue in limited_cues):
        return RiskLevel.LIMITED_RISK

    return RiskLevel.MINIMAL_RISK


def determine_role(
    description: str,
    annex_match: Optional[AnnexIIIMatch],
    article_25_trigger: Optional[Article25Trigger],
) -> Role:
    """Determine whether the organization is provider, deployer, or both.

    Logic:
    - If Article 25 is triggered → PROVIDER (the org has shifted from
      deployer to provider).
    - If the org is clearly just using a vendor product as-is → DEPLOYER.
    - If the org built/developed the system → PROVIDER.
    - Default for ambiguous cases → DEPLOYER.

    Args:
        description: Plain-English AI system description.
        annex_match: The Annex III match result (or None).
        article_25_trigger: The detected Article 25 trigger (or None).

    Returns:
        The determined Role.
    """
    desc_lower = description.lower()

    # Article 25 trigger means the org became a provider
    if article_25_trigger:
        return Role.PROVIDER

    # Check for pure deployer signals
    if any(cue in desc_lower for cue in PURE_DEPLOYER_CUES):
        return Role.DEPLOYER

    # Check for builder/developer signals (without Article 25)
    builder_cues = [
        "we built", "we developed", "we created", "we designed",
        "our team built", "our team developed", "we trained",
        "we implemented",
    ]
    if any(cue in desc_lower for cue in builder_cues):
        return Role.PROVIDER

    # Default: if using a GPAI without high-risk purpose, they are a deployer
    return Role.DEPLOYER


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_fraud_detection(desc_lower: str) -> bool:
    """Check if the description is about fraud detection."""
    return any(kw in desc_lower for kw in FRAUD_KEYWORDS)


_STOP_WORDS: set[str] = {
    "the", "a", "an", "to", "of", "in", "for", "and", "or",
    "be", "is", "are", "was", "were", "that", "this", "it",
    "by", "on", "with", "as", "at", "from", "not", "used",
    "intended", "systems", "ai", "system", "natural", "persons",
    "based", "their", "such", "who", "our", "we", "all",
    "no", "its", "has", "have", "been", "does", "do",
}


# Contextual domain phrases — these boost a category when the description
# contains strong multi-word or domain-specific signals that a single keyword
# list may not capture.  Each maps a subcategory id to a list of regex
# patterns that, if matched, add a scoring bonus.
_DOMAIN_BOOST: dict[str, list[str]] = {
    "4a": [
        r"\bhiring\b.*\btool\b", r"\bhr\b", r"\bhuman\s*resources\b",
        r"\brecruitment\b.*\btool\b", r"\bats\b",
        r"\bhiring\s+assessment\b",
    ],
    "4b": [r"\bperformance\s+review\b", r"\bemployee\s+monitoring\b"],
    "5a": [r"\bloan\s+approv", r"\bcredit\s+scor"],
}


def _score_match(
    desc_lower: str,
    keywords: list[str],
    examples: list[str],
    sub_description: str,
    subcategory_id: str = "",
) -> tuple[float, list[str]]:
    """Score how well a description matches a subcategory.

    Returns (score, list_of_matched_terms).
    """
    matched_terms: list[str] = []
    score: float = 0.0

    # Keyword matches (strongest signal)
    for kw in keywords:
        if kw.lower() in desc_lower:
            score += 0.3
            matched_terms.append(kw)

    # Domain-specific contextual boost
    for pattern in _DOMAIN_BOOST.get(subcategory_id, []):
        if re.search(pattern, desc_lower):
            score += 0.2
            matched_terms.append(f"domain:{pattern[:20]}")

    # Example matches — only count meaningful (non-stop) word overlap
    for ex in examples:
        ex_words = set(ex.lower().split()) - _STOP_WORDS
        meaningful_matches = [w for w in ex_words if w in desc_lower]
        if len(meaningful_matches) >= 3:
            score += 0.15
            matched_terms.append(f"example: {ex[:40]}...")

    # Sub-description word overlap (only meaningful words)
    desc_words = set(sub_description.lower().split()) - _STOP_WORDS
    input_words = set(desc_lower.split()) - _STOP_WORDS
    meaningful_overlap = desc_words & input_words
    if meaningful_overlap:
        score += len(meaningful_overlap) * 0.05
        matched_terms.extend(list(meaningful_overlap)[:3])

    return score, matched_terms


def _find_scenario(scenarios: list[dict], scenario_id: str) -> Optional[dict]:
    """Find a scenario by ID in the article_25 data."""
    for s in scenarios:
        if s.get("id") == scenario_id:
            return s
    return None


def _detect_product(description: str, provider_info: dict) -> Optional[str]:
    """Detect which specific product is mentioned in the description."""
    desc_lower = description.lower()
    products: list[str] = provider_info.get("products", [])
    for product in products:
        if product.lower() in desc_lower:
            return product
    return None
