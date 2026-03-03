"""
Main classifier orchestration for the EU AI Act Compliance Scanner.

Loads regulatory JSON data and applies rule-based classification to
determine risk level, provider/deployer role, Article 25 triggers,
and GPAI terms compliance.
"""
import json
from pathlib import Path
from typing import Optional

from src.classifier.rules import (
    check_gpai_terms,
    detect_article_25_triggers,
    detect_gpai_provider,
    detect_transparency_obligations,
    determine_risk_level,
    determine_role,
    match_annex_iii,
)
from src.shared.types import (
    AnnexIIIMatch,
    Article25Trigger,
    ClassificationResult,
    GPAITermsCheck,
    RiskLevel,
    Role,
)

DATA_DIR: Path = Path(__file__).parent.parent.parent / "data"


def _load_json(path: Path) -> dict:
    """Load and parse a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def classify(description: str) -> ClassificationResult:
    """Classify an AI system description under the EU AI Act.

    Pipeline order:
      1. Load regulatory data
      2. Detect GPAI provider
      3. Match Annex III category
      4. Determine risk level
      5. Detect Article 25 triggers
      6. Check GPAI terms
      7. Determine role
      8. Detect transparency obligations

    Args:
        description: Plain-English description of how the organization
            uses AI.

    Returns:
        A fully populated ClassificationResult.
    """
    # 1. Load all regulatory data
    annex_data = _load_json(DATA_DIR / "regulations" / "annex_iii.json")
    article_25_data = _load_json(DATA_DIR / "regulations" / "article_25.json")
    obligations_data = _load_json(DATA_DIR / "regulations" / "obligations.json")
    gpai_data = _load_json(DATA_DIR / "provider_terms" / "gpai_providers.json")

    # 2. Detect GPAI provider
    gpai_provider: Optional[str] = detect_gpai_provider(description, gpai_data)

    # 3. Match Annex III category
    annex_match: Optional[AnnexIIIMatch] = match_annex_iii(
        description, annex_data
    )

    # 4. Determine risk level
    risk_level: RiskLevel = determine_risk_level(annex_match, description)

    # 5. Detect Article 25 triggers
    article_25_trigger: Optional[Article25Trigger] = detect_article_25_triggers(
        description, article_25_data, annex_match, gpai_provider
    )

    # 6. Check GPAI terms (only if a GPAI provider was detected)
    gpai_terms_check: Optional[GPAITermsCheck] = None
    if gpai_provider:
        gpai_terms_check = check_gpai_terms(
            description, gpai_provider, annex_match, gpai_data
        )

    # 7. Determine role
    role: Role = determine_role(description, annex_match, article_25_trigger)

    # 8. Detect transparency obligations
    transparency: list[str] = detect_transparency_obligations(
        description, obligations_data
    )

    # Build overall confidence
    confidence = _compute_confidence(
        annex_match, article_25_trigger, risk_level
    )

    # Build reasoning summary
    reasoning = _build_reasoning(
        risk_level, annex_match, role, article_25_trigger,
        gpai_provider, gpai_terms_check,
    )

    return ClassificationResult(
        risk_level=risk_level,
        annex_iii_match=annex_match,
        role=role,
        article_25_trigger=article_25_trigger,
        gpai_terms_check=gpai_terms_check,
        transparency_obligations=transparency,
        confidence=confidence,
        reasoning=reasoning,
    )


def _compute_confidence(
    annex_match: Optional[AnnexIIIMatch],
    article_25_trigger: Optional[Article25Trigger],
    risk_level: RiskLevel,
) -> float:
    """Compute overall classification confidence."""
    scores: list[float] = []

    if annex_match:
        scores.append(annex_match.confidence)
    if article_25_trigger:
        scores.append(article_25_trigger.confidence)

    # High/unacceptable risk with strong matches → high confidence
    if risk_level in (RiskLevel.HIGH_RISK, RiskLevel.UNACCEPTABLE) and scores:
        return sum(scores) / len(scores)

    # Limited/minimal risk without Annex III match → moderate confidence
    if risk_level in (RiskLevel.LIMITED_RISK, RiskLevel.MINIMAL_RISK):
        return 0.7

    return 0.5


def _build_reasoning(
    risk_level: RiskLevel,
    annex_match: Optional[AnnexIIIMatch],
    role: Role,
    article_25_trigger: Optional[Article25Trigger],
    gpai_provider: Optional[str],
    gpai_terms_check: Optional[GPAITermsCheck],
) -> str:
    """Build a human-readable reasoning summary."""
    parts: list[str] = []

    parts.append(f"Risk Level: {risk_level.value}.")

    if annex_match:
        parts.append(
            f"Annex III match: Category {annex_match.subcategory_id} "
            f"({annex_match.category_name}) — {annex_match.reasoning}."
        )
    else:
        parts.append("No Annex III high-risk category match found.")

    parts.append(f"Role: {role.value}.")

    if article_25_trigger:
        parts.append(
            f"Article 25 trigger: {article_25_trigger.short_name} "
            f"({article_25_trigger.scenario_id}) — "
            f"{article_25_trigger.reasoning}"
        )

    if gpai_provider:
        parts.append(f"GPAI provider detected: {gpai_provider}.")
        if gpai_terms_check and gpai_terms_check.terms_violated:
            parts.append(
                f"Terms violation: {gpai_terms_check.reasoning}"
            )

    return " ".join(parts)
