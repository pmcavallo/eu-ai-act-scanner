"""
Priority scoring and obligation gap analysis for the EU AI Act Compliance Scanner.

Loads regulatory obligations and scores compliance gaps based on classification results.
"""
import json
from pathlib import Path
from typing import Optional

from src.shared.types import (
    ClassificationResult,
    ObligationGap,
    Priority,
    RiskLevel,
    Role,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "regulations"
OBLIGATIONS_FILE = DATA_DIR / "obligations.json"

PRIORITY_MAP = {
    "critical": Priority.CRITICAL,
    "high": Priority.HIGH,
    "medium": Priority.MEDIUM,
    "low": Priority.LOW,
}

# Action item templates keyed by obligation ID
_PROVIDER_ACTION_ITEMS: dict[str, list[str]] = {
    "P1": [
        "Establish a documented risk management system covering the AI system's full lifecycle",
        "Conduct a formal risk assessment identifying foreseeable risks from intended use and misuse",
        "Define and implement risk mitigation measures with measurable acceptance criteria",
    ],
    "P2": [
        "Create a data governance policy covering collection, preparation, labelling, and bias examination",
        "Audit training and evaluation datasets for representativeness, gaps, and potential biases",
        "Document statistical properties and relevance of all datasets used",
    ],
    "P3": [
        "Draft technical documentation per Annex IV before the system is placed on the market",
        "Include system description, development process, risk management, and monitoring details",
        "Establish a process for keeping documentation up to date through the system's lifecycle",
    ],
    "P4": [
        "Design automatic logging capabilities into the system architecture",
        "Ensure logs capture period of use, input data references, and involved natural persons",
    ],
    "P5": [
        "Prepare instructions for use that describe system capabilities, limitations, and intended purpose",
        "Include provider identity, performance characteristics, and human oversight measures",
        "Document the expected lifetime, maintenance needs, and any pre-determined changes",
    ],
    "P6": [
        "Implement a human-in-the-loop mechanism so operators can review, override, or halt AI outputs",
        "Ensure oversight personnel have the competence and authority to intervene",
        "Build and test a kill switch that can interrupt the system in real time",
    ],
    "P7": [
        "Define and publish accuracy benchmarks for the intended use case",
        "Conduct robustness testing against errors, faults, and adversarial inputs",
        "Implement cybersecurity measures protecting the system from unauthorized manipulation",
    ],
    "P8": [
        "Determine whether internal or third-party conformity assessment applies (third-party for biometrics)",
        "Plan and schedule the conformity assessment before market placement or deployment",
    ],
    "P9": [
        "Draft an EU declaration of conformity stating compliance with Chapter III Section 2 requirements",
        "Establish a process for keeping the declaration up to date",
    ],
    "P10": [
        "Register the high-risk AI system in the EU database (Article 71) before market placement",
        "Assign responsibility for maintaining the registration as the system evolves",
    ],
}

_DEPLOYER_ACTION_ITEMS: dict[str, list[str]] = {
    "D1": [
        "Obtain and review the provider's instructions for use",
        "Ensure operational procedures align with the provider's intended use and documented limitations",
    ],
    "D2": [
        "Assign human oversight to personnel with relevant competence, training, and authority",
        "Document the oversight process and escalation procedures",
    ],
    "D3": [
        "Review input data for relevance and representativeness given the intended purpose",
        "Establish a process to monitor input data quality on an ongoing basis",
    ],
    "D4": [
        "Implement ongoing monitoring of the AI system's operation per the provider's instructions",
        "Define thresholds and escalation procedures for anomalous system behavior",
    ],
    "D5": [
        "Retain automatically generated logs for a period appropriate to the intended purpose",
        "Ensure log storage complies with data protection requirements",
    ],
    "D6": [
        "Inform workers' representatives and affected workers about the use of the high-risk AI system",
        "Document the notification process and retain evidence of disclosure",
    ],
    "D7": [
        "Conduct a data protection impact assessment (DPIA) under GDPR using information from the provider",
        "Document the DPIA findings and any mitigating measures taken",
    ],
}


def _load_obligations() -> dict:
    """Load obligations data from the regulations JSON file."""
    with open(OBLIGATIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_obligation_gap(
    ob: dict, action_items: list[str]
) -> ObligationGap:
    """Create an ObligationGap from a raw obligation dict and action items."""
    priority_str = ob.get("priority", "medium")
    return ObligationGap(
        obligation_id=ob["id"],
        article=ob["article"],
        name=ob["name"],
        description=ob["description"],
        priority=PRIORITY_MAP.get(priority_str, Priority.MEDIUM),
        status="not_met",
        action_items=action_items,
    )


def score_obligations(
    classification: ClassificationResult,
) -> list[ObligationGap]:
    """Score and prioritize compliance gaps based on classification results.

    Determines which obligations apply based on the organization's role
    (provider, deployer, or both) and returns sorted ObligationGap entries.

    Args:
        classification: The classification result from the classifier agent.

    Returns:
        List of ObligationGap objects sorted by priority (critical first).
    """
    if classification.risk_level in (RiskLevel.MINIMAL_RISK, RiskLevel.LIMITED_RISK):
        return []

    if classification.risk_level == RiskLevel.UNACCEPTABLE:
        return [
            ObligationGap(
                obligation_id="BANNED",
                article="Article 5",
                name="Prohibited AI Practice",
                description="This AI system falls under a prohibited practice and cannot be deployed in the EU.",
                priority=Priority.CRITICAL,
                status="not_met",
                action_items=[
                    "Immediately cease deployment of this AI system in the EU",
                    "Consult legal counsel regarding potential penalties under Article 99",
                    "Evaluate whether the system can be redesigned to avoid the prohibited category",
                ],
            )
        ]

    data = _load_obligations()
    gaps: list[ObligationGap] = []

    if classification.role in (Role.PROVIDER, Role.BOTH):
        for ob in data.get("provider_obligations", []):
            actions = _PROVIDER_ACTION_ITEMS.get(ob["id"], [])
            gaps.append(_build_obligation_gap(ob, actions))

    if classification.role in (Role.DEPLOYER, Role.BOTH):
        for ob in data.get("deployer_obligations", []):
            actions = _DEPLOYER_ACTION_ITEMS.get(ob["id"], [])
            gaps.append(_build_obligation_gap(ob, actions))

    priority_order = {
        Priority.CRITICAL: 0,
        Priority.HIGH: 1,
        Priority.MEDIUM: 2,
        Priority.LOW: 3,
    }
    gaps.sort(key=lambda g: priority_order.get(g.priority, 99))

    return gaps


def calculate_risk_score(
    classification: ClassificationResult,
    gaps: list[ObligationGap],
) -> float:
    """Calculate a 0-100 risk score based on classification and unmet obligations.

    Scoring formula:
    - Base score from risk level and role
    - Adjustment for unmet obligation count
    - Bonus penalty for Article 25 triggers, especially scenario (c)

    Args:
        classification: The classification result from the classifier agent.
        gaps: List of obligation gaps from score_obligations.

    Returns:
        A float risk score between 0 and 100.
    """
    if classification.risk_level == RiskLevel.UNACCEPTABLE:
        return 100.0

    if classification.risk_level == RiskLevel.MINIMAL_RISK:
        return 5.0

    if classification.risk_level == RiskLevel.LIMITED_RISK:
        base = 30.0
        return min(base, 40.0)

    # High-risk scoring
    if classification.role == Role.PROVIDER or classification.role == Role.BOTH:
        base = 80.0
    else:
        base = 60.0

    # Adjust based on unmet obligations
    unmet_count = sum(1 for g in gaps if g.status == "not_met")
    total_count = len(gaps) if gaps else 1
    obligation_penalty = (unmet_count / total_count) * 15.0

    # Article 25 adjustment
    article_25_penalty = 0.0
    if classification.article_25_trigger:
        scenario = classification.article_25_trigger.scenario_number
        if scenario == 3:  # scenario (c) is the most dangerous
            article_25_penalty = 5.0
        elif scenario == 2:
            article_25_penalty = 3.0
        elif scenario == 1:
            article_25_penalty = 2.0

    score = base + obligation_penalty + article_25_penalty
    return min(score, 100.0)
