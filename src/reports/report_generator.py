"""
Report generator for the EU AI Act Compliance Scanner.

Takes classification output and produces a structured ComplianceReport
with executive summary, obligation gaps, and prioritized next steps.
"""
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.shared.types import (
    ClassificationResult,
    ComplianceReport,
    ObligationGap,
    Priority,
    RiskLevel,
    Role,
)
from src.reports.priority_scorer import calculate_risk_score, score_obligations

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _role_label(role: Role) -> str:
    """Human-readable role label."""
    labels = {
        Role.PROVIDER: "PROVIDER",
        Role.DEPLOYER: "DEPLOYER",
        Role.BOTH: "PROVIDER (via Article 25 role shift)",
    }
    return labels.get(role, role.value.upper())


def _risk_label(level: RiskLevel) -> str:
    """Human-readable risk level label."""
    labels = {
        RiskLevel.UNACCEPTABLE: "UNACCEPTABLE (Banned)",
        RiskLevel.HIGH_RISK: "HIGH-RISK",
        RiskLevel.LIMITED_RISK: "LIMITED RISK",
        RiskLevel.MINIMAL_RISK: "MINIMAL RISK",
    }
    return labels.get(level, level.value.upper())


def _build_executive_summary(
    classification: ClassificationResult,
    gaps: list[ObligationGap],
) -> str:
    """Generate a 2-3 sentence executive summary for leadership audience."""
    risk = _risk_label(classification.risk_level)
    role = _role_label(classification.role)

    parts: list[str] = []

    # Sentence 1: Risk classification
    if classification.annex_iii_match:
        cat = classification.annex_iii_match.category_name
        cat_id = classification.annex_iii_match.subcategory_id
        parts.append(
            f"This AI system is classified as {risk} under the EU AI Act "
            f"(Annex III, Category {cat_id}: {cat})."
        )
    else:
        parts.append(
            f"This AI system is classified as {risk} under the EU AI Act."
        )

    # Sentence 2: Role and Article 25
    if classification.article_25_trigger:
        trigger = classification.article_25_trigger
        parts.append(
            f"Your organization functions as a {role} due to "
            f"Article 25(1)({chr(96 + trigger.scenario_number)}) role shift "
            f"({trigger.short_name.lower()})."
        )
    else:
        parts.append(f"Your organization functions as a {role}.")

    # Sentence 3: Obligation gap summary
    if gaps:
        unmet = sum(1 for g in gaps if g.status == "not_met")
        total = len(gaps)
        parts.append(f"{unmet} of {total} applicable obligations are not met.")
    elif classification.risk_level == RiskLevel.UNACCEPTABLE:
        parts.append(
            "This system is banned under Article 5 and cannot be deployed in the EU."
        )
    else:
        parts.append("No specific compliance obligations apply at this risk level.")

    return " ".join(parts)


def _build_next_steps(
    classification: ClassificationResult,
    gaps: list[ObligationGap],
) -> list[str]:
    """Generate prioritized next steps ordered critical -> high -> medium -> low."""
    if classification.risk_level == RiskLevel.UNACCEPTABLE:
        return [
            "[CRITICAL] Immediately cease deployment of this AI system in the EU.",
            "[CRITICAL] Engage legal counsel to assess exposure under Article 99 penalties.",
            "[HIGH] Evaluate whether the system can be redesigned to fall outside Article 5 prohibitions.",
        ]

    if not gaps:
        if classification.risk_level == RiskLevel.LIMITED_RISK:
            steps = []
            for t_ob in classification.transparency_obligations:
                steps.append(f"[MEDIUM] Implement transparency obligation: {t_ob}")
            if not steps:
                steps.append(
                    "[LOW] Review Article 50 transparency requirements to confirm no obligations apply."
                )
            return steps
        return ["[LOW] No immediate compliance actions required at minimal risk level."]

    priority_label = {
        Priority.CRITICAL: "CRITICAL",
        Priority.HIGH: "HIGH",
        Priority.MEDIUM: "MEDIUM",
        Priority.LOW: "LOW",
    }

    steps: list[str] = []

    # Add Article 25 warning as first step if triggered
    if classification.article_25_trigger:
        trigger = classification.article_25_trigger
        steps.append(
            f"[CRITICAL] Address Article 25 role shift: your organization has become "
            f"a provider due to {trigger.short_name.lower()}. "
            f"Full provider obligations now apply."
        )

    # Add GPAI terms warning if violated
    if classification.gpai_terms_check and classification.gpai_terms_check.terms_violated:
        provider = classification.gpai_terms_check.provider_name
        steps.append(
            f"[CRITICAL] Review and address {provider} terms of use violations. "
            f"Current usage may breach the provider's acceptable use policy."
        )

    # Add obligation-specific steps
    for gap in gaps:
        if gap.obligation_id == "BANNED":
            continue
        label = priority_label.get(gap.priority, "MEDIUM")
        if gap.action_items:
            steps.append(f"[{label}] {gap.name} ({gap.article}): {gap.action_items[0]}")
        else:
            steps.append(f"[{label}] Address {gap.name} requirement ({gap.article}).")

    return steps


def generate_report(
    description: str,
    classification: ClassificationResult,
) -> ComplianceReport:
    """Generate a full compliance report from a system description and classification.

    Args:
        description: The plain-English description of the AI system.
        classification: The ClassificationResult from the classifier agent.

    Returns:
        A populated ComplianceReport dataclass.
    """
    gaps = score_obligations(classification)
    risk_score = calculate_risk_score(classification, gaps)
    executive_summary = _build_executive_summary(classification, gaps)
    next_steps = _build_next_steps(classification, gaps)

    total = len(gaps)
    met = sum(1 for g in gaps if g.status == "met")

    return ComplianceReport(
        system_description=description,
        classification=classification,
        obligation_gaps=gaps,
        total_obligations=total,
        obligations_met=met,
        risk_score=risk_score,
        executive_summary=executive_summary,
        next_steps=next_steps,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def render_html(report: ComplianceReport) -> str:
    """Render the compliance report as HTML using the Jinja2 template.

    Args:
        report: A populated ComplianceReport dataclass.

    Returns:
        Rendered HTML string.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("compliance_report.html")

    # Prepare template context
    risk_colors = {
        RiskLevel.UNACCEPTABLE: "#dc3545",
        RiskLevel.HIGH_RISK: "#dc3545",
        RiskLevel.LIMITED_RISK: "#ffc107",
        RiskLevel.MINIMAL_RISK: "#28a745",
    }

    priority_colors = {
        Priority.CRITICAL: "#dc3545",
        Priority.HIGH: "#fd7e14",
        Priority.MEDIUM: "#ffc107",
        Priority.LOW: "#28a745",
    }

    context = {
        "report": report,
        "risk_label": _risk_label(report.classification.risk_level),
        "role_label": _role_label(report.classification.role),
        "risk_color": risk_colors.get(
            report.classification.risk_level, "#6c757d"
        ),
        "priority_colors": priority_colors,
    }

    return template.render(**context)
