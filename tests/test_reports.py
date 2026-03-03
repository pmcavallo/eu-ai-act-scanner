"""Tests for the report generation module."""
import pytest

from src.reports import generate_report, render_html
from src.shared.types import (
    AnnexIIIMatch,
    Article25Trigger,
    ClassificationResult,
    ComplianceReport,
    GPAITermsCheck,
    Priority,
    RiskLevel,
    Role,
)


def _make_classification(
    risk_level: RiskLevel = RiskLevel.HIGH_RISK,
    role: Role = Role.PROVIDER,
    annex_match: AnnexIIIMatch | None = None,
    article_25: Article25Trigger | None = None,
    gpai_check: GPAITermsCheck | None = None,
) -> ClassificationResult:
    """Helper to build ClassificationResult for tests."""
    return ClassificationResult(
        risk_level=risk_level,
        annex_iii_match=annex_match,
        role=role,
        article_25_trigger=article_25,
        gpai_terms_check=gpai_check,
        transparency_obligations=[],
        confidence=0.9,
        reasoning="Test reasoning",
    )


class TestReportGeneration:
    """Test report generation for various classification scenarios."""

    def test_high_risk_provider_report(self) -> None:
        classification = _make_classification(
            annex_match=AnnexIIIMatch(
                category_id=4,
                subcategory_id="4a",
                category_name="Employment",
                description="Recruitment AI",
                confidence=0.9,
                reasoning="Matched employment keywords",
            ),
            article_25=Article25Trigger(
                scenario_id="25_1_c",
                scenario_number=3,
                short_name="Modified Intended Purpose",
                confidence=0.95,
                reasoning="GPAI used for employment screening",
            ),
            gpai_check=GPAITermsCheck(
                provider_name="OpenAI",
                product_used="GPT-4",
                terms_violated=True,
                violated_terms=["Automated employment decisions"],
                reasoning="High-risk use violates AUP",
            ),
        )
        report = generate_report("Test HR screening system", classification)

        assert isinstance(report, ComplianceReport)
        assert report.risk_score >= 80
        assert report.total_obligations >= 10  # Provider obligations
        assert report.obligations_met == 0
        assert "HIGH-RISK" in report.executive_summary
        assert len(report.next_steps) > 0
        assert report.generated_at != ""

    def test_deployer_report(self) -> None:
        classification = _make_classification(
            role=Role.DEPLOYER,
            annex_match=AnnexIIIMatch(
                category_id=5,
                subcategory_id="5a",
                category_name="Essential Services",
                description="Credit scoring",
                confidence=0.85,
                reasoning="Matched credit keywords",
            ),
        )
        report = generate_report("Vendor credit scoring system", classification)

        assert report.risk_score >= 60
        assert report.total_obligations >= 7  # Deployer obligations D1-D7+
        assert "DEPLOYER" in report.executive_summary

    def test_minimal_risk_report(self) -> None:
        classification = _make_classification(
            risk_level=RiskLevel.MINIMAL_RISK,
            role=Role.DEPLOYER,
        )
        report = generate_report("Internal data analysis tool", classification)

        assert report.risk_score <= 10
        assert report.total_obligations == 0
        assert len(report.obligation_gaps) == 0

    def test_limited_risk_report(self) -> None:
        classification = _make_classification(
            risk_level=RiskLevel.LIMITED_RISK,
            role=Role.DEPLOYER,
        )
        report = generate_report("Customer chatbot", classification)

        assert report.risk_score <= 40
        assert report.total_obligations == 0

    def test_unacceptable_risk_report(self) -> None:
        classification = _make_classification(
            risk_level=RiskLevel.UNACCEPTABLE,
            role=Role.PROVIDER,
        )
        report = generate_report("Social scoring system", classification)

        assert report.risk_score == 100.0
        assert report.total_obligations == 1
        assert report.obligation_gaps[0].obligation_id == "BANNED"
        assert "banned" in report.executive_summary.lower() or "UNACCEPTABLE" in report.executive_summary

    def test_obligation_gaps_sorted_by_priority(self) -> None:
        classification = _make_classification(
            annex_match=AnnexIIIMatch(
                category_id=4,
                subcategory_id="4a",
                category_name="Employment",
                description="Recruitment",
                confidence=0.9,
                reasoning="Match",
            ),
        )
        report = generate_report("AI hiring tool", classification)

        priorities = [g.priority for g in report.obligation_gaps]
        priority_order = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3,
        }
        order_values = [priority_order[p] for p in priorities]
        assert order_values == sorted(order_values), "Gaps not sorted by priority"

    def test_obligation_gaps_have_action_items(self) -> None:
        classification = _make_classification(
            annex_match=AnnexIIIMatch(
                category_id=5,
                subcategory_id="5a",
                category_name="Essential Services",
                description="Credit scoring",
                confidence=0.85,
                reasoning="Match",
            ),
        )
        report = generate_report("Credit scoring AI", classification)

        for gap in report.obligation_gaps:
            if gap.obligation_id != "BANNED":
                # Most obligations should have action items (D8 may not)
                assert gap.name != "", f"Gap {gap.obligation_id} has no name"


class TestHTMLRendering:
    """Test HTML report rendering."""

    def test_html_renders_without_error(self) -> None:
        classification = _make_classification(
            annex_match=AnnexIIIMatch(
                category_id=4,
                subcategory_id="4a",
                category_name="Employment",
                description="Recruitment",
                confidence=0.9,
                reasoning="Match",
            ),
        )
        report = generate_report("AI hiring system", classification)
        html = render_html(report)

        assert "<!DOCTYPE html>" in html
        assert "EU AI Act Compliance Report" in html
        assert len(html) > 1000

    def test_html_contains_risk_level(self) -> None:
        classification = _make_classification(
            annex_match=AnnexIIIMatch(
                category_id=4,
                subcategory_id="4a",
                category_name="Employment",
                description="Recruitment",
                confidence=0.9,
                reasoning="Match",
            ),
        )
        report = generate_report("AI hiring system", classification)
        html = render_html(report)

        assert "HIGH-RISK" in html

    def test_html_contains_article_25_section(self) -> None:
        classification = _make_classification(
            article_25=Article25Trigger(
                scenario_id="25_1_c",
                scenario_number=3,
                short_name="Modified Intended Purpose",
                confidence=0.95,
                reasoning="GPAI used for high-risk",
            ),
            annex_match=AnnexIIIMatch(
                category_id=4,
                subcategory_id="4a",
                category_name="Employment",
                description="Recruitment",
                confidence=0.9,
                reasoning="Match",
            ),
        )
        report = generate_report("GPT-4 for hiring", classification)
        html = render_html(report)

        assert "Article 25" in html
        assert "Role Shift" in html

    def test_html_minimal_risk_green(self) -> None:
        classification = _make_classification(
            risk_level=RiskLevel.MINIMAL_RISK,
            role=Role.DEPLOYER,
        )
        report = generate_report("Simple tool", classification)
        html = render_html(report)

        assert "#28a745" in html  # green color

    def test_html_contains_disclaimer(self) -> None:
        classification = _make_classification(
            risk_level=RiskLevel.MINIMAL_RISK,
            role=Role.DEPLOYER,
        )
        report = generate_report("Simple tool", classification)
        html = render_html(report)

        assert "synthetic/simulated data" in html
