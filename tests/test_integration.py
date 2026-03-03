"""End-to-end integration tests: description -> classification -> report."""
import json
from pathlib import Path

import pytest

from src.classifier import classify
from src.reports import generate_report, render_html
from src.shared.types import ComplianceReport, RiskLevel, Role

SCENARIOS_FILE = Path(__file__).parent / "test_scenarios.json"


def _load_scenarios() -> list[dict]:
    with open(SCENARIOS_FILE, encoding="utf-8") as f:
        return json.load(f)["test_scenarios"]


SCENARIOS = _load_scenarios()


@pytest.fixture(params=SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def scenario(request: pytest.FixtureRequest) -> dict:
    return request.param


class TestEndToEnd:
    """Full pipeline: description -> classify -> generate_report -> render_html."""

    def test_pipeline_produces_report(self, scenario: dict) -> None:
        """Every scenario should produce a valid ComplianceReport."""
        classification = classify(scenario["description"])
        report = generate_report(scenario["description"], classification)

        assert isinstance(report, ComplianceReport)
        assert report.system_description == scenario["description"]
        assert report.generated_at != ""
        assert report.risk_score >= 0
        assert report.risk_score <= 100

    def test_pipeline_produces_html(self, scenario: dict) -> None:
        """Every scenario should produce renderable HTML."""
        classification = classify(scenario["description"])
        report = generate_report(scenario["description"], classification)
        html = render_html(report)

        assert "<!DOCTYPE html>" in html
        assert "EU AI Act Compliance Report" in html
        assert len(html) > 500

    def test_high_risk_has_obligations(self, scenario: dict) -> None:
        """High-risk scenarios should have obligation gaps."""
        classification = classify(scenario["description"])
        if classification.risk_level != RiskLevel.HIGH_RISK:
            pytest.skip("Not high-risk")

        report = generate_report(scenario["description"], classification)
        assert report.total_obligations > 0
        assert len(report.obligation_gaps) > 0

    def test_provider_has_more_obligations_than_deployer(self) -> None:
        """Provider role should trigger more obligations than deployer."""
        # TS01 = provider, TS02 = deployer, both high-risk
        ts01 = next(s for s in SCENARIOS if s["id"] == "TS01")
        ts02 = next(s for s in SCENARIOS if s["id"] == "TS02")

        c1 = classify(ts01["description"])
        r1 = generate_report(ts01["description"], c1)

        c2 = classify(ts02["description"])
        r2 = generate_report(ts02["description"], c2)

        assert r1.total_obligations > r2.total_obligations

    def test_article_25_increases_risk_score(self) -> None:
        """Article 25 trigger should result in higher risk score than no trigger."""
        # TS01 has Article 25(c), TS02 has no Article 25
        ts01 = next(s for s in SCENARIOS if s["id"] == "TS01")
        ts02 = next(s for s in SCENARIOS if s["id"] == "TS02")

        c1 = classify(ts01["description"])
        r1 = generate_report(ts01["description"], c1)

        c2 = classify(ts02["description"])
        r2 = generate_report(ts02["description"], c2)

        assert r1.risk_score > r2.risk_score

    def test_limited_risk_no_obligation_gaps(self) -> None:
        """Limited risk scenarios should have no obligation gaps."""
        ts05 = next(s for s in SCENARIOS if s["id"] == "TS05")
        classification = classify(ts05["description"])
        report = generate_report(ts05["description"], classification)

        assert report.total_obligations == 0
        assert len(report.obligation_gaps) == 0

    def test_executive_summary_is_concise(self, scenario: dict) -> None:
        """Executive summary should be 1-3 sentences (under 500 chars)."""
        classification = classify(scenario["description"])
        report = generate_report(scenario["description"], classification)

        assert len(report.executive_summary) > 0
        assert len(report.executive_summary) < 500

    def test_next_steps_not_empty_for_high_risk(self, scenario: dict) -> None:
        """High-risk scenarios should always have next steps."""
        classification = classify(scenario["description"])
        if classification.risk_level != RiskLevel.HIGH_RISK:
            pytest.skip("Not high-risk")

        report = generate_report(scenario["description"], classification)
        assert len(report.next_steps) > 0


class TestGradioScanFunction:
    """Test the Gradio scan function directly."""

    def test_scan_returns_three_outputs(self) -> None:
        from app import scan

        summary, details, html = scan(
            "Our HR team uses GPT-4 to screen resumes."
        )
        assert "HIGH RISK" in summary
        assert "PROVIDER" in summary
        assert len(details) > 0
        assert "<!DOCTYPE html>" in html

    def test_scan_empty_input(self) -> None:
        from app import scan

        summary, details, html = scan("")
        assert "Please enter" in summary

    def test_scan_whitespace_input(self) -> None:
        from app import scan

        summary, details, html = scan("   ")
        assert "Please enter" in summary
