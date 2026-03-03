"""Tests for the classification engine against all test scenarios."""
import json
from pathlib import Path

import pytest

from src.classifier import classify
from src.shared.types import RiskLevel, Role

SCENARIOS_FILE = Path(__file__).parent / "test_scenarios.json"


def _load_scenarios() -> list[dict]:
    with open(SCENARIOS_FILE, encoding="utf-8") as f:
        return json.load(f)["test_scenarios"]


SCENARIOS = _load_scenarios()


@pytest.fixture(params=SCENARIOS, ids=[s["id"] for s in SCENARIOS])
def scenario(request: pytest.FixtureRequest) -> dict:
    return request.param


class TestClassifier:
    """Test the classifier against all 8 test scenarios."""

    def test_risk_level(self, scenario: dict) -> None:
        result = classify(scenario["description"])
        expected = scenario["expected_risk_level"]
        if expected == "not_high_risk":
            assert result.risk_level != RiskLevel.HIGH_RISK, (
                f"{scenario['id']}: expected not high_risk, got {result.risk_level.value}"
            )
        else:
            assert result.risk_level.value == expected, (
                f"{scenario['id']}: expected {expected}, got {result.risk_level.value}"
            )

    def test_annex_iii_match(self, scenario: dict) -> None:
        result = classify(scenario["description"])
        expected_cat = scenario["expected_annex_iii_category"]
        if expected_cat is None:
            assert result.annex_iii_match is None, (
                f"{scenario['id']}: expected no Annex III match, "
                f"got {result.annex_iii_match.subcategory_id}"
            )
        else:
            assert result.annex_iii_match is not None, (
                f"{scenario['id']}: expected Annex III {expected_cat}, got None"
            )
            assert result.annex_iii_match.subcategory_id == expected_cat, (
                f"{scenario['id']}: expected {expected_cat}, "
                f"got {result.annex_iii_match.subcategory_id}"
            )

    def test_role(self, scenario: dict) -> None:
        result = classify(scenario["description"])
        assert result.role.value == scenario["expected_role"], (
            f"{scenario['id']}: expected role {scenario['expected_role']}, "
            f"got {result.role.value}"
        )

    def test_article_25(self, scenario: dict) -> None:
        result = classify(scenario["description"])
        expected = scenario["expected_article_25_scenario"]
        if expected is None:
            assert result.article_25_trigger is None, (
                f"{scenario['id']}: expected no Article 25 trigger, "
                f"got {result.article_25_trigger.scenario_id}"
            )
        else:
            assert result.article_25_trigger is not None, (
                f"{scenario['id']}: expected Article 25 {expected}, got None"
            )
            assert result.article_25_trigger.scenario_id == expected, (
                f"{scenario['id']}: expected {expected}, "
                f"got {result.article_25_trigger.scenario_id}"
            )

    def test_gpai_terms_violation(self, scenario: dict) -> None:
        result = classify(scenario["description"])
        expected_violation = scenario["expected_terms_violation"]
        actual_violation = (
            result.gpai_terms_check.terms_violated
            if result.gpai_terms_check
            else False
        )
        assert actual_violation == expected_violation, (
            f"{scenario['id']}: expected terms_violated={expected_violation}, "
            f"got {actual_violation}"
        )

    def test_gpai_provider_detection(self, scenario: dict) -> None:
        result = classify(scenario["description"])
        expected_provider = scenario.get("expected_gpai_provider")
        if expected_provider is None:
            assert result.gpai_terms_check is None, (
                f"{scenario['id']}: expected no GPAI provider, "
                f"got {result.gpai_terms_check.provider_name}"
            )
        else:
            assert result.gpai_terms_check is not None, (
                f"{scenario['id']}: expected GPAI provider {expected_provider}, got None"
            )
            assert result.gpai_terms_check.provider_name.lower() == expected_provider.lower() or \
                expected_provider.lower() in result.gpai_terms_check.provider_name.lower(), (
                f"{scenario['id']}: expected provider {expected_provider}, "
                f"got {result.gpai_terms_check.provider_name}"
            )


class TestClassifierConfidence:
    """Test that confidence scores are reasonable."""

    def test_high_risk_confidence(self) -> None:
        result = classify(
            "Our HR team uses GPT-4 via API to screen resumes and rank candidates."
        )
        assert result.confidence > 0.5

    def test_limited_risk_confidence(self) -> None:
        result = classify(
            "Our marketing team uses Claude to draft email campaigns."
        )
        assert result.confidence > 0.5

    def test_classification_has_reasoning(self) -> None:
        result = classify(
            "We purchased a credit scoring AI from a vendor. No modifications."
        )
        assert len(result.reasoning) > 0
