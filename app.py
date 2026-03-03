"""
EU AI Act Compliance Scanner — Gradio UI.

Takes a plain-English description of how an organization uses AI, classifies
the system under the EU AI Act's risk taxonomy, and generates a compliance
gap report with prioritized next steps.
"""
import json
from pathlib import Path

import gradio as gr

from src.classifier import classify
from src.reports import generate_report, render_html

SCENARIOS_FILE = Path(__file__).parent / "tests" / "test_scenarios.json"


def _load_example_scenarios() -> list[list[str]]:
    """Load example scenarios from the test file for the UI examples panel."""
    with open(SCENARIOS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return [[s["description"]] for s in data["test_scenarios"]]


def scan(description: str) -> tuple[str, str, str]:
    """Run the full compliance scan pipeline.

    Args:
        description: Plain-English description of the AI system.

    Returns:
        Tuple of (summary_text, details_text, html_report).
    """
    if not description or not description.strip():
        return "Please enter a description of your AI system.", "", ""

    classification = classify(description.strip())
    report = generate_report(description.strip(), classification)
    html = render_html(report)

    # Build summary text
    summary_lines = [
        f"Risk Level: {report.classification.risk_level.value.upper().replace('_', ' ')}",
        f"Role: {report.classification.role.value.upper()}",
    ]
    if report.classification.annex_iii_match:
        m = report.classification.annex_iii_match
        summary_lines.append(f"Annex III: Category {m.subcategory_id} — {m.category_name}")
    if report.classification.article_25_trigger:
        t = report.classification.article_25_trigger
        summary_lines.append(f"Article 25: {t.short_name} ({t.scenario_id})")
    if report.classification.gpai_terms_check:
        g = report.classification.gpai_terms_check
        violation = "YES" if g.terms_violated else "No"
        summary_lines.append(f"GPAI Provider: {g.provider_name} — Terms Violated: {violation}")
    summary_lines.append(f"Risk Score: {report.risk_score:.0f}/100")
    summary_lines.append(f"Obligations: {report.obligations_met}/{report.total_obligations} met")

    summary = "\n".join(summary_lines)

    # Build details text
    details_parts = [report.executive_summary, ""]
    if report.next_steps:
        details_parts.append("NEXT STEPS:")
        for i, step in enumerate(report.next_steps, 1):
            details_parts.append(f"  {i}. {step}")

    details = "\n".join(details_parts)

    return summary, details, html


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    examples = _load_example_scenarios()

    with gr.Blocks(
        title="EU AI Act Compliance Scanner",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            "# EU AI Act Compliance Scanner\n"
            "Describe how your organization uses AI and get an instant compliance assessment "
            "under the EU AI Act's risk taxonomy.\n\n"
            "*All scenarios use synthetic/simulated data. No real company names.*"
        )

        with gr.Row():
            with gr.Column(scale=1):
                description_input = gr.Textbox(
                    label="AI System Description",
                    placeholder="Describe how your organization uses AI...",
                    lines=6,
                )
                scan_btn = gr.Button("Scan", variant="primary", size="lg")
                gr.Examples(
                    examples=examples,
                    inputs=description_input,
                    label="Example Scenarios",
                )

            with gr.Column(scale=1):
                summary_output = gr.Textbox(label="Classification Summary", lines=8)
                details_output = gr.Textbox(label="Executive Summary & Next Steps", lines=10)

        gr.Markdown("---")
        gr.Markdown("### Full Compliance Report")
        html_output = gr.HTML(label="Compliance Report")

        scan_btn.click(
            fn=scan,
            inputs=description_input,
            outputs=[summary_output, details_output, html_output],
        )

        description_input.submit(
            fn=scan,
            inputs=description_input,
            outputs=[summary_output, details_output, html_output],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch()
