# EU AI Act Compliance Scanner

A rule-based compliance assessment tool that classifies AI systems under the EU AI Act's risk taxonomy, detects deployer-to-provider role shifts under Article 25, and generates gap analysis reports with prioritized next steps.

**Built in 15 minutes by three AI agents** using [Claude Code Agent Teams](https://code.claude.com/docs/en/agent-teams).

> All scenarios use synthetic/simulated data. No real company names or proprietary information.

---

## What It Does

Describe how your organization uses AI in plain English. The scanner:

1. **Classifies risk level** — Maps your use case against all 8 Annex III high-risk categories
2. **Determines your role** — Provider, deployer, or provider via Article 25 role shift
3. **Detects Article 25 triggers** — All three scenarios (rebranding, substantial modification, modified intended purpose)
4. **Checks GPAI provider terms** — Flags potential acceptable use policy violations (OpenAI, Anthropic, Google, Microsoft, Meta)
5. **Generates a compliance report** — Obligation gaps, priority scoring, and actionable next steps

## Quick Start

```bash
git clone https://github.com/pmcavallo/eu-ai-act-scanner.git
cd eu-ai-act-scanner
pip install -r requirements.txt
py app.py
```

Open `http://localhost:7860` in your browser. No API key needed. Runs entirely offline.

## Example

**Input:**
> "Our HR team uses GPT-4 via API to screen resumes and rank candidates for engineering roles. We built custom prompts and connected it to our ATS."

**Output:**
- **Risk Level:** HIGH-RISK (Annex III, Category 4a — Employment)
- **Role:** PROVIDER (Article 25(1)(c) — Modified Intended Purpose)
- **GPAI Terms:** OpenAI — Violated
- **Obligations:** 0 of 10 met
- **Risk Score:** 100/100

The full report includes a compliance gap table with specific action items per obligation and prioritized next steps.

## How It Works

The scanner is 100% rule-based. No LLM calls, no API tokens, fully deterministic.

- **GPAI detection:** Regex patterns match product names (GPT-4, Claude, Gemini, Copilot, Llama, etc.)
- **Annex III matching:** Keyword scoring against all 8 high-risk categories with domain-specific boosting
- **Article 25 triggers:** Keyword detection for rebranding (Scenario A), substantial modification (Scenario B), and modified intended purpose (Scenario C)
- **Role determination:** Deployer/provider cue matching with Article 25 override
- **Fraud exclusion:** Correctly handles the Annex III Category 5a fraud detection carve-out
- **Report generation:** Jinja2 templates produce professional HTML reports with color-coded risk banners

## Project Structure

```
eu-ai-act-scanner/
├── app.py                          # Gradio UI
├── CLAUDE.md                       # Agent team coordination file
├── data/
│   ├── regulations/
│   │   ├── annex_iii.json          # 8 high-risk categories with keywords
│   │   ├── article_25.json         # 3 deployer-to-provider scenarios
│   │   └── obligations.json        # Provider + deployer obligations
│   └── provider_terms/
│       └── gpai_providers.json     # Terms for OpenAI, Anthropic, Google, Microsoft, Meta
├── src/
│   ├── classifier/
│   │   ├── classifier.py           # Classification pipeline orchestration
│   │   └── rules.py                # Rule-based matching engine (565 LOC)
│   ├── reports/
│   │   ├── report_generator.py     # Report assembly + executive summary
│   │   ├── priority_scorer.py      # Obligation scoring + risk calculation
│   │   └── templates/
│   │       └── compliance_report.html  # Jinja2 HTML report template
│   └── shared/
│       └── types.py                # Shared dataclasses
├── tests/
│   ├── test_scenarios.json         # 8 test cases with expected outputs
│   ├── test_classifier.py          # 51 classifier tests
│   ├── test_reports.py             # 12 report tests
│   └── test_integration.py         # 46 integration tests
└── requirements.txt
```

## Test Results

```
109 collected, 103 passed, 6 skipped, 0 failed
```

The 6 skipped tests are non-high-risk scenarios correctly bypassed by report obligation tests (no obligations apply to minimal/limited risk systems).

## Tech Stack

- **Python 3.10+** — Core language
- **Gradio** — Web UI
- **Jinja2** — HTML report templates
- **pytest** — Test framework
- **No LLM dependencies** — Fully offline, deterministic output

## Built With Agent Teams

This project was built using Claude Code's experimental Agent Teams feature. Three specialized agents worked in parallel:

| Agent | Role | Output |
|-------|------|--------|
| Regulatory Researcher | Verified and enriched regulatory JSON data | 4 enriched data files |
| Classifier Agent | Built the rule-based classification engine | `rules.py` (565 LOC), `classifier.py` (181 LOC) |
| Report Writer | Built the report generator and HTML templates | Report pipeline + Jinja2 template |

The Lead (Claude Code) coordinated task dependencies, ran smoke tests after each agent delivered, then built the Gradio UI and full test suite. Total build time: ~15 minutes.

See the [project page](https://pmcavallo.github.io/eu-ai-act-scanner/) for a detailed writeup of the Agent Teams build process.

## Regulatory Coverage

| Framework | Coverage |
|-----------|----------|
| EU AI Act Annex III | All 8 high-risk categories with subcategories |
| EU AI Act Article 25 | All 3 deployer-to-provider scenarios |
| EU AI Act Articles 8-15 | Full provider obligation mapping |
| EU AI Act Article 26 | Full deployer obligation mapping |
| EU AI Act Article 50 | Transparency obligations |
| GPAI Provider Terms | OpenAI, Anthropic, Google, Microsoft, Meta |

## Limitations

- Rule-based classification depends on keyword density. Very vague descriptions may not match.
- Provider terms data is a curated summary, not a legal opinion. Check original ToS for current versions.
- The scanner assumes no existing compliance measures are in place (all obligations default to "not met").
- This is a portfolio project for educational purposes, not legal advice.

## License

MIT

## Author

**Paulo Cavallo, PhD** — Senior AI Orchestrator | AI Governance in Regulated Industries

- [LinkedIn](https://www.linkedin.com/in/paulocavallo/)
- [AI Under Audit Newsletter](https://pmcavallo.github.io)
- [Portfolio](https://pmcavallo.github.io)
