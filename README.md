# Credit Verdict
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) [![Python 3.10+](https://img.shields.io/badge/python-3.10+-green.svg)](https://python.org) [![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-1C3C3C.svg)](https://langchain-ai.github.io/langgraph/)

> **You've probably seen a Home Credit Default Risk notebook before — here's what's different about this one:** a **calibrated** model (not just AUC-optimized) and a **RAG-grounded Adverse Action Notice generator** that cites real ECOA/Regulation B and OSFI E-23 text, with its own citations audited for faithfulness.
>
> ![Demo](assets/demo.gif)
> *[Insert 10–15s screen capture: an application scored → decision returned → cited Adverse Action explanation generated]*

## Table of Contents
- [Overview](#overview)
- [Status](#status)
- [Results](#results)
- [LangGraph Multi-Agent Flow](#langgraph-multi-agent-flow)
- [Backend Technologies](#backend-technologies)
- [Frontend Technologies](#frontend-technologies)
- [Development Tools](#development-tools)
- [Business Impact](#business-impact)
- [Key Features](#key-features)
- [System Components](#system-components)
- [Responsible AI Components](#responsible-ai-components)
- [Why the Chatbot and Dashboard Are Separate](#why-the-chatbot-and-dashboard-are-separate)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Dataset](#dataset)
- [Model Card](#model-card)
- [Future Development](#future-development)
- [Assumptions](#assumptions)
- [Reference](#reference)
- [License](#license)

## Overview

Credit Verdict is a loan-default risk system built on Kaggle's [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) dataset, designed around a problem most credit-risk portfolio projects ignore: **a model that scores applicants isn't enough — a real lender is legally required to give a specific, defensible reason when it denies credit** (ECOA/Regulation B in the U.S., OSFI Guideline E-23 for federally regulated Canadian institutions).

Credit Verdict addresses both halves of that problem in one system:

- **Accuracy & calibration** — a LightGBM classifier, benchmarked honestly against a logistic regression baseline, with its probability outputs explicitly calibrated (not just AUC-optimized) so a "12% probability of default" actually corresponds to real-world default rates.
- **Explainability & compliance** — SHAP identifies the specific factors driving each decision; a multi-agent RAG pipeline (LangGraph + Azure OpenAI) turns those factors into a plain-English, regulation-cited Adverse Action Notice, with its own citations audited for faithfulness via RAGAS before they're ever shown to a user.

## Status

- [x] Phase 0 — Project scaffolding
- [x] Phase 1 — ML fundamentals
- [x] Phase 2 — Credit risk & regulatory domain research
- [x] Phase 3 — Data & exploratory data analysis
- [x] Phase 4 — Feature engineering
- [x] Phase 5 — Model training & experimentation (MLflow)
- [x] Phase 6 — Calibration
- [x] Phase 7 — SHAP explainability
- [ ] Phase 8 — Multi-agent RAG / Adverse Action Notice generation
- [ ] Phase 9 — Full-stack build (FastAPI, chatbot, Streamlit dev view)
- [ ] Phase 10 — Business case, dashboard, monitoring, README polish

## Results

| Metric | Value |
|---|---|
| **AUC (ROC)** | [X] |
| **Brier score, pre-calibration** | [X] |
| **Brier score, post-calibration** | [X] |
| **Features** | [X] engineered from `application_train`, `bureau`, `previous_application`, `credit_card_balance` |
| **Dataset** | 307,511 loan applications (8.1% default rate) |
| **RAGAS faithfulness (Adverse Action explanations)** | [X] |

AUC alone isn't the headline metric here — see [`notebooks/model_evaluation.ipynb`](notebooks/model_evaluation.ipynb) for the calibration curve and Brier score comparison that this project treats as equally important.

## LangGraph Multi-Agent Flow

This section outlines the agent routing behind the Adverse Action chatbot.

### Agent Definitions

- **`ASSISTANT_AGENT`**: Handles greetings, general queries, and fallback responses.
- **`EXPLANATION_AGENT`**: Pulls the precomputed SHAP factors for the applicant in context. Deterministic — no LLM call, cannot invent a factor that wasn't actually in the model's output.
- **`COMPLIANCE_RAG_AGENT`**: Retrieves grounding text from the ECOA/OSFI-E23 compliance corpus (pgvector) and runs the RAGAS faithfulness check before any explanation is finalized.
- **`REPORTING_AGENT`**: Synthesizes `EXPLANATION_AGENT`'s factors and `COMPLIANCE_RAG_AGENT`'s grounded text into the final plain-English response.

### Selection Strategy & Agent Flow

Routing scales with query complexity — not every question invokes every agent.

#### 1. General Queries
**Example**: `"Hello, can you help me?"`
```
User Query → ASSISTANT_AGENT → End Conversation
```

#### 2. "Why was I declined / approved?"
**Example**: `"Why was my application declined?"`
```
User Query → EXPLANATION_AGENT → REPORTING_AGENT → End Conversation
```
SHAP factors are precomputed at scoring time — no retrieval call is needed for the core "why."

#### 3. "What can I do to improve my chances? / Can you cite the rule?"
**Example**: `"What could I do differently if I reapply?"`
```
User Query → EXPLANATION_AGENT → COMPLIANCE_RAG_AGENT → REPORTING_AGENT → End Conversation
```
Retrieval is only invoked when grounding is actually needed for the answer — SHAP determines the facts deterministically; the LLM only ever phrases and grounds them.

### Technical Implementation Details

- **Scoped demo input**: the chatbot answers questions about pre-validated Phase 8 test-set applicants, not free-form live input, to keep the first version's failure surface bounded and testable.
- **Faithfulness-gated responses**: `COMPLIANCE_RAG_AGENT` output is checked against a RAGAS faithfulness threshold before `REPORTING_AGENT` is allowed to use it.
- **Model backend**: Azure OpenAI for production/demo runs; Groq (free tier) during local development and iteration.

## Backend Technologies

- **LangGraph** – Multi-agent orchestration; state-machine-based routing between the four agents above.
- **Azure OpenAI Service** – LLM backend for `ASSISTANT_AGENT`, `COMPLIANCE_RAG_AGENT`, and `REPORTING_AGENT`.
- **LightGBM / scikit-learn** – Classifier (LightGBM) and calibration/baseline tooling (`CalibratedClassifierCV`, logistic regression).
- **SHAP** – Global and per-prediction local explainability (`TreeExplainer`).
- **PostgreSQL + pgvector** – Feature store, model logs, and the compliance-corpus vector index — one database for the whole system, not a separate vector DB.
- **RAGAS** – Faithfulness evaluation for generated Adverse Action explanations.
- **MLflow** – Experiment tracking across model variants.
- **FastAPI** – REST API (`/predict`, `/explain/{id}`, `/adverse-action/{id}`, `/chat`, `/health`).
- **Streamlit** – Developer-view debug panel: which agent fired, raw SHAP values, retrieved corpus chunks, faithfulness score per turn.
- **python-docx** – Adverse Action Notice export to Word, unrestricted (no page-limit license constraints).
- **Evidently AI** – Drift monitoring on model confidence/feature distributions over time.

## Frontend Technologies

- **React** – Applicant-facing chat interface and dashboard views.
- **Tailwind CSS** – Styling.
- **Power BI** – Business/risk-team dashboard (published via "Publish to Web," kept separate from the applicant chatbot — see below).

## Development Tools

- **VS Code** – Primary editor.
- **Claude Code** – Phase-by-phase implementation, with resource checkpoints and report updates at each stage (see `roadmap.md`).
- **Postman** – API testing.
- **Git / GitHub Actions** – Version control, CI (tests + lint on push).

## Business Impact

Credit Verdict addresses a real, named compliance problem, not a hypothetical one:

- **Legal defensibility** — CFPB Circular 2023-03 explicitly states creditors cannot rely on generic denial reasons that don't reflect the model's actual decision factors; this project's SHAP-to-citation pipeline is a direct answer to that requirement.
- **Regulatory alignment ahead of a real deadline** — OSFI Guideline E-23's AI-specific explainability and independent-validation requirements take effect May 1, 2027; this project's design already anticipates that bar.
- **Trustworthy risk decisions** — calibration means the model's probability outputs are usable for actual business decisions (portfolio expected-loss estimates, cutoff tradeoffs), not just ranking.
- **Auditability** — every generated explanation is faithfulness-checked and traceable back to the specific regulatory/policy text it cites.

## Key Features

### Calibrated Risk Scoring
- LightGBM benchmarked honestly against a logistic regression baseline
- Explicit calibration (Platt scaling or isotonic regression, chosen based on the observed miscalibration pattern) — not just an AUC number

### Explainable, Citation-Audited Adverse Action Notices
- Per-applicant SHAP breakdown (global and local)
- Plain-English denial explanations grounded in real ECOA §1002.9 and OSFI E-23 text
- RAGAS faithfulness check on every generated explanation before it's shown

### Conversational Decision Explanation
- Multi-agent chatbot answering "why" and reasonable "what now" follow-ups
- Scoped to validated test applicants for the first version — see [Technical Implementation Details](#technical-implementation-details)

### Business Dashboard (Separate from the Chatbot)
- Risk segmentation, cutoff tradeoff analysis, portfolio expected loss
- Deliberately kept as a separate surface from the applicant-facing chatbot — see below

### Transparent AI Reasoning
- Every pipeline step (SHAP computation → retrieval → generation → faithfulness check) is logged
- Streamlit developer view for inspecting agent decisions and retrieved evidence in real time

## System Components

The system follows a modular, multi-agent design pattern:

### Agent Layer
- **Explanation Agent** — deterministic SHAP factor retrieval
- **Compliance RAG Agent** — grounded retrieval + faithfulness check
- **Reporting Agent** — final synthesis
- **Assistant Agent** — general conversation handling

### Manager Layer
- **Chatbot Orchestrator** — LangGraph state machine; selection and termination logic

### Plugin/Module Layer
- **SHAP module** — wraps the Phase 7 explainer
- **Corpus retrieval module** — pgvector lookup against the compliance corpus
- **Faithfulness check module** — RAGAS scoring
- **Notice generation module** — plain-English + Word export via python-docx
- **Logging module** — agent reasoning trace

### API & Interface Layer
- **FastAPI application** — REST endpoints for scoring, explanation, chat
- **Streamlit interface** — developer debug view
- **React application** — applicant-facing chat + business dashboard link

## Responsible AI Components

Credit Verdict incorporates several responsible-AI practices to ensure transparency, accountability, and regulatory alignment:

- **Transparent Reasoning Process** — every pipeline step is logged with timestamps, from SHAP computation through faithfulness verification, so a decision can be traced end to end.
- **Structured Agent Reasoning** — `EXPLANATION_AGENT`'s deterministic facts are kept separate from `COMPLIANCE_RAG_AGENT`'s grounded phrasing; intermediate steps are logged and auditable, not collapsed into a single opaque LLM call.
- **Decision Criteria Transparency** — calibration methodology, cutoff thresholds, and their business tradeoffs are documented in `CreditRisk_Project_Report.md`, not just implied by the model.
- **Citation Faithfulness Auditing** — every generated explanation is checked against its retrieved source before being shown, the same role RAGAS's faithfulness metric plays that Bing-search citation tracking plays in comparable reference architectures.
- **Scoped Deployment** — the chatbot only answers on pre-validated test applicants in its first version, a deliberate constraint to keep the system's failure surface known and testable rather than open-ended.

## Why the Chatbot and Dashboard Are Separate

This is a deliberate design decision, not an oversight: the chatbot and the Power BI dashboard serve **different audiences with different appropriate data access**.

- The **dashboard** is for the internal risk/business team — portfolio-wide analytics, aggregate risk segments.
- The **chatbot** is for an individual applicant — and should only ever see *their own* decision, never portfolio-wide data.

Merging these into one interface would mean building an access-control problem that doesn't need to exist. Keeping them separate — cross-linked from one landing page — respects that boundary by construction rather than by policy.

## Quick Start

### Prerequisites
- Python 3.10+
- Docker
- PostgreSQL (or use the provided `docker-compose.yml`)
- Azure OpenAI resource (or a Groq API key for local development)
- Kaggle account + API credentials (`kaggle.json`) for the dataset

### Installation
```bash
git clone https://github.com/darang1412/credit-verdict.git
cd credit-verdict

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Environment Setup
```bash
cp .env.example .env
# Edit .env with your DB connection string, Azure OpenAI / Groq keys, Kaggle credentials
```

### Running the Application
```bash
# Backend + Postgres/pgvector
docker compose up -d

# API
PYTHONPATH=backend uvicorn backend.main:app --reload --port 8000

# Developer view (separate terminal)
streamlit run backend/dev_view.py
```

## Project Structure

```
.
├── data/                     # raw/interim/processed/external (gitignored; regenerate via ml/pipeline)
├── notebooks/                # EDA, modeling, and evaluation notebooks, numbered by run order
├── ml/
│   ├── pipeline/             # Data ingestion, cleaning, feature engineering
│   ├── modeling/             # Training, calibration
│   └── explain/              # SHAP analysis
├── models/                   # Trained/serialized model artifacts (gitignored)
├── reports/                  # Generated figures + SHAP output (reports/figures/, reports/top_shap_factors.json)
├── rag/                      # LangGraph agents, corpus builder, Adverse Action generator
├── backend/                  # FastAPI app, Streamlit dev view
├── frontend/                 # React chat interface + dashboard link
├── business/                 # Power BI dashboard assets, business case materials
├── assets/                   # Demo GIF / README screenshots
├── docs/
│   └── adr/                  # Architecture Decision Records (e.g., scaling strategy)
├── tests/
├── roadmap.md                      # Phase-by-phase build roadmap
├── CreditRisk_Project_Report.md    # Full 9-section project report
├── MODEL_CARD.md                   # Model card (intended use, limitations, performance)
├── docker-compose.yml
└── requirements.txt
```

## Testing
```bash
pytest tests/ -v
```
CI runs tests and lint on every push (`.github/workflows/ci.yml`).

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (Kaggle). Real, multi-table application data — see `notebooks/` for the full EDA.

## Model Card

See [`MODEL_CARD.md`](MODEL_CARD.md) for intended use, training data, performance across slices, and known limitations.

## Future Development

1. Formal fairness/disparate-impact audit across protected classes — required before any real deployment, given ECOA's anti-discrimination provisions
2. Live-data connector instead of the static Kaggle dataset
3. Human-in-the-loop review UI for generated Adverse Action notices before they're sent
4. Tableau mirror of the Power BI dashboard (for roles that specifically screen on Tableau)
5. Expanded compliance corpus (additional CFPB Circulars, provincial Canadian consumer-protection sources)

## Assumptions

- The Home Credit Default Risk dataset (2018) reflects lending patterns from that period; a real deployment would require current, proprietary data.
- The Adverse Action generator is a working prototype of the pattern, not a compliance-certified system — real deployment requires legal review of the policy corpus and human sign-off on generated notices.
- Processing time target: real-time scoring; the current architecture is a single FastAPI service, documented as a starting point for scale in `docs/adr/001-scaling-strategy.md`, not a production SLA claim.

## Reference

- Microsoft AI Agents Hackathon — [RiskWise: Procurement Risk Analysis System](https://github.com/microsoft/AI_Agents_Hackathon/issues/526) (multi-agent routing pattern reference)
- CFPB Circulars 2022-03 and 2023-03 — AI/ML credit decisions and specific-reasons requirements
- OSFI Guideline E-23 (Model Risk Management, 2027)
- [kozodoi/Kaggle_Home_Credit](https://github.com/kozodoi/Kaggle_Home_Credit), [jamesdellinger/kaggle_home_credit_default_risk_competition](https://github.com/jamesdellinger/kaggle_home_credit_default_risk_competition), [Bougeant/Home_Credit_default_risk](https://github.com/Bougeant/Home_Credit_default_risk), [NoxMoon/home-credit-default-risk](https://github.com/NoxMoon/home-credit-default-risk) — feature engineering and imbalance-handling references
- Chip Huyen, *Designing Machine Learning Systems*

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
