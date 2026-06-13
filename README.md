# CV-Align-Agents

> Multi-agent resume screening pipeline built with **LangGraph** and **Gemini**, served via **FastAPI**.

CV-Align-Agents screens resumes against a job description using a pipeline of
specialized LLM agents — a **parser**, a **JD parser**, a **matcher**, a
deterministic **scorer**, and a **critic** — with a self-correction loop and a
full audit trail for explainability.

## Why a multi-agent design?

A single LLM call that "scores a resume" is a black box: you can't tell *why* a
candidate ranked where they did, and you can't improve one stage without
risking the others. CV-Align-Agents splits the job into focused stages:

| Stage | Type | Responsibility |
|-------|------|----------------|
| Parser | agent | Resume PDF text → structured fields (skills, projects, experience) |
| JD Parser | agent | Job description → structured requirements |
| Matcher | agent | Per-section sub-scores + quoted evidence |
| Scorer | function | Deterministic weighted score from sub-scores |
| Critic | agent | Gaps, suggestions, verdict, and a confidence check |

If the critic is not confident the score is well-supported, it loops back to the
matcher once for a re-evaluation (capped to avoid infinite loops).

## Status

🚧 Early development — built step by step. See `docs/architecture.md` for design.

## Tech stack

- Python 3.12
- FastAPI + Uvicorn
- LangGraph + LangChain (agent orchestration)
- Google Gemini (free tier) — provider-abstracted, Groq swappable
- pypdf (PDF text extraction)
- SQLite (run history / explainability)

## Getting started

```bash
# 1. Clone
git clone https://github.com/ShreyanshMehra/CV-Align-Agents.git
cd CV-Align-Agents

# 2. Create & activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
copy .env.example .env        # Windows  (use `cp` on macOS/Linux)
# then edit .env and add your free Gemini key from
# https://aistudio.google.com/apikey
```

Running the API and the full usage guide will be documented as those pieces land.

## Project layout

```
src/cv_align_agents/
├── state.py          # Shared Pydantic state passed between agents
├── llm/              # Provider-abstracted LLM client
├── agents/           # parser, jd_parser, matcher, critic
├── pipeline/         # deterministic scorer + LangGraph wiring
├── api/              # FastAPI app
└── storage/          # SQLite run persistence
```

## License

[MIT](LICENSE) © 2026 Shreyansh Dutt Mehra
