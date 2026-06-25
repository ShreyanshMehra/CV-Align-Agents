# CV-Align-Agents

> Multi-agent resume screening pipeline built with **LangGraph** and **Gemini**, served via **FastAPI**.

CV-Align-Agents screens resumes against a job description using a pipeline of
specialized agents — a **parser**, a **JD parser**, a **matcher**, a
deterministic **scorer**, a deterministic **hygiene** checker, and a **critic** —
with a self-correction loop and a full audit trail for explainability.

It serves two personas from one engine:

- **Candidate mode** — "How well does my CV fit this job, and how do I improve it?"
- **Recruiter mode** — "Rank these resumes for this job, with reasons."

## Why a multi-agent design?

A single LLM call that "scores a resume" is a black box: you can't tell *why* a
candidate ranked where they did, and you can't improve one stage without
risking the others. CV-Align-Agents splits the job into focused stages, using a
plain function where the task is deterministic and an LLM agent only where the
task needs judgment:

| Stage | Type | Responsibility |
|-------|------|----------------|
| Parser | agent | Resume PDF text → structured fields (skills, projects, experience) |
| JD Parser | agent | Job description → structured requirements |
| Matcher | agent | Per-section sub-scores + quoted evidence, vs **this** JD |
| Scorer | function | Deterministic weighted score from sub-scores |
| Hygiene | function | Objective resume checks (links, quantified bullets, generic names…) |
| Critic | agent | Gaps, suggestions, verdict, and a confidence check |

If the critic is not confident the score is well-supported, it loops back to the
matcher once for a re-evaluation (capped to avoid infinite loops).

```
                resume.pdf          job description
                    │                     │
                    ▼                     ▼
                 parser ─────────────► jd_parser
                    └──────────┬──────────┘
                               ▼
                    ┌────► matcher ──► scorer ──► hygiene ──► critic ─┐
                    │                                                  │
                    └──────── self-correction (≤1 retry) ◄────────────┘
                                          │
                                          ▼
                          ranked, explainable results
```

## Status

✅ Core engine + HTTP API complete and tested (68 tests, all live-verified
against Gemini). See `docs/architecture.md` for the full design.

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

## Usage

### Run the API

```bash
uvicorn cv_align_agents.api.main:app --reload
# open http://127.0.0.1:8000/docs for interactive Swagger UI
```

Screen resumes against a job description:

```bash
# Recruiter mode: rank multiple resumes
curl -X POST http://127.0.0.1:8000/screen \
  -F "jd=Backend engineer. Required: Python, Go, PostgreSQL, 3+ years." \
  -F "mode=recruiter" \
  -F "resumes=@alice.pdf" \
  -F "resumes=@bob.pdf"

# Candidate mode: feedback to improve one CV
curl -X POST http://127.0.0.1:8000/screen \
  -F "jd=Backend engineer..." \
  -F "mode=candidate" \
  -F "resumes=@my_cv.pdf"

# Retrieve a stored run later
curl http://127.0.0.1:8000/runs/<run_id>
```

### Command-line scripts

```bash
python scripts/check_llm.py                       # verify your LLM key works
python scripts/parse_resume.py resume.pdf         # PDF → structured resume
python scripts/screen.py resume.pdf job.txt       # full pipeline + agent trace
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/screen` | Screen resume PDF(s) against a JD; returns ranked results |
| GET | `/runs/{id}` | Retrieve a stored screening run |
| GET | `/runs` | List recent runs |

`/screen` form fields: `jd` (text), `mode` (`candidate`\|`recruiter`),
`critic_mode` (`fast`\|`full`), `critic_top_k` (int), `resumes` (PDF file(s)).

## Testing

```bash
pip install -e ".[dev]"
pytest -q                       # 68 tests, fully offline (fake LLM)
ruff check src tests scripts    # lint
```

## Project layout

```
src/cv_align_agents/
├── state.py          # Shared Pydantic state + public result models
├── settings.py       # Typed config from .env
├── pdf.py            # Deterministic PDF → text
├── llm/              # Provider-abstracted LLM client (Gemini/Groq)
├── agents/           # parser, jd_parser, matcher, critic
├── pipeline/         # scorer, hygiene (deterministic), graph, screen
├── api/              # FastAPI app
└── storage/          # SQLite run persistence
```

## Deployment

The app ships with a `Dockerfile` and a Render blueprint (`render.yaml`).

```bash
# Build and run locally with Docker
docker build -t cv-align-agents .
docker run -p 8000:8000 -e GOOGLE_API_KEY=your-key cv-align-agents
```

**Render (free tier):** push to GitHub, create a new Blueprint from the repo,
and set `GOOGLE_API_KEY` in the dashboard. The blueprint wires `/health` checks
and keeps the SQLite file in `/tmp` (free-tier disk is ephemeral, so run history
resets on redeploy — fine for a demo).

## License

[MIT](LICENSE) © 2026 Shreyansh Dutt Mehra
