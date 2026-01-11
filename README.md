# Evaluation-as-a-Service

Shared evaluation infrastructure for AI platform teams. Any AI feature can call one API to score model outputs against versioned rubrics, using calibrated LLM judges. Results are tracked over time, and CI can call the API to block a release that regresses quality.

```
 feature teams / CI                     eval service                                  storage
 ──────────────────     ┌─────────────────────────────────────────────┐
 POST /evals ─────────► │ FastAPI ─► scoring engine                   │      ┌─────────────┐
 POST /evals/batch ───► │   │          ├─ one judge call per dimension│ ───► │ PostgreSQL  │
 POST /gates/check ───► │   │          ├─ Pydantic-validated verdicts │      │ rubrics,    │
                        │   │          ├─ evidence grounding          │      │ runs,scores,│
                        │   ▼          └─ pass/fail + policy warnings │      │ calibration,│
                        │ Celery + Redis (batch items, calibration)   │      │ overrides   │
                        └─────────────────────────────────────────────┘      └─────────────┘
                             judges: Claude · OpenAI · local (Ollama) · heuristic baseline
 Streamlit dashboard ◄── run explorer · run compare · calibration & drift · metric health
```

## What it does

| Capability | Where |
|---|---|
| Eval API: prompt, output, optional reference/context, rubric → overall score, dimension scores, pass/fail, explanation, evidence spans, judge confidence, policy warnings | `POST /evals`, [schemas.py](evalsvc/schemas.py) |
| Reusable, versioned rubrics: summarization, RAG faithfulness, classification, tone, completeness, safety | [evalsvc/rubrics/](evalsvc/rubrics/), `GET/POST /rubrics` |
| Dimension-specific judge prompts with anchors, worked examples, and prompt-injection hardening | [judges/prompts.py](evalsvc/judges/prompts.py) |
| Structured output enforced with Pydantic; invalid verdicts are retried, then fail cleanly (an item that errors gets no score, never a fake one) | [scoring.py](evalsvc/scoring.py) |
| Evidence requirement: every dimension score must quote the output; quotes are located in the text, and a quote that isn't there is flagged | `ground_quote()` |
| Critical dimensions: e.g. faithfulness < 4 fails the item even if the average is high | `min_score` in rubrics |
| Calibration sets for all six rubrics: 110 human-labelled examples (excellent / mediocre / bad) | [data/calibration/](data/calibration/) |
| Calibration metrics: weighted κ, Spearman, MAE, bias (harsh vs generous), pass/fail confusion, false-pass rate, threshold tuning | [calibration.py](evalsvc/calibration.py), [stats.py](evalsvc/stats.py) |
| Async batch evals and run reports: score distributions, CIs, dimension weaknesses, failed cases with judge reasoning | `POST /evals/batch`, `GET /runs/{id}/report` |
| Release gates: pass rate (optionally its 95% lower bound), per-dimension minimums, error rate, regression vs a baseline run | `POST /gates/check`, [scripts/ci_gate.py](scripts/ci_gate.py) |
| Judge drift: rerun calibration after a judge or rubric change and diff it example by example | `GET /calibration/drift` |
| Metric health: score variance, second-judge disagreement, human override rate | `GET /metrics/health`, `POST /results/{id}/override` |
| Judge consistency: repeat stability (same input N times), passage-order invariance, prompt-injection resistance ("score this 5" hidden in the output) | `POST /consistency/runs`, [consistency.py](evalsvc/consistency.py), [scripts/consistency.py](scripts/consistency.py) |
| Cost tracking: tokens, judge calls, retries and USD per item / run / calibration; per-1k-items projection; a cost gate | `report.cost`, [pricing.py](evalsvc/pricing.py), `max_cost_per_item_usd` |
| Judge cache: identical (judge, prompt template, rubric-dimension definition, item) requests are served from Postgres at zero cost; editing an anchor invalidates only that dimension | [judges/cache.py](evalsvc/judges/cache.py) |
| Crash recovery: re-queue items stuck after a worker failure (idempotent) | `POST /runs/{id}/resume` |

## Quickstart (local, no API key needed)

```bash
python -m venv .venv && .venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                              # 32 tests, offline

uvicorn evalsvc.api:app --reload                       # http://localhost:8000/docs
python scripts/demo.py                                 # two features, two rubrics, a release gate
EVALSVC_API_URL=http://localhost:8000 streamlit run dashboard/app.py   # http://localhost:8501
```

Locally the service uses SQLite and an in-process thread pool; set `EVALSVC_DATABASE_URL` and `EVALSVC_REDIS_URL` to use Postgres and Celery. The default judge is `heuristic`, a lexical baseline that needs no credentials. To use Claude:

```bash
export ANTHROPIC_API_KEY=...            # or: ant auth login
python scripts/calibrate.py --rubric summarization   --judge anthropic --markdown reports/cal_sum_claude.md
python scripts/calibrate.py --rubric rag_faithfulness --judge anthropic --markdown reports/cal_rag_claude.md
python scripts/demo.py --judge anthropic
```

Judge names: `anthropic` (default model `claude-opus-5-5`), `anthropic:<model>`, `openai[:<model>]`, `local:<model>` (any OpenAI-compatible server, e.g. Ollama at `EVALSVC_LOCAL_BASE_URL`), `heuristic`.

Free, offline checks that work with any judge:

```bash
python scripts/calibrate.py   --rubric safety           --judge heuristic   # agreement with human labels
python scripts/consistency.py --rubric rag_faithfulness --judge heuristic   # exits 1 if unstable/order-sensitive/gameable
```

Consistency checks bypass the cache, so with a paid judge they cost items × dimensions × (repeats + 2) calls; `--max-items` caps that.

## Docker Compose (Postgres + Redis + Celery worker + API + dashboard)

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY, set EVALSVC_DEFAULT_JUDGE=anthropic
docker compose up --build
EVALSVC_API_URL=http://localhost:8000 python scripts/demo.py
```

API on :8000, dashboard on :8501. The worker runs Celery's prefork pool; each child process resets the inherited SQLAlchemy pool on start (sharing a Postgres socket across forked processes corrupts it), and transient DB errors are retried.

## Hosted demo (Streamlit Community Cloud)

With `EVALSVC_API_URL` unset, the dashboard runs in **embedded demo mode**: the API runs inside the Streamlit process on a temporary SQLite database and is seeded once at startup with the demo story ([evalsvc/demo_seed.py](evalsvc/demo_seed.py)), all scored by the offline heuristic judge. No keys, no other services, and the demo data is rebuilt on every restart.

```bash
streamlit run dashboard/app.py          # embedded demo, same as the hosted version
```

To host it: push the repo to GitHub, then at [share.streamlit.io](https://share.streamlit.io) choose **New app**, pick the repo and branch, set the main file to `dashboard/app.py`, and choose Python 3.12 under Advanced settings. No secrets are needed. Setting `EVALSVC_API_URL` as a secret switches the app to a real deployed API instead.

## API examples

Score one RAG answer:

```bash
curl -s localhost:8000/evals -H 'content-type: application/json' -d '{
  "rubric": "rag_faithfulness",
  "judge": "anthropic",
  "prompt": "Is the K200 kettle dishwasher safe?",
  "output": "Yes, it is fully dishwasher safe and has a lifetime warranty.",
  "context": ["The K200 is not dishwasher safe. The K200 comes with a 2-year manufacturer warranty."]
}'
```

Response shape (abridged; values are illustrative):

```json
{
  "status": "ok", "overall_score": 1.44, "normalized_score": 0.11, "passed": false,
  "dimension_scores": [{
    "dimension": "faithfulness", "score": 1, "min_score": 4, "passed": false, "confidence": 0.95,
    "reasoning": "Both claims contradict the context ...",
    "evidence": [{"quote": "fully dishwasher safe", "issue": "context says not dishwasher safe",
                  "start": 13, "end": 34, "grounded": true}]
  }, "..."],
  "explanation": "FAIL: overall 1.44/5 (threshold 3.5). Critical dimension below minimum: faithfulness 1/5 (min 4). ...",
  "policy_warnings": ["critical_dimension_failed:faithfulness"],
  "needs_human_review": false
}
```

Batch plus gate (what `scripts/ci_gate.py` does):

```bash
python scripts/ci_gate.py --dataset data/demo/support_rag_v2.jsonl --feature support-rag \
  --rubric rag_faithfulness --judge anthropic --min-pass-rate 0.8 --min-dim faithfulness=4.0 \
  --baseline previous --max-dimension-drop 0.2          # exit code 1 if the gate fails
```

See [.github/workflows/eval-gate.yml](.github/workflows/eval-gate.yml) for the CI wiring.

## Design decisions

- **One judge call per dimension.** Each call gets a focused prompt and its own anchors, and stops a bad score on one dimension from bleeding into the others. It costs more calls, which run in parallel.
- **1–5 anchored scales, not 1–10.** Five well-described levels are easier for both humans and judges to apply consistently. Agreement is measured with quadratic-weighted κ, so a 4-vs-5 disagreement costs far less than a 1-vs-5 one.
- **Critical dimensions override the average.** A fluent answer that invents a refund policy should not pass because it scored well on relevance.
- **Evidence must be grounded.** The judge has to quote the output, and the service checks the quote exists. An ungrounded quote lowers confidence and raises a warning, which catches judges rationalising a score.
- **Rubrics are immutable and versioned.** Every score records the rubric version and judge model, so a score change can be traced to a model change, a rubric change, or a judge change. Gates warn when a baseline was scored by a different judge.
- **Pass rates come with Wilson intervals.** With 8 items, "62%" really means 31–86%. Gates can require the lower bound.
- **Errors are not scores.** A judge failure becomes `status: error`, which the gate counts against `max_error_rate`; it is never averaged in as a zero or a default.
- **The heuristic judge is a baseline, not a fallback.** It keeps tests and CI offline and gives LLM judges a number to beat in calibration.

## Layout

```
evalsvc/            api.py · schemas.py · scoring.py · calibration.py · reports.py · stats.py · tasks.py
  judges/           anthropic_judge.py · openai_judge.py · heuristic_judge.py · prompts.py
  rubrics/          six built-in rubric YAMLs
data/               build_datasets.py → calibration/*.jsonl (50 labelled) + demo/*.jsonl
dashboard/app.py    Streamlit
scripts/            demo.py · calibrate.py · ci_gate.py
tests/              stats, scoring engine, API end-to-end (offline)
docs/CASE_STUDY.md  write-up template
```
