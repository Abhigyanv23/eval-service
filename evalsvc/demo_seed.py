"""Populate an empty eval service with the demo story, through the public API.

Used by the dashboard's embedded mode (e.g. on Streamlit Community Cloud), where there is no separate API
or database: every restart starts from an empty SQLite file and calls seed_demo() once. Uses the offline
heuristic judge only, so it needs no keys and costs nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"
CALIBRATED_RUBRICS = ["summarization", "rag_faithfulness", "classification", "tone", "completeness", "safety"]
GATE = {"min_pass_rate": 0.6, "min_dimension_means": {"faithfulness": 4.0}, "max_dimension_drop": 0.25}


def _items(name: str) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (DEMO / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def seed_demo(client) -> dict[str, Any]:
    """`client` is anything with .get/.post like httpx.Client or FastAPI's TestClient. Batches must complete
    synchronously (EVALSVC_SYNC_TASKS=true), which embedded mode sets."""
    if client.get("/runs", params={"limit": 1}).json():
        return {"seeded": False}

    def batch(name, feature, rubric, file):
        r = client.post("/evals/batch", json={"name": name, "feature": feature, "rubric": rubric,
                                              "judge": "heuristic", "items": _items(file)})
        r.raise_for_status()
        return r.json()["run_id"]

    def gate(run_id, baseline):
        client.post("/gates/check", json={"run_id": run_id, "baseline_run_id": baseline, "thresholds": GATE})

    # Feature 1: a RAG support bot. v2 is a prompt change that invents details and is blocked by the gate;
    # v3 reverts the regression and passes.
    v1 = batch("support-rag v1", "support-rag", "rag_faithfulness", "support_rag_v1.jsonl")
    gate(v1, None)
    v2 = batch("support-rag v2", "support-rag", "rag_faithfulness", "support_rag_v2.jsonl")
    gate(v2, v1)
    v3 = batch("support-rag v3 (fix)", "support-rag", "rag_faithfulness", "support_rag_v1.jsonl")
    gate(v3, v2)

    # Feature 2: a news summarizer.
    s1 = batch("news-summarizer v1", "news-summarizer", "summarization", "news_summarizer.jsonl")

    # A human reviewer disagreeing with one judge decision, so metric health has an override to show.
    results = client.get(f"/runs/{s1}/results").json()
    if results:
        first = results[0]
        client.post(f"/results/{first['id']}/override",
                    json={"reviewer": "demo-reviewer", "passed": not first["passed"],
                          "note": "Seeded example of a human override"})

    for rubric in CALIBRATED_RUBRICS:
        client.post("/calibration/runs", json={"rubric": rubric, "judge": "heuristic"})
    for rubric in ("rag_faithfulness", "tone"):
        client.post("/consistency/runs", json={"rubric": rubric, "judge": "heuristic", "repeats": 3, "max_items": 25})
    return {"seeded": True, "runs": [v1, v2, v3, s1]}
