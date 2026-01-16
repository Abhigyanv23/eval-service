"""Portfolio demo: two AI features share one eval service with different rubrics.

  1. support-rag      - a RAG support bot, scored with rag_faithfulness. Release v1, then v2 (a
                        prompt change that reads well but invents details). The release gate catches it.
  2. news-summarizer  - a summarization feature, scored with summarization.

Start the API first (uvicorn evalsvc.api:app), then:  python scripts/demo.py [--judge anthropic]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import httpx

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"


def load(name):
    return [json.loads(x) for x in (DEMO / name).read_text(encoding="utf-8").splitlines() if x.strip()]


def run_batch(c, name, feature, rubric, file, judge):
    r = c.post("/evals/batch", json={"name": name, "feature": feature, "rubric": rubric, "judge": judge,
                                     "items": load(file)})
    r.raise_for_status()
    run_id = r.json()["run_id"]
    while c.get(f"/runs/{run_id}").json()["status"] != "done":
        time.sleep(1)
    rep = c.get(f"/runs/{run_id}/report").json()
    dims = ", ".join(f"{k} {v['mean']:.2f}" for k, v in rep["dimensions"].items())
    print(f"  {name:<22} pass rate {rep['pass_rate']:.0%} (95% CI {rep['pass_rate_ci95'][0]:.0%}-"
          f"{rep['pass_rate_ci95'][1]:.0%}) | {dims} | weakest: {rep['weakest_dimension']}")
    return run_id, rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default=os.environ.get("EVALSVC_API_URL", "http://localhost:8000"))
    p.add_argument("--judge", default=None, help="heuristic (default, offline), anthropic, openai, local:<model>")
    a = p.parse_args()
    headers = {"X-API-Key": os.environ["EVALSVC_API_KEY"]} if os.environ.get("EVALSVC_API_KEY") else {}
    c = httpx.Client(base_url=a.api, headers=headers, timeout=120)
    print(f"Eval service: {c.get('/healthz').json()}\n")

    print("Feature 1: support-rag (rubric: rag_faithfulness)")
    v1, _ = run_batch(c, "support-rag v1", "support-rag", "rag_faithfulness", "support_rag_v1.jsonl", a.judge)
    v2, rep2 = run_batch(c, "support-rag v2", "support-rag", "rag_faithfulness", "support_rag_v2.jsonl", a.judge)

    gate = c.post("/gates/check", json={"run_id": v2, "baseline_run_id": v1, "thresholds": {
        "min_pass_rate": 0.75, "min_dimension_means": {"faithfulness": 4.0}, "max_dimension_drop": 0.25}}).json()
    print(f"\n  Release gate for v2: {gate['summary']}")
    for ch in gate["checks"]:
        print(f"    [{'PASS' if ch['passed'] else 'FAIL'}] {ch['name']:<24} actual={ch['actual']} threshold={ch['threshold']}")
    cmp = c.get("/runs/compare", params={"a": v1, "b": v2}).json()
    print(f"  Regressed items (passed in v1, fail in v2): {cmp['regressed_items']}")
    for fc in rep2["failed_cases"][:3]:
        print(f"    - {fc['item_id']}: {fc['explanation'][:160]}")

    print("\nFeature 2: news-summarizer (rubric: summarization)")
    _, rep_s = run_batch(c, "news-summarizer v1", "news-summarizer", "summarization", "news_summarizer.jsonl", a.judge)
    for fc in rep_s["failed_cases"][:3]:
        print(f"    - {fc['item_id']}: {fc['explanation'][:160]}")

    print("\nOpen the dashboard (streamlit run dashboard/app.py) to explore these runs.")


if __name__ == "__main__":
    main()
