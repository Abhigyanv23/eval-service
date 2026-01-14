import json
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "data" / "demo"


def _load(name):
    return [json.loads(line) for line in (DEMO / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def test_healthz_and_rubrics(client):
    assert client.get("/healthz").json()["status"] == "ok"
    names = {r["name"] for r in client.get("/rubrics").json()}
    assert {"summarization", "rag_faithfulness", "safety"} <= names


def test_single_eval_shape(client):
    r = client.post("/evals", json={
        "rubric": "rag_faithfulness", "prompt": "Is the K200 dishwasher safe?",
        "output": "Yes, the K200 is fully dishwasher safe, and it has a lifetime warranty.",
        "context": ["The K200 is not dishwasher safe. The K200 comes with a 2-year manufacturer's warranty."],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert {d["dimension"] for d in body["dimension_scores"]} == {"faithfulness", "relevance", "completeness"}
    for d in body["dimension_scores"]:
        assert d["evidence"] and "grounded" in d["evidence"][0]
    assert 0 <= body["normalized_score"] <= 1


def test_inline_rubric(client):
    rubric = {"name": "brevity", "task_type": "generation", "description": "Short answers",
              "dimensions": [{"name": "conciseness", "description": "short",
                              "anchors": {1: "long", 2: "x", 3: "x", 4: "x", 5: "short"}}]}
    r = client.post("/evals", json={"rubric": rubric, "prompt": "a long prompt " * 20, "output": "Short."})
    assert r.status_code == 200, r.text
    assert r.json()["rubric"] == "brevity"


def test_unknown_rubric_404(client):
    assert client.post("/evals", json={"rubric": "nope", "prompt": "p", "output": "o"}).status_code == 404


def test_rubric_versioning(client):
    spec = client.get("/rubrics/tone").json()
    spec["pass_threshold"] = 4.0
    created = client.post("/rubrics", json=spec).json()
    assert created["version"] == 2
    assert client.get("/rubrics/tone", params={"version": 1}).json()["pass_threshold"] == 3.5


def _batch(client, name, file, feature="support-rag", rubric="rag_faithfulness"):
    r = client.post("/evals/batch", json={"name": name, "feature": feature, "rubric": rubric, "items": _load(file)})
    assert r.status_code == 202, r.text
    return r.json()["run_id"]


def test_batch_report_gate_and_regression(client):
    v1 = _batch(client, "rag v1", "support_rag_v1.jsonl")
    run = client.get(f"/runs/{v1}").json()
    assert run["status"] == "done" and run["completed"] == run["total"] == 8

    report = client.get(f"/runs/{v1}/report").json()
    assert report["counts"]["scored"] == 8
    assert set(report["dimensions"]) == {"faithfulness", "relevance", "completeness"}
    assert len(report["pass_rate_ci95"]) == 2

    v2 = _batch(client, "rag v2", "support_rag_v2.jsonl")
    cmp = client.get("/runs/compare", params={"a": v1, "b": v2}).json()
    assert cmp["dimensions"]["faithfulness"]["delta"] < 0

    gate = client.post("/gates/check", json={
        "run_id": v2, "baseline_run_id": "previous",
        "thresholds": {"min_pass_rate": 0.0, "max_dimension_drop": 0.1},
    }).json()
    assert gate["baseline_run_id"] == v1
    assert gate["passed"] is False
    assert any(c["name"] == "max_drop:faithfulness" and not c["passed"] for c in gate["checks"])
    assert client.get(f"/runs/{v2}").json()["gate_passed"] is False


def test_override_and_metric_health(client):
    run_id = _batch(client, "sum", "news_summarizer.jsonl", feature="summarizer", rubric="summarization")
    results = client.get(f"/runs/{run_id}/results").json()
    target = results[0]
    r = client.post(f"/results/{target['id']}/override",
                    json={"reviewer": "qa", "passed": not target["passed"], "note": "disagree"})
    assert r.json()["human_override"]["reviewer"] == "qa"
    health = client.get("/metrics/health", params={"feature": "summarizer"}).json()
    assert health["human_reviewed"] == 1 and health["human_override_rate"] == 1.0
    assert "faithfulness" in health["dimensions"]


def test_calibration_and_drift(client):
    ex = client.get("/calibration/examples", params={"rubric": "summarization"}).json()
    assert len(ex) == 25
    assert len(client.get("/calibration/examples").json()) == 110
    a = client.post("/calibration/runs", json={"rubric": "summarization", "judge": "heuristic"}).json()
    b = client.post("/calibration/runs", json={"rubric": "summarization", "judge": "heuristic"}).json()
    full = client.get(f"/calibration/runs/{a['id']}").json()
    assert full["status"] == "done", full.get("error")
    m = full["metrics"]
    assert m["n_scored"] == 25
    assert "weighted_kappa" in m["dimensions"]["faithfulness"]
    assert "best_threshold" in m["threshold_tuning"]
    drift = client.get("/calibration/drift", params={"a": a["id"], "b": b["id"]}).json()
    assert drift["pass_decision_flips"] == 0 and drift["drifted"] is False  # same deterministic judge


def test_consistency_run(client):
    r = client.post("/consistency/runs", json={"rubric": "rag_faithfulness", "judge": "heuristic", "repeats": 3,
                                               "max_items": 10})
    assert r.status_code == 202, r.text
    run = client.get(f"/consistency/runs/{r.json()['id']}").json()
    assert run["status"] == "done", run.get("error")
    c = run["metrics"]["checks"]
    # The heuristic judge is deterministic: identical inputs must give identical scores
    assert c["repeat"]["pass_flip_rate"] == 0 and c["repeat"]["verdict"] == "stable"
    assert all(v["stable_rate"] == 1.0 for v in c["repeat"]["dimensions"].values())
    assert c["context_order"]["n"] == 10
    assert c["injection"]["attack_success_rate"] == 0
    assert run["metrics"]["cost"]["total_usd"] == 0.0


def test_report_includes_cost_and_cost_gate(client):
    run_id = _batch(client, "cost", "support_rag_v1.jsonl", feature="cost-check")
    rep = client.get(f"/runs/{run_id}/report").json()
    assert rep["cost"]["total_usd"] == 0.0 and rep["cost"]["items"] == 8
    gate = client.post("/gates/check", json={"run_id": run_id, "thresholds": {
        "min_pass_rate": 0.0, "max_cost_per_item_usd": 0.01}}).json()
    assert any(c["name"] == "max_cost_per_item" and c["passed"] for c in gate["checks"])
