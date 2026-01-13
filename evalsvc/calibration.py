"""Judge calibration: score a human-labelled set with a judge and measure how far it is from the humans.

Also covers drift: when the judge model or rubric version changes, a fresh calibration run is compared
example-by-example with the previous one."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalsvc import stats
from evalsvc.db import SessionLocal
from evalsvc.judges import get_judge
from evalsvc.models import CalibrationExample, CalibrationRun
from evalsvc.rubric_store import get_rubric
from evalsvc.schemas import EvalItem
from evalsvc.scoring import evaluate

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "calibration"


def seed_calibration_examples(session: Session, data_dir: Path = DATA_DIR) -> int:
    """Load *.jsonl calibration files. Re-seeding updates labels in place (labels get corrected over time)."""
    n = 0
    for path in sorted(data_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ex = json.loads(line)
            row = session.get(CalibrationExample, ex["id"]) or CalibrationExample(id=ex["id"])
            row.rubric_name = ex["rubric"]
            row.quality_tier = ex["quality_tier"]
            row.request = {k: ex.get(k) for k in ("prompt", "output", "reference", "context")}
            row.human_scores = ex["human"]["scores"]
            row.human_pass = ex["human"]["pass"]
            row.notes = ex.get("notes")
            session.merge(row)
            n += 1
    session.commit()
    return n


def start_calibration(session: Session, rubric_name: str, judge_name: str | None) -> CalibrationRun:
    rubric = get_rubric(session, rubric_name)
    judge = get_judge(judge_name)
    count = session.scalar(select(CalibrationExample.id).where(CalibrationExample.rubric_name == rubric_name).limit(1))
    if count is None:
        raise LookupError(f"no calibration examples for rubric '{rubric_name}'")
    run = CalibrationRun(rubric_name=rubric.name, rubric_version=rubric.version, judge=judge.id, status="pending")
    session.add(run)
    session.commit()
    return run


def execute_calibration(calibration_run_id: str) -> None:
    with SessionLocal() as session:
        run = session.get(CalibrationRun, calibration_run_id)
        if run is None or run.status != "pending":
            return
        try:
            rubric = get_rubric(session, run.rubric_name, run.rubric_version)
            judge = get_judge(_judge_name_from_id(run.judge))
            examples = session.scalars(select(CalibrationExample)
                                       .where(CalibrationExample.rubric_name == run.rubric_name)
                                       .order_by(CalibrationExample.id)).all()
            with ThreadPoolExecutor(max_workers=4) as pool:
                scored = list(pool.map(lambda ex: evaluate(EvalItem(**ex.request), rubric, judge), examples))
            per_example = []
            for ex, s in zip(examples, scored, strict=True):
                per_example.append({
                    "id": ex.id, "quality_tier": ex.quality_tier, "status": s.status, "error": s.error,
                    "human_scores": ex.human_scores, "human_pass": ex.human_pass,
                    "human_overall": _weighted(ex.human_scores, rubric),
                    "judge_scores": {d.dimension: d.score for d in s.dimension_scores},
                    "judge_overall": s.overall_score, "judge_pass": s.passed,
                    "judge_reasoning": {d.dimension: d.reasoning for d in s.dimension_scores},
                    "warnings": s.policy_warnings,
                })
            run.per_example = per_example
            run.metrics = compute_metrics(per_example, rubric.pass_threshold, [d.name for d in rubric.dimensions])
            costs = [s.cost_usd for s in scored]
            run.metrics["cost"] = {
                "judge_calls": sum(s.usage.calls for s in scored), "cached_calls": sum(s.usage.cached for s in scored),
                "input_tokens": sum(s.usage.input_tokens for s in scored),
                "output_tokens": sum(s.usage.output_tokens for s in scored),
                "total_usd": None if any(c is None for c in costs) else round(sum(costs), 6)}
            run.status = "done"
        except Exception as e:  # noqa: BLE001
            run.status, run.error = "error", f"{type(e).__name__}: {e}"
        session.commit()


def _judge_name_from_id(judge_id: str) -> str:
    return "heuristic" if judge_id.startswith("heuristic") else judge_id


def _weighted(scores: dict[str, int], rubric) -> float:
    dims = [d for d in rubric.dimensions if d.name in scores]
    return round(sum(scores[d.name] * d.weight for d in dims) / sum(d.weight for d in dims), 3)


def compute_metrics(per_example: list[dict[str, Any]], pass_threshold: float, dims: list[str]) -> dict[str, Any]:
    ok = [e for e in per_example if e["status"] == "ok"]
    out: dict[str, Any] = {"n_examples": len(per_example), "n_scored": len(ok),
                           "n_errors": len(per_example) - len(ok), "pass_threshold": pass_threshold}
    out["dimensions"] = {}
    for d in dims:
        pairs = [(e["judge_scores"][d], e["human_scores"][d]) for e in ok
                 if d in e["judge_scores"] and d in e["human_scores"]]
        if pairs:
            j, h = zip(*pairs, strict=True)
            out["dimensions"][d] = stats.ordinal_agreement(j, h)

    jp = [bool(e["judge_pass"]) for e in ok]
    hp = [bool(e["human_pass"]) for e in ok]
    out["pass_fail"] = stats.binary_agreement(jp, hp)
    jo = [e["judge_overall"] for e in ok]
    ho = [e["human_overall"] for e in ok]
    out["overall"] = {
        "mae": float(np.mean(np.abs(np.subtract(jo, ho)))) if ok else None,
        "mean_bias": float(np.mean(np.subtract(jo, ho))) if ok else None,
        "spearman": stats.spearman(jo, ho),
    }
    # Threshold tuning: which judge-overall cut-off best reproduces the human pass/fail labels?
    out["threshold_tuning"] = stats.sweep_threshold(jo, hp)
    out["by_tier"] = {}
    for tier in ("excellent", "mediocre", "bad"):
        t = [e for e in ok if e["quality_tier"] == tier]
        if t:
            out["by_tier"][tier] = {"n": len(t),
                                    "judge_mean": round(float(np.mean([e["judge_overall"] for e in t])), 3),
                                    "human_mean": round(float(np.mean([e["human_overall"] for e in t])), 3)}
    # The largest disagreements are what you read when tuning the rubric prompts.
    worst = []
    for e in ok:
        for d in dims:
            if d in e["judge_scores"] and d in e["human_scores"]:
                gap = e["judge_scores"][d] - e["human_scores"][d]
                if abs(gap) >= 2:
                    worst.append({"id": e["id"], "dimension": d, "judge": e["judge_scores"][d],
                                  "human": e["human_scores"][d], "direction": "generous" if gap > 0 else "harsh",
                                  "judge_reasoning": e["judge_reasoning"].get(d, "")})
    out["largest_disagreements"] = sorted(worst, key=lambda w: -abs(w["judge"] - w["human"]))[:15]
    return out


def compare_calibration_runs(a: CalibrationRun, b: CalibrationRun) -> dict[str, Any]:
    """Drift report: how scores on the same labelled examples moved between two calibration runs (a -> b)."""
    ea = {e["id"]: e for e in (a.per_example or []) if e["status"] == "ok"}
    eb = {e["id"]: e for e in (b.per_example or []) if e["status"] == "ok"}
    common = sorted(ea.keys() & eb.keys())
    dims = sorted({d for i in common for d in ea[i]["judge_scores"]} & {d for i in common for d in eb[i]["judge_scores"]})
    per_dim = {}
    for d in dims:
        deltas = [eb[i]["judge_scores"][d] - ea[i]["judge_scores"][d] for i in common
                  if d in ea[i]["judge_scores"] and d in eb[i]["judge_scores"]]
        ka = (a.metrics or {}).get("dimensions", {}).get(d, {}).get("weighted_kappa")
        kb = (b.metrics or {}).get("dimensions", {}).get(d, {}).get("weighted_kappa")
        per_dim[d] = {"mean_shift": float(np.mean(deltas)) if deltas else None,
                      "share_changed": float(np.mean([x != 0 for x in deltas])) if deltas else None,
                      "kappa_before": ka, "kappa_after": kb,
                      "kappa_change": (kb - ka) if ka is not None and kb is not None else None}
    flips = [i for i in common if ea[i]["judge_pass"] != eb[i]["judge_pass"]]
    acc_a = (a.metrics or {}).get("pass_fail", {}).get("accuracy")
    acc_b = (b.metrics or {}).get("pass_fail", {}).get("accuracy")
    return {
        "from": {"id": a.id, "judge": a.judge, "rubric_version": a.rubric_version, "created_at": a.created_at.isoformat()},
        "to": {"id": b.id, "judge": b.judge, "rubric_version": b.rubric_version, "created_at": b.created_at.isoformat()},
        "n_common": len(common),
        "dimensions": per_dim,
        "pass_decision_flips": len(flips),
        "flipped_examples": flips,
        "pass_fail_accuracy_before": acc_a, "pass_fail_accuracy_after": acc_b,
        "drifted": bool(flips and len(flips) / max(len(common), 1) > 0.1)
                   or any(abs(v["mean_shift"] or 0) >= 0.5 for v in per_dim.values()),
    }
