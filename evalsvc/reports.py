"""Run reports, run comparison, release gates and metric health."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalsvc import stats
from evalsvc.models import EvalResult, EvalRun, HumanOverride
from evalsvc.schemas import GateCheck, GateRequest, GateResponse


def _results(session: Session, run_id: str) -> list[EvalResult]:
    return list(session.scalars(select(EvalResult).where(EvalResult.run_id == run_id).order_by(EvalResult.created_at)))


def run_report(session: Session, run: EvalRun, max_failed: int = 25) -> dict[str, Any]:
    rows = _results(session, run.id)
    ok = [r for r in rows if r.status == "ok"]
    errors = [r for r in rows if r.status == "error"]
    pending = [r for r in rows if r.status == "pending"]
    n_pass = sum(1 for r in ok if r.passed)
    scores = [r.overall_score for r in ok]

    dims: dict[str, list[int]] = defaultdict(list)
    dim_fail: Counter[str] = Counter()
    for r in ok:
        for d in r.dimension_scores:
            dims[d["dimension"]].append(d["score"])
            if not d["passed"]:
                dim_fail[d["dimension"]] += 1
    dimension_stats = {
        name: {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3),
               "mean_ci95": [round(x, 3) for x in stats.bootstrap_mean_ci(v)],
               "fail_rate": round(dim_fail[name] / len(v), 3),
               "histogram": {str(k): v.count(k) for k in range(1, 6)}}
        for name, v in dims.items()
    }
    weakest = min(dimension_stats, key=lambda k: dimension_stats[k]["mean"]) if dimension_stats else None

    failed = [r for r in ok if not r.passed]
    failed.sort(key=lambda r: r.overall_score or 0)
    failed_cases = [{
        "result_id": r.id, "item_id": r.item_id, "overall_score": r.overall_score,
        "explanation": r.explanation, "warnings": r.policy_warnings,
        "prompt": (r.request.get("prompt") or "")[:400], "output": (r.request.get("output") or "")[:4000],
        "failing_dimensions": [{"dimension": d["dimension"], "score": d["score"], "min_score": d.get("min_score"),
                                "reasoning": d["reasoning"],
                                "evidence": [{"quote": e["quote"], "issue": e["issue"], "start": e.get("start"),
                                              "end": e.get("end"), "grounded": e.get("grounded")} for e in d["evidence"]]}
                               for d in r.dimension_scores if not d["passed"]],
    } for r in failed[:max_failed]]

    warn_counts = Counter(w.split(":")[0] for r in rows for w in (r.policy_warnings or []))
    return {
        "run_id": run.id, "name": run.name, "feature": run.feature, "status": run.status,
        "rubric": f"{run.rubric_name}@v{run.rubric_version}", "judge": run.judge, "git_sha": run.git_sha,
        "counts": {"total": len(rows), "scored": len(ok), "errors": len(errors), "pending": len(pending),
                   "passed": n_pass, "failed": len(ok) - n_pass,
                   "needs_human_review": sum(1 for r in ok if r.needs_human_review)},
        "pass_rate": round(n_pass / len(ok), 4) if ok else None,
        "pass_rate_ci95": [round(x, 4) for x in stats.wilson_interval(n_pass, len(ok))] if ok else None,
        "error_rate": round(len(errors) / (len(ok) + len(errors)), 4) if ok or errors else None,
        "overall": {"mean": round(float(np.mean(scores)), 3) if scores else None,
                    "mean_ci95": [round(x, 3) for x in stats.bootstrap_mean_ci(scores)] if scores else None,
                    "std": round(float(np.std(scores)), 3) if scores else None,
                    "histogram": _histogram(scores)},
        "dimensions": dimension_stats,
        "weakest_dimension": weakest,
        "warning_counts": dict(warn_counts),
        "latency_ms": {"p50": _pct([r.latency_ms for r in ok], 50), "p95": _pct([r.latency_ms for r in ok], 95)},
        "cost": cost_summary([r for r in rows if r.status != "pending"]),
        "failed_cases": failed_cases,
        "gate": {"passed": run.gate_passed, "detail": run.gate_detail},
    }


def cost_summary(rows: list[EvalResult]) -> dict[str, Any]:
    """Token and dollar totals. Cost is None if any item used a judge model with no known price."""
    n = len(rows)
    costs = [r.cost_usd for r in rows]
    total = None if any(c is None for c in costs) else round(sum(costs), 6)
    calls = sum(r.judge_calls or 0 for r in rows)
    cached = sum(r.cached_calls or 0 for r in rows)
    return {
        "items": n, "judge_calls": calls, "cached_calls": cached,
        "cache_hit_rate": round(cached / (calls + cached), 4) if calls + cached else None,
        "input_tokens": sum(r.input_tokens or 0 for r in rows),
        "output_tokens": sum(r.output_tokens or 0 for r in rows),
        "total_usd": total,
        "per_item_usd": round(total / n, 6) if total is not None and n else None,
        "per_1k_items_usd": round(1000 * total / n, 2) if total is not None and n else None,
    }


def _histogram(scores: list[float]) -> dict[str, int]:
    bins = ["1.0-1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "4.0-4.5", "4.5-5.0"]
    h = dict.fromkeys(bins, 0)
    for s in scores:
        h[bins[min(int((s - 1) / 0.5), 7)]] += 1
    return h


def _pct(values: list[int | None], q: int) -> int | None:
    v = [x for x in values if x is not None]
    return int(np.percentile(v, q)) if v else None


def compare_runs(session: Session, a: EvalRun, b: EvalRun) -> dict[str, Any]:
    ra, rb = run_report(session, a, max_failed=0), run_report(session, b, max_failed=0)
    dims = sorted(set(ra["dimensions"]) | set(rb["dimensions"]))

    def delta(x, y):
        return None if x is None or y is None else round(y - x, 4)

    # Item-level regressions: same item_id, passed in A, failed in B.
    ia = {r.item_id: r for r in _results(session, a.id) if r.status == "ok"}
    ib = {r.item_id: r for r in _results(session, b.id) if r.status == "ok"}
    common = ia.keys() & ib.keys()
    regressions = sorted(i for i in common if ia[i].passed and not ib[i].passed)
    fixes = sorted(i for i in common if not ia[i].passed and ib[i].passed)
    return {
        "a": {"run_id": a.id, "name": a.name, "pass_rate": ra["pass_rate"], "mean": ra["overall"]["mean"]},
        "b": {"run_id": b.id, "name": b.name, "pass_rate": rb["pass_rate"], "mean": rb["overall"]["mean"]},
        "pass_rate_delta": delta(ra["pass_rate"], rb["pass_rate"]),
        "mean_delta": delta(ra["overall"]["mean"], rb["overall"]["mean"]),
        "dimensions": {d: {"a": ra["dimensions"].get(d, {}).get("mean"), "b": rb["dimensions"].get(d, {}).get("mean"),
                           "delta": delta(ra["dimensions"].get(d, {}).get("mean"),
                                          rb["dimensions"].get(d, {}).get("mean"))} for d in dims},
        "n_common_items": len(common), "regressed_items": regressions, "fixed_items": fixes,
        "same_judge": a.judge == b.judge, "same_rubric": (a.rubric_name, a.rubric_version) == (b.rubric_name, b.rubric_version),
    }


def resolve_baseline(session: Session, run: EvalRun, baseline_id: str | None) -> EvalRun | None:
    if baseline_id is None:
        return None
    if baseline_id != "previous":
        return session.get(EvalRun, baseline_id)
    return session.scalars(select(EvalRun).where(EvalRun.feature == run.feature, EvalRun.status == "done",
                                                 EvalRun.rubric_name == run.rubric_name, EvalRun.id != run.id,
                                                 EvalRun.created_at < run.created_at)
                           .order_by(EvalRun.created_at.desc()).limit(1)).first()


def check_gate(session: Session, run: EvalRun, req: GateRequest) -> GateResponse:
    t = req.thresholds
    rep = run_report(session, run, max_failed=0)
    checks: list[GateCheck] = []

    if run.status != "done":
        checks.append(GateCheck(name="run_complete", passed=False, actual=None, threshold=None,
                                detail=f"run status is '{run.status}'; gates need a finished run"))
    if t.min_pass_rate is not None:
        actual = rep["pass_rate_ci95"][0] if (t.use_lower_confidence_bound and rep["pass_rate_ci95"]) else rep["pass_rate"]
        label = "pass_rate (95% lower bound)" if t.use_lower_confidence_bound else "pass_rate"
        checks.append(GateCheck(name="min_pass_rate", passed=actual is not None and actual >= t.min_pass_rate,
                                actual=actual, threshold=t.min_pass_rate, detail=f"{label} >= {t.min_pass_rate}"))
    for dim, minimum in t.min_dimension_means.items():
        mean = rep["dimensions"].get(dim, {}).get("mean")
        checks.append(GateCheck(name=f"min_mean:{dim}", passed=mean is not None and mean >= minimum, actual=mean,
                                threshold=minimum,
                                detail=f"mean {dim} score >= {minimum}" if mean is not None
                                else f"dimension '{dim}' not in this run's rubric"))
    if t.max_cost_per_item_usd is not None:
        per_item = rep["cost"]["per_item_usd"]
        checks.append(GateCheck(name="max_cost_per_item", passed=per_item is not None and per_item <= t.max_cost_per_item_usd,
                                actual=per_item, threshold=t.max_cost_per_item_usd,
                                detail="mean judge cost per item (USD)" if per_item is not None
                                else "cost unknown: judge model has no price; add it via EVALSVC_PRICING_JSON"))
    err = rep["error_rate"] or 0.0
    checks.append(GateCheck(name="max_error_rate", passed=err <= t.max_error_rate, actual=err,
                            threshold=t.max_error_rate, detail="share of items the judge failed to score"))

    baseline = resolve_baseline(session, run, req.baseline_run_id)
    if req.baseline_run_id and baseline is None:
        checks.append(GateCheck(name="baseline", passed=True, actual=None, threshold=None,
                                detail="no baseline run found; regression checks skipped"))
    if baseline is not None:
        cmp = compare_runs(session, baseline, run)
        if not cmp["same_judge"] or not cmp["same_rubric"]:
            checks.append(GateCheck(name="comparable_baseline", passed=True, actual=None, threshold=None,
                                    detail="WARNING: baseline used a different judge or rubric version; "
                                           "deltas mix model change with judge change"))
        if t.max_pass_rate_drop is not None and cmp["pass_rate_delta"] is not None:
            checks.append(GateCheck(name="max_pass_rate_drop", passed=-cmp["pass_rate_delta"] <= t.max_pass_rate_drop,
                                    actual=cmp["pass_rate_delta"], threshold=-t.max_pass_rate_drop,
                                    detail=f"pass rate change vs baseline {baseline.id[:8]}"))
        if t.max_dimension_drop is not None:
            for dim, v in cmp["dimensions"].items():
                if v["delta"] is not None:
                    checks.append(GateCheck(name=f"max_drop:{dim}", passed=-v["delta"] <= t.max_dimension_drop,
                                            actual=v["delta"], threshold=-t.max_dimension_drop,
                                            detail=f"{dim} mean change vs baseline"))

    passed = all(c.passed for c in checks)
    failed = [c.name for c in checks if not c.passed]
    summary = "Gate PASSED" if passed else f"Gate FAILED: {', '.join(failed)}"
    resp = GateResponse(run_id=run.id, baseline_run_id=baseline.id if baseline else None, passed=passed,
                        checks=checks, summary=summary)
    run.gate_passed = passed
    run.gate_detail = resp.model_dump(mode="json")
    session.commit()
    return resp


def metric_health(session: Session, rubric_name: str | None = None, feature: str | None = None,
                  limit_results: int = 5000) -> dict[str, Any]:
    """Is the metric itself trustworthy? Score variance, judge-vs-judge disagreement, human override rate."""
    q = select(EvalResult).where(EvalResult.status.in_(["ok", "error"]))
    if rubric_name:
        q = q.where(EvalResult.rubric_name == rubric_name)
    if feature:
        q = q.join(EvalRun, EvalRun.id == EvalResult.run_id).where(EvalRun.feature == feature)
    rows = list(session.scalars(q.order_by(EvalResult.created_at.desc()).limit(limit_results)))
    ok = [r for r in rows if r.status == "ok"]

    by_dim: dict[str, list[int]] = defaultdict(list)
    sec_pairs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    conf_by_dim: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        for d in r.dimension_scores:
            by_dim[d["dimension"]].append(d["score"])
            conf_by_dim[d["dimension"]].append(d["confidence"])
            if d.get("secondary_score") is not None:
                sec_pairs[d["dimension"]].append((d["score"], d["secondary_score"]))

    ids = [r.id for r in ok]
    overrides = list(session.scalars(select(HumanOverride).where(HumanOverride.result_id.in_(ids)))) if ids else []
    by_result = {o.result_id: o for o in sorted(overrides, key=lambda o: o.created_at)}
    flipped = sum(1 for rid, o in by_result.items() if o.passed != next(r.passed for r in ok if r.id == rid))
    reviewed = sum(1 for r in ok if r.needs_human_review)

    dims = {}
    for name, v in by_dim.items():
        pairs = sec_pairs.get(name, [])
        dims[name] = {
            "n": len(v), "mean": round(float(np.mean(v)), 3), "variance": round(float(np.var(v)), 3),
            "share_at_mode": round(Counter(v).most_common(1)[0][1] / len(v), 3),  # ~1.0 => metric not discriminating
            "mean_confidence": round(float(np.mean(conf_by_dim[name])), 3),
            "judge_disagreement": None if not pairs else {
                "n": len(pairs),
                "mean_abs_diff": round(float(np.mean([abs(a - b) for a, b in pairs])), 3),
                "weighted_kappa": round(stats.weighted_kappa([a for a, _ in pairs], [b for _, b in pairs]), 3),
            },
        }
    return {
        "rubric": rubric_name, "feature": feature, "n_results": len(rows), "n_scored": len(ok),
        "error_rate": round(1 - len(ok) / len(rows), 4) if rows else None,
        "flagged_for_review_rate": round(reviewed / len(ok), 4) if ok else None,
        "human_reviewed": len(by_result),
        "human_override_rate": round(flipped / len(by_result), 4) if by_result else None,
        "dimensions": dims,
        "n_runs": session.scalar(select(func.count(func.distinct(EvalResult.run_id)))
                                 .where(EvalResult.id.in_(ids))) if ids else 0,
    }
