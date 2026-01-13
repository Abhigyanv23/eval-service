"""Persistence around the scoring engine: single evals, batch runs and per-item processing."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalsvc.db import SessionLocal
from evalsvc.judges import get_judge
from evalsvc.models import EvalResult, EvalRun, HumanOverride
from evalsvc.rubric_store import get_rubric
from evalsvc.schemas import BatchRequest, EvalItem, EvalRequest, EvalResultOut, RubricSpec, RunOut
from evalsvc.scoring import Scored, evaluate


def _apply(row: EvalResult, scored: Scored) -> None:
    row.status = scored.status
    row.overall_score = scored.overall_score
    row.passed = scored.passed
    row.confidence = scored.confidence
    row.needs_human_review = scored.needs_human_review
    row.dimension_scores = [d.model_dump(mode="json") for d in scored.dimension_scores]
    row.explanation = scored.explanation
    row.policy_warnings = scored.policy_warnings
    row.error = scored.error
    row.latency_ms = scored.latency_ms
    row.input_tokens = scored.usage.input_tokens
    row.output_tokens = scored.usage.output_tokens
    row.judge_calls = scored.usage.calls
    row.cached_calls = scored.usage.cached
    row.cost_usd = scored.cost_usd


def result_out(row: EvalResult, override: HumanOverride | None = None) -> EvalResultOut:
    return EvalResultOut(
        id=row.id, run_id=row.run_id, item_id=row.item_id, rubric=row.rubric_name,
        rubric_version=row.rubric_version, judge=row.judge, status=row.status,
        overall_score=row.overall_score,
        normalized_score=None if row.overall_score is None else round((row.overall_score - 1) / 4, 4),
        passed=row.passed, dimension_scores=row.dimension_scores or [], explanation=row.explanation,
        confidence=row.confidence, policy_warnings=row.policy_warnings or [],
        needs_human_review=row.needs_human_review, latency_ms=row.latency_ms, error=row.error,
        usage={"input_tokens": row.input_tokens or 0, "output_tokens": row.output_tokens or 0,
               "judge_calls": row.judge_calls or 0, "cached_calls": row.cached_calls or 0},
        cost_usd=row.cost_usd,
        human_override=None if override is None else {
            "reviewer": override.reviewer, "passed": override.passed,
            "overall_score": override.overall_score, "note": override.note,
            "created_at": override.created_at.isoformat()},
    )


def evaluate_one(session: Session, req: EvalRequest) -> EvalResultOut:
    rubric = req.rubric if isinstance(req.rubric, RubricSpec) else get_rubric(session, req.rubric, req.rubric_version)
    judge = get_judge(req.judge)
    secondary = get_judge(req.secondary_judge) if req.secondary_judge else None
    item = EvalItem.model_validate(req.model_dump(include=set(EvalItem.model_fields)))
    scored = evaluate(item, rubric, judge, secondary)
    row = EvalResult(item_id=req.item_id, request=req.model_dump(mode="json"), rubric_name=rubric.name,
                     rubric_version=rubric.version, judge=judge.id)
    _apply(row, scored)
    session.add(row)
    session.commit()
    return result_out(row)


def create_batch(session: Session, req: BatchRequest) -> tuple[EvalRun, list[str]]:
    rubric = get_rubric(session, req.rubric, req.rubric_version)  # resolve "latest" once, pin the version
    judge = get_judge(req.judge)
    secondary = get_judge(req.secondary_judge) if req.secondary_judge else None
    run = EvalRun(name=req.name, feature=req.feature, rubric_name=rubric.name, rubric_version=rubric.version,
                  judge=judge.id, secondary_judge=secondary.id if secondary else None, status="pending",
                  total=len(req.items), git_sha=req.git_sha, run_metadata=req.metadata)
    session.add(run)
    session.flush()
    ids = []
    for i, item in enumerate(req.items):
        row = EvalResult(run_id=run.id, item_id=item.item_id or str(i), request=item.model_dump(mode="json"),
                         rubric_name=rubric.name, rubric_version=rubric.version, judge=judge.id, status="pending")
        session.add(row)
        session.flush()
        ids.append(row.id)
    session.commit()
    return run, ids


def process_result(result_id: str) -> None:
    """Score one pending batch item. Idempotent: an item that is already scored is skipped."""
    with SessionLocal() as session:
        row = session.get(EvalResult, result_id)
        if row is None or row.status != "pending":
            return
        run = session.get(EvalRun, row.run_id) if row.run_id else None
        if run is not None and run.status == "pending":
            run.status = "running"
            session.commit()
        try:
            rubric = get_rubric(session, row.rubric_name, row.rubric_version)
            judge = get_judge(run.judge if run else row.judge)
            secondary = get_judge(run.secondary_judge) if run and run.secondary_judge else None
            scored = evaluate(EvalItem.model_validate(row.request), rubric, judge, secondary)
            _apply(row, scored)
        except Exception as e:  # noqa: BLE001 - record the failure on the item, keep the batch going
            row.status, row.error, row.policy_warnings = "error", f"{type(e).__name__}: {e}", ["judge_error"]
        session.commit()
        if run is not None:
            finalize_run_if_done(session, run.id)


def finalize_run_if_done(session: Session, run_id: str) -> None:
    pending = session.scalar(select(func.count()).select_from(EvalResult)
                             .where(EvalResult.run_id == run_id, EvalResult.status == "pending"))
    if pending == 0:
        run = session.get(EvalRun, run_id)
        if run and run.status != "done":
            run.status = "done"
            run.finished_at = datetime.now(timezone.utc)
            session.commit()


def run_out(session: Session, run: EvalRun) -> RunOut:
    agg = session.execute(
        select(func.count().filter(EvalResult.status != "pending"),
               func.count().filter(EvalResult.passed.is_(True)),
               func.count().filter(EvalResult.status == "ok"),
               func.avg(EvalResult.overall_score),
               func.sum(EvalResult.cost_usd))
        .where(EvalResult.run_id == run.id)).one()
    completed, n_pass, n_ok, mean, cost = agg
    return RunOut(id=run.id, name=run.name, feature=run.feature, rubric_name=run.rubric_name,
                  rubric_version=run.rubric_version, judge=run.judge, secondary_judge=run.secondary_judge,
                  status=run.status, total=run.total, completed=completed, git_sha=run.git_sha,
                  gate_passed=run.gate_passed, created_at=run.created_at, finished_at=run.finished_at,
                  pass_rate=(n_pass / n_ok) if n_ok else None,
                  mean_score=round(mean, 3) if mean is not None else None,
                  cost_usd=round(cost, 4) if cost is not None else None)


def latest_overrides(session: Session, result_ids: list[str]) -> dict[str, HumanOverride]:
    if not result_ids:
        return {}
    rows = session.scalars(select(HumanOverride).where(HumanOverride.result_id.in_(result_ids))
                           .order_by(HumanOverride.created_at))
    return {r.result_id: r for r in rows}  # later rows overwrite earlier ones
