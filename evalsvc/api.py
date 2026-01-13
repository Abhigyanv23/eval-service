"""HTTP API. Run with: uvicorn evalsvc.api:app --reload"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from evalsvc import __version__, reports, service
from evalsvc.calibration import compare_calibration_runs, seed_calibration_examples, start_calibration
from evalsvc.consistency import start_consistency
from evalsvc.config import settings
from evalsvc.db import SessionLocal, get_session, init_db
from evalsvc.models import CalibrationExample, CalibrationRun, ConsistencyRun, EvalResult, EvalRun, HumanOverride
from evalsvc.rubric_store import RubricNotFound, create_rubric_version, get_rubric, list_rubrics, seed_rubrics
from evalsvc.schemas import (BatchAccepted, BatchRequest, CalibrationRequest, CalibrationRunOut, ConsistencyRequest,
                             ConsistencyRunOut, EvalRequest, EvalResultOut, GateRequest, GateResponse, OverrideRequest,
                             RubricSpec, RunOut)
from evalsvc.tasks import dispatch_calibration, dispatch_consistency, dispatch_items


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with SessionLocal() as s:
        seed_rubrics(s)
        seed_calibration_examples(s)
    yield


def require_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


app = FastAPI(
    title="Evaluation-as-a-Service",
    version=__version__,
    description="Shared LLM-as-judge scoring, judge calibration, batch evals and release gates.",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)
DB = Annotated[Session, Depends(get_session)]


def _get_run(session: Session, run_id: str) -> EvalRun:
    run = session.get(EvalRun, run_id)
    if run is None:
        raise HTTPException(404, f"run {run_id} not found")
    return run


def _get_rubric_or_404(session: Session, name: str, version: int | None = None) -> RubricSpec:
    try:
        return get_rubric(session, name, version)
    except RubricNotFound as e:
        raise HTTPException(404, str(e)) from e


@app.get("/healthz", tags=["meta"])
def healthz() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "default_judge": settings.default_judge,
            "queue": "celery" if settings.redis_url else "in-process"}


# ----------------------------------------------------------------------------- rubrics

@app.get("/rubrics", response_model=list[RubricSpec], tags=["rubrics"])
def rubrics_list(session: DB):
    return list_rubrics(session)


@app.get("/rubrics/{name}", response_model=RubricSpec, tags=["rubrics"])
def rubrics_get(name: str, session: DB, version: int | None = None):
    return _get_rubric_or_404(session, name, version)


@app.post("/rubrics", response_model=RubricSpec, status_code=201, tags=["rubrics"])
def rubrics_create(spec: RubricSpec, session: DB):
    """Store a new version of a rubric. Existing versions are never modified."""
    return create_rubric_version(session, spec)


# ----------------------------------------------------------------------------- scoring

@app.post("/evals", response_model=EvalResultOut, tags=["evals"])
def eval_single(req: EvalRequest, session: DB):
    """Score one output synchronously."""
    if isinstance(req.rubric, str):
        _get_rubric_or_404(session, req.rubric, req.rubric_version)
    try:
        return service.evaluate_one(session, req)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/evals/batch", response_model=BatchAccepted, status_code=202, tags=["evals"])
def eval_batch(req: BatchRequest, session: DB):
    """Queue a batch; poll GET /runs/{run_id} until status is 'done'."""
    _get_rubric_or_404(session, req.rubric, req.rubric_version)
    try:
        run, ids = service.create_batch(session, req)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    dispatch_items(ids)
    session.refresh(run)
    return BatchAccepted(run_id=run.id, status=run.status, total=run.total,
                         status_url=f"/runs/{run.id}", report_url=f"/runs/{run.id}/report")


@app.get("/results/{result_id}", response_model=EvalResultOut, tags=["evals"])
def result_get(result_id: str, session: DB):
    row = session.get(EvalResult, result_id)
    if row is None:
        raise HTTPException(404, "result not found")
    return service.result_out(row, service.latest_overrides(session, [row.id]).get(row.id))


@app.post("/results/{result_id}/override", response_model=EvalResultOut, tags=["evals"])
def result_override(result_id: str, body: OverrideRequest, session: DB):
    """Record a human reviewer's decision. Feeds the human-override-rate health metric."""
    row = session.get(EvalResult, result_id)
    if row is None:
        raise HTTPException(404, "result not found")
    ov = HumanOverride(result_id=row.id, **body.model_dump())
    session.add(ov)
    session.commit()
    return service.result_out(row, ov)


# ----------------------------------------------------------------------------- runs

@app.get("/runs", response_model=list[RunOut], tags=["runs"])
def runs_list(session: DB, feature: str | None = None, limit: int = Query(50, le=500)):
    q = select(EvalRun).order_by(EvalRun.created_at.desc()).limit(limit)
    if feature:
        q = q.where(EvalRun.feature == feature)
    return [service.run_out(session, r) for r in session.scalars(q)]


@app.get("/runs/compare", tags=["runs"])
def runs_compare(a: str, b: str, session: DB) -> dict[str, Any]:
    return reports.compare_runs(session, _get_run(session, a), _get_run(session, b))


@app.get("/runs/{run_id}", response_model=RunOut, tags=["runs"])
def runs_get(run_id: str, session: DB):
    return service.run_out(session, _get_run(session, run_id))


@app.post("/runs/{run_id}/resume", response_model=RunOut, tags=["runs"])
def runs_resume(run_id: str, session: DB):
    """Re-queue items still pending (e.g. after a worker crash). Safe to call repeatedly: scored items are skipped."""
    run = _get_run(session, run_id)
    pending = list(session.scalars(select(EvalResult.id).where(EvalResult.run_id == run_id,
                                                               EvalResult.status == "pending")))
    if pending:
        dispatch_items(pending)
    else:
        service.finalize_run_if_done(session, run_id)
    session.refresh(run)
    return service.run_out(session, run)


@app.get("/runs/{run_id}/results", response_model=list[EvalResultOut], tags=["runs"])
def runs_results(run_id: str, session: DB, failed_only: bool = False, needs_review: bool = False,
                 limit: int = Query(500, le=5000), offset: int = 0):
    _get_run(session, run_id)
    q = select(EvalResult).where(EvalResult.run_id == run_id)
    if failed_only:
        q = q.where((EvalResult.passed.is_(False)) | (EvalResult.status == "error"))
    if needs_review:
        q = q.where(EvalResult.needs_human_review.is_(True))
    rows = list(session.scalars(q.order_by(EvalResult.created_at).offset(offset).limit(limit)))
    ovs = service.latest_overrides(session, [r.id for r in rows])
    return [service.result_out(r, ovs.get(r.id)) for r in rows]


@app.get("/runs/{run_id}/report", tags=["runs"])
def runs_report(run_id: str, session: DB) -> dict[str, Any]:
    return reports.run_report(session, _get_run(session, run_id))


# ----------------------------------------------------------------------------- gates

@app.post("/gates/check", response_model=GateResponse, tags=["gates"])
def gates_check(req: GateRequest, session: DB):
    """Release gate for CI: returns passed=false when quality thresholds or regression limits are missed."""
    return reports.check_gate(session, _get_run(session, req.run_id), req)


# ----------------------------------------------------------------------------- calibration

@app.get("/calibration/examples", tags=["calibration"])
def calibration_examples(session: DB, rubric: str | None = None) -> list[dict[str, Any]]:
    q = select(CalibrationExample).order_by(CalibrationExample.id)
    if rubric:
        q = q.where(CalibrationExample.rubric_name == rubric)
    return [{"id": e.id, "rubric": e.rubric_name, "quality_tier": e.quality_tier, **e.request,
             "human_scores": e.human_scores, "human_pass": e.human_pass, "notes": e.notes}
            for e in session.scalars(q)]


@app.post("/calibration/runs", response_model=CalibrationRunOut, status_code=202, tags=["calibration"])
def calibration_start(req: CalibrationRequest, session: DB):
    """Score the human-labelled calibration set with a judge and compute agreement metrics."""
    _get_rubric_or_404(session, req.rubric)
    try:
        run = start_calibration(session, req.rubric, req.judge)
    except (LookupError, ValueError) as e:
        raise HTTPException(422, str(e)) from e
    dispatch_calibration(run.id)
    session.refresh(run)
    return CalibrationRunOut.model_validate(run, from_attributes=True)


@app.get("/calibration/runs", response_model=list[CalibrationRunOut], tags=["calibration"])
def calibration_list(session: DB, rubric: str | None = None, include_examples: bool = False):
    q = select(CalibrationRun).order_by(CalibrationRun.created_at.desc())
    if rubric:
        q = q.where(CalibrationRun.rubric_name == rubric)
    out = []
    for r in session.scalars(q):
        o = CalibrationRunOut.model_validate(r, from_attributes=True)
        if not include_examples:
            o.per_example = None
        out.append(o)
    return out


@app.get("/calibration/runs/{cal_id}", response_model=CalibrationRunOut, tags=["calibration"])
def calibration_get(cal_id: str, session: DB):
    r = session.get(CalibrationRun, cal_id)
    if r is None:
        raise HTTPException(404, "calibration run not found")
    return CalibrationRunOut.model_validate(r, from_attributes=True)


@app.get("/calibration/drift", tags=["calibration"])
def calibration_drift(session: DB, a: str | None = None, b: str | None = None,
                      rubric: str | None = None) -> dict[str, Any]:
    """Compare two calibration runs (a -> b). With only `rubric`, compares its two most recent finished runs."""
    if a and b:
        ra, rb = session.get(CalibrationRun, a), session.get(CalibrationRun, b)
    elif rubric:
        latest = list(session.scalars(select(CalibrationRun).where(CalibrationRun.rubric_name == rubric,
                                                                    CalibrationRun.status == "done")
                                      .order_by(CalibrationRun.created_at.desc()).limit(2)))
        if len(latest) < 2:
            raise HTTPException(404, "need two finished calibration runs for this rubric")
        rb, ra = latest
    else:
        raise HTTPException(422, "pass a and b, or rubric")
    if ra is None or rb is None:
        raise HTTPException(404, "calibration run not found")
    return compare_calibration_runs(ra, rb)


# ----------------------------------------------------------------------------- consistency

@app.post("/consistency/runs", response_model=ConsistencyRunOut, status_code=202, tags=["consistency"])
def consistency_start(req: ConsistencyRequest, session: DB):
    """Check whether the judge agrees with itself: repeat scoring, passage-order and prompt-injection probes.
    Bypasses the judge cache, so with a paid judge this costs items x dimensions x (repeats + checks) calls."""
    _get_rubric_or_404(session, req.rubric)
    if req.source != "calibration":
        _get_run(session, req.source)
    try:
        run = start_consistency(session, req.rubric, req.judge, source=req.source, repeats=req.repeats,
                                checks=list(req.checks), max_items=req.max_items)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    dispatch_consistency(run.id)
    session.refresh(run)
    return ConsistencyRunOut.model_validate(run, from_attributes=True)


@app.get("/consistency/runs", response_model=list[ConsistencyRunOut], tags=["consistency"])
def consistency_list(session: DB, rubric: str | None = None):
    q = select(ConsistencyRun).order_by(ConsistencyRun.created_at.desc())
    if rubric:
        q = q.where(ConsistencyRun.rubric_name == rubric)
    out = []
    for r in session.scalars(q):
        o = ConsistencyRunOut.model_validate(r, from_attributes=True)
        o.per_item = None
        out.append(o)
    return out


@app.get("/consistency/runs/{run_id}", response_model=ConsistencyRunOut, tags=["consistency"])
def consistency_get(run_id: str, session: DB):
    r = session.get(ConsistencyRun, run_id)
    if r is None:
        raise HTTPException(404, "consistency run not found")
    return ConsistencyRunOut.model_validate(r, from_attributes=True)


# ----------------------------------------------------------------------------- metric health

@app.get("/metrics/health", tags=["analytics"])
def metrics_health(session: DB, rubric: str | None = None, feature: str | None = None) -> dict[str, Any]:
    return reports.metric_health(session, rubric, feature)
