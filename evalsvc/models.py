"""ORM tables: rubrics, eval runs, per-item results, human overrides, calibration sets and runs."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from evalsvc.db import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Rubric(Base):
    __tablename__ = "rubrics"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    version: Mapped[int] = mapped_column(Integer)
    task_type: Mapped[str] = mapped_column(String(50))
    definition: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200))
    feature: Mapped[str] = mapped_column(String(100), index=True)
    rubric_name: Mapped[str] = mapped_column(String(100))
    rubric_version: Mapped[int] = mapped_column(Integer)
    judge: Mapped[str] = mapped_column(String(100))
    secondary_judge: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | running | done
    total: Mapped[int] = mapped_column(Integer, default=0)
    git_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    run_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    gate_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    gate_detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class EvalResult(Base):
    __tablename__ = "eval_results"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("eval_runs.id"), nullable=True, index=True)
    item_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    request: Mapped[dict[str, Any]] = mapped_column(JSON)
    rubric_name: Mapped[str] = mapped_column(String(100), index=True)
    rubric_version: Mapped[int] = mapped_column(Integer)
    judge: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | ok | error
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    needs_human_review: Mapped[bool] = mapped_column(Boolean, default=False)
    dimension_scores: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    policy_warnings: Mapped[list[str]] = mapped_column(JSON, default=list)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    judge_calls: Mapped[int] = mapped_column(Integer, default=0)
    cached_calls: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)  # None = judge model has no known price
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class HumanOverride(Base):
    __tablename__ = "human_overrides"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    result_id: Mapped[str] = mapped_column(ForeignKey("eval_results.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(100))
    passed: Mapped[bool] = mapped_column(Boolean)
    overall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class CalibrationExample(Base):
    __tablename__ = "calibration_examples"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    rubric_name: Mapped[str] = mapped_column(String(100), index=True)
    quality_tier: Mapped[str] = mapped_column(String(20))  # excellent | mediocre | bad
    request: Mapped[dict[str, Any]] = mapped_column(JSON)
    human_scores: Mapped[dict[str, int]] = mapped_column(JSON)
    human_pass: Mapped[bool] = mapped_column(Boolean)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)


class CalibrationRun(Base):
    __tablename__ = "calibration_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    rubric_name: Mapped[str] = mapped_column(String(100), index=True)
    rubric_version: Mapped[int] = mapped_column(Integer)
    judge: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | done | error
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    per_example: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


class JudgeCache(Base):
    """Verdicts keyed by a hash of everything that determines them: judge, prompt template, rubric dimension
    definition and the item. Identical requests (re-running an unchanged golden set) cost nothing."""

    __tablename__ = "judge_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    judge: Mapped[str] = mapped_column(String(100), index=True)
    verdict: Mapped[dict[str, Any]] = mapped_column(JSON)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ConsistencyRun(Base):
    """Judge consistency check: re-score items several times (and under meaning-preserving perturbations)
    and measure how much the judge's scores move."""

    __tablename__ = "consistency_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    rubric_name: Mapped[str] = mapped_column(String(100), index=True)
    rubric_version: Mapped[int] = mapped_column(Integer)
    judge: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending | done | error
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    per_item: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
