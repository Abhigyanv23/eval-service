"""Load, version and fetch rubrics. Built-in rubrics ship as YAML and are seeded into the DB."""
from __future__ import annotations

from pathlib import Path

import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalsvc.models import Rubric
from evalsvc.schemas import RubricSpec

BUILTIN_DIR = Path(__file__).parent / "rubrics"


class RubricNotFound(LookupError):
    pass


def load_builtin_rubrics() -> list[RubricSpec]:
    return [RubricSpec.model_validate(yaml.safe_load(p.read_text(encoding="utf-8")))
            for p in sorted(BUILTIN_DIR.glob("*.yaml"))]


def seed_rubrics(session: Session) -> int:
    """Insert built-in rubrics whose (name, version) is not stored yet. Returns the number inserted."""
    added = 0
    for spec in load_builtin_rubrics():
        exists = session.scalar(select(Rubric.id).where(Rubric.name == spec.name, Rubric.version == spec.version))
        if exists is None:
            session.add(Rubric(name=spec.name, version=spec.version, task_type=spec.task_type,
                               definition=spec.model_dump(mode="json")))
            added += 1
    session.commit()
    return added


def get_rubric(session: Session, name: str, version: int | None = None) -> RubricSpec:
    q = select(Rubric).where(Rubric.name == name)
    q = q.where(Rubric.version == version) if version is not None else q.order_by(Rubric.version.desc())
    row = session.scalars(q.limit(1)).first()
    if row is None:
        raise RubricNotFound(f"rubric '{name}'" + (f" v{version}" if version else "") + " not found")
    return RubricSpec.model_validate(row.definition)


def list_rubrics(session: Session) -> list[RubricSpec]:
    latest = (select(Rubric.name, func.max(Rubric.version).label("v")).group_by(Rubric.name).subquery())
    rows = session.scalars(select(Rubric).join(latest, (Rubric.name == latest.c.name) & (Rubric.version == latest.c.v))
                           .order_by(Rubric.name))
    return [RubricSpec.model_validate(r.definition) for r in rows]


def create_rubric_version(session: Session, spec: RubricSpec) -> RubricSpec:
    """Store a rubric as a new version (version = latest + 1). Rubrics are immutable once stored,
    so historical scores stay tied to the exact rubric that produced them."""
    current = session.scalar(select(func.max(Rubric.version)).where(Rubric.name == spec.name)) or 0
    spec = spec.model_copy(update={"version": current + 1})
    session.add(Rubric(name=spec.name, version=spec.version, task_type=spec.task_type,
                       definition=spec.model_dump(mode="json")))
    session.commit()
    return spec
