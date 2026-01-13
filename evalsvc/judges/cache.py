"""Judge result cache. Wraps any judge; a hit returns the stored verdict with zero tokens spent."""
from __future__ import annotations

import hashlib
import json

from sqlalchemy.exc import IntegrityError

from evalsvc.db import SessionLocal
from evalsvc.judges.base import Judge, JudgeResponse, Usage, as_response
from evalsvc.judges.prompts import SYSTEM_PROMPT
from evalsvc.models import JudgeCache
from evalsvc.schemas import EvalItem, JudgeVerdict, RubricDimension, RubricSpec

_PROMPT_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:16]


def cache_key(judge_id: str, rubric: RubricSpec, dim: RubricDimension, item: EvalItem) -> str:
    # Hash the dimension's full definition, not the rubric version: inline rubrics are all "v1",
    # and editing an anchor must never return a verdict produced under the old wording.
    payload = {
        "judge": judge_id, "system_prompt": _PROMPT_HASH, "rubric": rubric.name,
        "rubric_description": rubric.description, "dimension": dim.model_dump(mode="json"),
        "item": item.model_dump(mode="json", include={"prompt", "output", "reference", "context"}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class CachedJudge(Judge):
    def __init__(self, inner: Judge) -> None:
        self.inner = inner
        self.id = inner.id
        self.model = inner.model

    def judge_dimension(self, rubric: RubricSpec, dim: RubricDimension, item: EvalItem) -> JudgeResponse:
        key = cache_key(self.inner.id, rubric, dim, item)
        with SessionLocal() as s:
            row = s.get(JudgeCache, key)
            if row is not None:
                row.hits += 1
                s.commit()
                return JudgeResponse(JudgeVerdict.model_validate(row.verdict), Usage(cached=1))
        resp = as_response(self.inner.judge_dimension(rubric, dim, item))
        with SessionLocal() as s:
            s.add(JudgeCache(key=key, judge=self.inner.id, verdict=resp.verdict.model_dump(mode="json"),
                             input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens))
            try:
                s.commit()
            except IntegrityError:  # another worker cached the same request concurrently
                s.rollback()
        return resp
