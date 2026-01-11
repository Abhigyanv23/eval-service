"""Turn per-dimension judge verdicts into one score object with pass/fail, evidence and policy warnings."""
from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from evalsvc.config import settings
from evalsvc.judges import Judge, JudgeError, JudgeOutputError
from evalsvc.judges.base import JudgeResponse, Usage, as_response
from evalsvc.pricing import cost_usd
from evalsvc.schemas import DimensionScore, EvalItem, EvidenceSpan, RubricDimension, RubricSpec

_pool = ThreadPoolExecutor(max_workers=max(settings.judge_concurrency, 1) * 4, thread_name_prefix="judge")


@dataclass
class Scored:
    """Engine output; the API layer adds ids and persistence."""

    status: str  # ok | error
    overall_score: float | None
    passed: bool | None
    dimension_scores: list[DimensionScore]
    explanation: str
    confidence: float | None
    policy_warnings: list[str]
    needs_human_review: bool
    latency_ms: int
    error: str | None = None
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = 0.0

    @property
    def normalized_score(self) -> float | None:
        return None if self.overall_score is None else round((self.overall_score - 1) / 4, 4)


def judge_with_retries(judge: Judge, rubric: RubricSpec, dim: RubricDimension, item: EvalItem,
                       max_retries: int | None = None) -> JudgeResponse:
    """Malformed output is retried; API errors and refusals are not (the SDKs already retry transient errors).
    Failed attempts are added to the call count (their tokens aren't reported by the SDK on a parse failure)."""
    retries = settings.judge_max_retries if max_retries is None else max_retries
    last: JudgeOutputError | None = None
    for attempt in range(retries + 1):
        try:
            resp = as_response(judge.judge_dimension(rubric, dim, item))
            resp.usage.calls += attempt
            return resp
        except JudgeOutputError as e:
            last = e
    raise JudgeError(f"{dim.name}: no valid verdict after {retries + 1} attempts ({last})")


def _maybe_cached(judge: Judge, use_cache: bool) -> Judge:
    if use_cache and settings.judge_cache and judge.cacheable:
        from evalsvc.judges.cache import CachedJudge

        return CachedJudge(judge)
    return judge


def _cost(judge: Judge, usage: Usage) -> float | None:
    return cost_usd(judge.id, judge.model, usage.input_tokens, usage.output_tokens)


_ws = re.compile(r"\s+")


def ground_quote(quote: str, output: str) -> tuple[int | None, int | None, bool]:
    """Locate an evidence quote in the output. Exact match first, then whitespace/case-insensitive."""
    q = quote.strip().strip('"').strip()
    if not q:
        return None, None, False
    i = output.find(q)
    if i >= 0:
        return i, i + len(q), True
    # Normalized search: map positions in the collapsed lower-case text back to the original.
    norm_chars, index_map, prev_space = [], [], False
    for idx, ch in enumerate(output):
        if ch.isspace():
            if prev_space:
                continue
            norm_chars.append(" ")
            prev_space = True
        else:
            norm_chars.append(ch.lower())
            prev_space = False
        index_map.append(idx)
    j = "".join(norm_chars).find(_ws.sub(" ", q).lower())
    if j >= 0:
        nq = len(_ws.sub(" ", q))
        return index_map[j], index_map[j + nq - 1] + 1, True
    return None, None, False


def missing_inputs(dim: RubricDimension, item: EvalItem) -> list[str]:
    """Inputs a dimension cannot be judged without (it has no other input to fall back on)."""
    if dim.uses == ["context"] and not item.context:
        return ["context"]
    if dim.uses == ["reference"] and not item.reference:
        return ["reference"]
    return []


def evaluate(item: EvalItem, rubric: RubricSpec, judge: Judge, secondary: Judge | None = None,
             use_cache: bool = True) -> Scored:
    t0 = time.perf_counter()
    j1 = _maybe_cached(judge, use_cache)
    j2 = _maybe_cached(secondary, use_cache) if secondary else None
    futures = {d.name: _pool.submit(judge_with_retries, j1, rubric, d, item) for d in rubric.dimensions}
    sec_futures = ({d.name: _pool.submit(judge_with_retries, j2, rubric, d, item) for d in rubric.dimensions}
                   if j2 else {})

    responses: dict[str, JudgeResponse] = {}
    errors: list[str] = []
    for name, f in futures.items():
        try:
            responses[name] = f.result()
        except JudgeError as e:
            errors.append(str(e))
        except Exception as e:  # noqa: BLE001 - one bad dimension must not crash a whole batch
            errors.append(f"{name}: unexpected judge failure: {type(e).__name__}: {e}")

    sec_scores: dict[str, int] = {}
    sec_usage = Usage()
    for name, f in sec_futures.items():
        try:
            r = f.result()
            sec_scores[name] = r.verdict.score
            sec_usage = sec_usage + r.usage
        except Exception:  # noqa: BLE001 - the secondary judge is advisory only
            pass

    usage = sum((r.usage for r in responses.values()), Usage())
    costs = [_cost(judge, usage)] + ([_cost(secondary, sec_usage)] if secondary else [])
    total_cost = None if any(c is None for c in costs) else round(sum(costs), 6)
    usage = usage + sec_usage

    latency = int((time.perf_counter() - t0) * 1000)
    if errors:
        return Scored(status="error", overall_score=None, passed=None, dimension_scores=[],
                      explanation="Scoring failed; no score was produced.", confidence=None,
                      policy_warnings=["judge_error"], needs_human_review=rubric.high_risk,
                      latency_ms=latency, error="; ".join(errors), usage=usage, cost_usd=total_cost)

    warnings: list[str] = []
    dims: list[DimensionScore] = []
    for d in rubric.dimensions:
        v = responses[d.name].verdict
        spans = []
        for ev in v.evidence:
            start, end, grounded = ground_quote(ev.quote, item.output)
            spans.append(EvidenceSpan(quote=ev.quote, issue=ev.issue, start=start, end=end, grounded=grounded))
        conf = v.confidence
        if not any(s.grounded for s in spans):
            warnings.append(f"ungrounded_evidence:{d.name}")
            conf = min(conf, 0.5)  # a judge that can't point at the output is less trustworthy
        for m in missing_inputs(d, item):
            warnings.append(f"missing_input:{d.name}:{m}")
            conf = min(conf, 0.4)
        dim_pass = v.score >= (d.min_score or 3)
        if d.min_score and v.score < d.min_score:
            warnings.append(f"critical_dimension_failed:{d.name}")
        sec = sec_scores.get(d.name)
        if sec is not None and abs(sec - v.score) >= settings.disagreement_threshold:
            warnings.append(f"judge_disagreement:{d.name}")
        dims.append(DimensionScore(dimension=d.name, score=v.score, weight=d.weight, min_score=d.min_score,
                                   passed=dim_pass, reasoning=v.reasoning, evidence=spans,
                                   confidence=round(conf, 3), judge=judge.id, secondary_score=sec))

    total_w = sum(d.weight for d in dims)
    overall = round(sum(d.score * d.weight for d in dims) / total_w, 3)
    critical_ok = all(d.min_score is None or d.score >= d.min_score for d in dims)
    passed = overall >= rubric.pass_threshold and critical_ok
    confidence = round(min(d.confidence for d in dims), 3)

    if abs(overall - rubric.pass_threshold) <= settings.borderline_margin:
        warnings.append("borderline_score")
    if confidence < settings.low_confidence_threshold:
        warnings.append("low_judge_confidence")
    needs_review = (
        any(w.startswith("judge_disagreement") for w in warnings)
        or (rubric.high_risk and (not passed or confidence < settings.low_confidence_threshold))
        or ("borderline_score" in warnings and confidence < settings.low_confidence_threshold)
    )

    return Scored(status="ok", overall_score=overall, passed=passed, dimension_scores=dims,
                  explanation=_explain(rubric, dims, overall, passed, critical_ok), confidence=confidence,
                  policy_warnings=warnings, needs_human_review=needs_review, latency_ms=latency,
                  usage=usage, cost_usd=total_cost)


def _explain(rubric: RubricSpec, dims: list[DimensionScore], overall: float, passed: bool, critical_ok: bool) -> str:
    verdict = "PASS" if passed else "FAIL"
    head = f"{verdict}: overall {overall:.2f}/5 (threshold {rubric.pass_threshold})."
    if not critical_ok:
        failed = [f"{d.dimension} {d.score}/5 (min {d.min_score})" for d in dims
                  if d.min_score and d.score < d.min_score]
        head += " Critical dimension below minimum: " + ", ".join(failed) + "."
    weakest = min(dims, key=lambda d: (d.score, -d.weight))
    return f"{head} Weakest: {weakest.dimension} {weakest.score}/5 - {weakest.reasoning}"
