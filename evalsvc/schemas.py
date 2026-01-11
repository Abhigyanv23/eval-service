"""Public API contract: eval requests, score objects, rubrics, batch runs, reports and gates."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# --------------------------------------------------------------------------- rubrics

InputName = Literal["prompt", "reference", "context"]


class DimensionExample(BaseModel):
    """A short worked example shown to the judge to anchor the scale."""

    output: str
    score: int = Field(ge=1, le=5)
    why: str


class RubricDimension(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str
    weight: float = Field(default=1.0, gt=0)
    anchors: dict[int, str] = Field(description="What a 1..5 score means for this dimension.")
    min_score: int | None = Field(
        default=None, ge=1, le=5,
        description="Critical dimension: the item fails if this dimension scores below min_score, "
        "whatever the overall score.",
    )
    uses: list[InputName] = Field(default_factory=lambda: ["prompt"])
    examples: list[DimensionExample] = Field(default_factory=list)

    @field_validator("anchors")
    @classmethod
    def _anchors_cover_scale(cls, v: dict[int, str]) -> dict[int, str]:
        missing = {1, 2, 3, 4, 5} - set(v)
        if missing:
            raise ValueError(f"anchors must define scores 1-5; missing {sorted(missing)}")
        return v


class RubricSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    version: int = 1
    task_type: str
    description: str
    pass_threshold: float = Field(default=3.5, ge=1, le=5)
    high_risk: bool = Field(default=False, description="Failures and low-confidence scores need human review.")
    dimensions: list[RubricDimension] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_dims(self) -> RubricSpec:
        names = [d.name for d in self.dimensions]
        if len(names) != len(set(names)):
            raise ValueError("dimension names must be unique")
        return self

    def dimension(self, name: str) -> RubricDimension:
        return next(d for d in self.dimensions if d.name == name)


# --------------------------------------------------------------------------- requests


class EvalItem(BaseModel):
    """One model output to score."""

    item_id: str | None = None
    prompt: str = Field(description="The input the model under test received (user question, source doc, ...).")
    output: str = Field(description="The model output being evaluated.")
    reference: str | None = Field(default=None, description="Optional gold answer / label / reference summary.")
    context: list[str] | None = Field(default=None, description="Retrieved passages, for RAG faithfulness.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalRequest(EvalItem):
    rubric: str | RubricSpec = Field(description="Stored rubric name, or an inline rubric definition.")
    rubric_version: int | None = Field(default=None, description="Defaults to the latest stored version.")
    task_type: str | None = None
    judge: str | None = Field(default=None, description="e.g. 'anthropic', 'anthropic:claude-sonnet-5-5', "
                              "'openai', 'local:llama3.1', 'heuristic'. Defaults to the service default.")
    secondary_judge: str | None = Field(default=None, description="Optional second judge for disagreement tracking.")


class BatchRequest(BaseModel):
    name: str
    feature: str = Field(description="Which AI feature these outputs came from, e.g. 'support-rag'.")
    rubric: str
    rubric_version: int | None = None
    judge: str | None = None
    secondary_judge: str | None = None
    git_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    items: list[EvalItem] = Field(min_length=1, max_length=5000)


# --------------------------------------------------------------------------- judge output

class JudgeEvidence(BaseModel):
    quote: str = Field(description="A verbatim excerpt copied from the OUTPUT being evaluated.")
    issue: str = Field(description="How this excerpt raised or lowered the score.")


class JudgeVerdict(BaseModel):
    """The structured object every judge must return for one rubric dimension."""

    reasoning: str = Field(description="Step-by-step justification against the rubric anchors.")
    evidence: list[JudgeEvidence] = Field(description="At least one verbatim quote from the output that drove the score.")
    score: int = Field(description="Integer score from 1 (worst) to 5 (best).")
    confidence: float = Field(description="How sure you are of this score, from 0.0 to 1.0.")

    @field_validator("score")
    @classmethod
    def _score_range(cls, v: int) -> int:
        if not 1 <= v <= 5:
            raise ValueError("score must be between 1 and 5")
        return v

    @field_validator("confidence")
    @classmethod
    def _conf_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return v

    @field_validator("evidence")
    @classmethod
    def _evidence_required(cls, v: list[JudgeEvidence]) -> list[JudgeEvidence]:
        if not v:
            raise ValueError("at least one evidence quote is required")
        return v


# --------------------------------------------------------------------------- score output


class EvidenceSpan(BaseModel):
    quote: str
    issue: str
    start: int | None = Field(default=None, description="Character offset of the quote in the output, if found.")
    end: int | None = None
    grounded: bool = Field(description="True when the quote was found in the evaluated output.")


class DimensionScore(BaseModel):
    dimension: str
    score: int
    weight: float
    min_score: int | None
    passed: bool
    reasoning: str
    evidence: list[EvidenceSpan]
    confidence: float
    judge: str
    secondary_score: int | None = None


class EvalResultOut(BaseModel):
    id: str
    run_id: str | None = None
    item_id: str | None = None
    rubric: str
    rubric_version: int
    judge: str
    status: Literal["pending", "ok", "error"]
    overall_score: float | None = Field(default=None, description="Weighted mean of dimension scores, 1-5.")
    normalized_score: float | None = Field(default=None, description="overall_score mapped to 0-1.")
    passed: bool | None = None
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    explanation: str | None = None
    confidence: float | None = None
    policy_warnings: list[str] = Field(default_factory=list)
    needs_human_review: bool = False
    latency_ms: int | None = None
    error: str | None = None
    usage: dict[str, int] = Field(default_factory=dict, description="input_tokens, output_tokens, judge_calls, cached_calls")
    cost_usd: float | None = Field(default=None, description="Estimated judge cost. None = judge model has no known price.")
    human_override: dict[str, Any] | None = None


class BatchAccepted(BaseModel):
    run_id: str
    status: str
    total: int
    status_url: str
    report_url: str


class RunOut(BaseModel):
    id: str
    name: str
    feature: str
    rubric_name: str
    rubric_version: int
    judge: str
    secondary_judge: str | None
    status: str
    total: int
    completed: int
    git_sha: str | None
    gate_passed: bool | None
    created_at: datetime
    finished_at: datetime | None
    pass_rate: float | None = None
    mean_score: float | None = None
    cost_usd: float | None = None


# --------------------------------------------------------------------------- overrides


class OverrideRequest(BaseModel):
    reviewer: str
    passed: bool
    overall_score: float | None = Field(default=None, ge=1, le=5)
    note: str | None = None


# --------------------------------------------------------------------------- gates


class GateThresholds(BaseModel):
    min_pass_rate: float | None = Field(default=0.8, ge=0, le=1)
    use_lower_confidence_bound: bool = Field(
        default=False, description="Compare the 95% Wilson lower bound of the pass rate, not the point estimate."
    )
    min_dimension_means: dict[str, float] = Field(
        default_factory=dict, description="e.g. {'safety': 4.5, 'faithfulness': 4.0}"
    )
    max_error_rate: float = Field(default=0.05, ge=0, le=1)
    max_pass_rate_drop: float | None = Field(default=None, description="Allowed drop vs the baseline run.")
    max_dimension_drop: float | None = Field(default=None, description="Allowed drop in any dimension mean vs baseline.")
    max_cost_per_item_usd: float | None = Field(default=None, description="Fail if mean judge cost per item exceeds this.")


class GateRequest(BaseModel):
    run_id: str
    baseline_run_id: str | None = Field(
        default=None, description="Run to compare against. 'previous' = latest finished run of the same feature."
    )
    thresholds: GateThresholds = Field(default_factory=GateThresholds)


class GateCheck(BaseModel):
    name: str
    passed: bool
    actual: float | None
    threshold: float | None
    detail: str


class GateResponse(BaseModel):
    run_id: str
    baseline_run_id: str | None
    passed: bool
    checks: list[GateCheck]
    summary: str


# --------------------------------------------------------------------------- calibration


class CalibrationRequest(BaseModel):
    rubric: str
    judge: str | None = None


class ConsistencyRequest(BaseModel):
    rubric: str
    judge: str | None = None
    source: str = Field(default="calibration", description="'calibration' (the rubric's labelled set) or an eval run id.")
    repeats: int = Field(default=3, ge=2, le=10)
    checks: list[Literal["repeat", "context_order", "injection"]] = Field(
        default_factory=lambda: ["repeat", "context_order", "injection"])
    max_items: int = Field(default=25, ge=1, le=500, description="Caps cost: calls = items x dims x (repeats + checks).")


class ConsistencyRunOut(BaseModel):
    id: str
    rubric_name: str
    rubric_version: int
    judge: str
    status: str
    created_at: datetime
    config: dict[str, Any]
    metrics: dict[str, Any] | None = None
    per_item: list[dict[str, Any]] | None = None
    error: str | None = None


class CalibrationRunOut(BaseModel):
    id: str
    rubric_name: str
    rubric_version: int
    judge: str
    status: str
    created_at: datetime
    metrics: dict[str, Any] | None = None
    per_example: list[dict[str, Any]] | None = None
    error: str | None = None
