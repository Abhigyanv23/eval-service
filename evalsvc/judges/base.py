from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from evalsvc.schemas import EvalItem, JudgeVerdict, RubricDimension, RubricSpec


class JudgeError(Exception):
    """The judge could not produce a score (API failure, refusal, or invalid output after retries)."""


class JudgeOutputError(JudgeError):
    """The judge answered, but the answer did not match the JudgeVerdict schema. Retryable."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0          # API calls made, including retries of malformed output
    cached: int = 0         # dimension verdicts served from the judge cache

    def __add__(self, other: Usage) -> Usage:
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens,
                     self.calls + other.calls, self.cached + other.cached)


@dataclass
class JudgeResponse:
    verdict: JudgeVerdict
    usage: Usage


class Judge(ABC):
    #: Stable identifier recorded with every score, including the model, e.g. "anthropic:claude-opus-5-5".
    #: Changing it is what triggers a drift check against the calibration set.
    id: str
    #: Model name used for pricing; None for judges that make no API calls.
    model: str | None = None
    #: Whether verdicts are worth caching (paid or slow judges). Deterministic local heuristics aren't.
    cacheable: bool = True

    @abstractmethod
    def judge_dimension(self, rubric: RubricSpec, dim: RubricDimension,
                        item: EvalItem) -> JudgeResponse | JudgeVerdict:
        """Score one rubric dimension. Return the verdict with token usage (a bare JudgeVerdict counts as
        zero tokens). Raise JudgeOutputError for malformed output, JudgeError otherwise."""


def as_response(result: JudgeResponse | JudgeVerdict) -> JudgeResponse:
    return result if isinstance(result, JudgeResponse) else JudgeResponse(result, Usage(calls=1))
