"""Judge registry. Judge names look like 'provider' or 'provider:model'."""
from __future__ import annotations

from functools import lru_cache

from evalsvc.config import settings
from evalsvc.judges.base import Judge, JudgeError, JudgeOutputError

__all__ = ["Judge", "JudgeError", "JudgeOutputError", "get_judge"]


@lru_cache(maxsize=32)
def get_judge(name: str | None = None) -> Judge:
    name = name or settings.default_judge
    provider, _, model = name.partition(":")
    model = model or None
    if provider == "heuristic":
        from evalsvc.judges.heuristic_judge import HeuristicJudge

        return HeuristicJudge()
    if provider == "anthropic":
        from evalsvc.judges.anthropic_judge import AnthropicJudge

        return AnthropicJudge(model)
    if provider == "openai":
        from evalsvc.judges.openai_judge import OpenAIJudge

        return OpenAIJudge(model)
    if provider == "local":
        from evalsvc.judges.openai_judge import OpenAIJudge

        return OpenAIJudge(model, local=True)
    raise ValueError(f"unknown judge '{name}' (expected heuristic, anthropic[:model], openai[:model], local[:model])")
