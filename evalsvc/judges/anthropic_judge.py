from __future__ import annotations

import anthropic
from pydantic import ValidationError

from evalsvc.config import settings
from evalsvc.judges.base import Judge, JudgeError, JudgeOutputError, JudgeResponse, Usage
from evalsvc.judges.prompts import SYSTEM_PROMPT, build_user_prompt
from evalsvc.schemas import EvalItem, JudgeVerdict, RubricDimension, RubricSpec


class AnthropicJudge(Judge):
    """Claude as judge, using structured outputs so every response parses into a JudgeVerdict."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.anthropic_model
        self.id = f"anthropic:{self.model}"
        # Reads ANTHROPIC_API_KEY (or an `ant auth login` profile). The SDK retries 429/5xx itself.
        self.client = anthropic.Anthropic(max_retries=3)

    def judge_dimension(self, rubric: RubricSpec, dim: RubricDimension, item: EvalItem) -> JudgeResponse:
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=settings.judge_max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(rubric, dim, item)}],
                output_format=JudgeVerdict,
            )
        except ValidationError as e:  # schema-valid JSON that failed our validators (e.g. score 7)
            raise JudgeOutputError(f"invalid verdict: {e.errors()[0]['msg']}") from e
        except anthropic.BadRequestError as e:
            raise JudgeError(f"anthropic request rejected: {e.message}") from e
        except anthropic.APIStatusError as e:
            raise JudgeError(f"anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise JudgeError(f"anthropic connection error: {e}") from e

        if resp.stop_reason == "refusal":
            raise JudgeError("judge declined to score this item")
        if resp.stop_reason == "max_tokens":
            raise JudgeOutputError("judge output truncated at max_tokens")
        if resp.parsed_output is None:
            raise JudgeOutputError("judge returned no parsable verdict")
        return JudgeResponse(resp.parsed_output,
                             Usage(resp.usage.input_tokens, resp.usage.output_tokens, calls=1))
