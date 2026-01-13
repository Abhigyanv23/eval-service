"""OpenAI judge. Also serves local models (Ollama, vLLM) through their OpenAI-compatible endpoints."""
from __future__ import annotations

from pydantic import ValidationError

from evalsvc.config import settings
from evalsvc.judges.base import Judge, JudgeError, JudgeOutputError, JudgeResponse, Usage
from evalsvc.judges.prompts import SYSTEM_PROMPT, build_user_prompt
from evalsvc.schemas import EvalItem, JudgeVerdict, RubricDimension, RubricSpec


class OpenAIJudge(Judge):
    def __init__(self, model: str | None = None, *, local: bool = False) -> None:
        import openai  # optional dependency

        self._openai = openai
        if local:
            self.model = model or settings.local_model
            self.id = f"local:{self.model}"
            self.client = openai.OpenAI(base_url=settings.local_base_url, api_key="local", max_retries=2)
        else:
            self.model = model or settings.openai_model
            self.id = f"openai:{self.model}"
            self.client = openai.OpenAI(max_retries=3)  # reads OPENAI_API_KEY

    def judge_dimension(self, rubric: RubricSpec, dim: RubricDimension, item: EvalItem) -> JudgeResponse:
        openai = self._openai
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(rubric, dim, item)},
                ],
                response_format=JudgeVerdict,
            )
        except ValidationError as e:
            raise JudgeOutputError(f"invalid verdict: {e.errors()[0]['msg']}") from e
        except openai.LengthFinishReasonError as e:
            raise JudgeOutputError("judge output truncated") from e
        except openai.APIStatusError as e:
            raise JudgeError(f"openai API error {e.status_code}: {e.message}") from e
        except openai.APIConnectionError as e:
            raise JudgeError(f"openai connection error: {e}") from e

        msg = completion.choices[0].message
        if msg.refusal:
            raise JudgeError(f"judge declined to score this item: {msg.refusal}")
        if msg.parsed is None:
            raise JudgeOutputError("judge returned no parsable verdict")
        u = completion.usage
        return JudgeResponse(msg.parsed, Usage(u.prompt_tokens if u else 0, u.completion_tokens if u else 0, calls=1))
