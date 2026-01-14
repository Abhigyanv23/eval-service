from evalsvc.db import init_db
from evalsvc.judges import Judge
from evalsvc.judges.base import JudgeResponse, Usage
from evalsvc.pricing import cost_usd
from evalsvc.rubric_store import load_builtin_rubrics
from evalsvc.schemas import EvalItem, JudgeEvidence, JudgeVerdict
from evalsvc.scoring import evaluate

RUBRIC = {r.name: r for r in load_builtin_rubrics()}["rag_faithfulness"]


class PaidJudge(Judge):
    """Pretends to be Claude Opus 5.5: 1,000 input + 200 output tokens per dimension call."""

    model = "claude-opus-5-5"

    def __init__(self, ident="anthropic:claude-opus-5-5-test"):
        self.id, self.calls = ident, 0

    def judge_dimension(self, rubric, dim, item):
        self.calls += 1
        v = JudgeVerdict(reasoning="r", evidence=[JudgeEvidence(quote="warranty", issue="i")], score=4, confidence=0.9)
        return JudgeResponse(v, Usage(input_tokens=1000, output_tokens=200, calls=1))


def _item(output="The warranty is 2 years."):
    return EvalItem(prompt="What is the warranty?", output=output, context=["The warranty is 2 years."])


def test_pricing():
    assert cost_usd("anthropic:claude-opus-5-5", "claude-opus-5-5", 1_000_000, 0) == 4.0
    assert cost_usd("heuristic:v1", None, 10, 10) == 0.0
    assert cost_usd("local:llama3.1", "llama3.1", 10, 10) == 0.0
    assert cost_usd("openai:some-unpriced-model", "some-unpriced-model", 10, 10) is None


def test_cost_is_summed_across_dimensions():
    init_db()
    s = evaluate(_item(), RUBRIC, PaidJudge("paid-a"), use_cache=False)
    assert s.usage.calls == 3 and s.usage.input_tokens == 3000 and s.usage.output_tokens == 600
    # 3 x (1000 * $4 + 200 * $20) / 1M
    assert abs(s.cost_usd - 0.024) < 1e-9


def test_cache_hit_costs_nothing_and_misses_when_output_changes():
    init_db()
    j = PaidJudge("paid-b")
    first = evaluate(_item(), RUBRIC, j)
    second = evaluate(_item(), RUBRIC, j)
    assert j.calls == 3                         # second evaluation never reached the judge
    assert second.usage.cached == 3 and second.cost_usd == 0.0
    assert first.overall_score == second.overall_score
    evaluate(_item("The warranty is two years."), RUBRIC, j)
    assert j.calls == 6                         # different output -> different cache key


def test_cache_misses_when_rubric_anchor_changes():
    init_db()
    j = PaidJudge("paid-c")
    evaluate(_item(), RUBRIC, j)
    edited = RUBRIC.model_copy(deep=True)
    edited.dimensions[0].anchors[5] = "Every claim is supported, quoted or closely paraphrased."
    evaluate(_item(), edited, j)
    assert j.calls == 4                         # only the edited dimension was re-judged
