import pytest

from evalsvc.judges import Judge, JudgeError, JudgeOutputError
from evalsvc.rubric_store import load_builtin_rubrics
from evalsvc.schemas import EvalItem, JudgeEvidence, JudgeVerdict
from evalsvc.scoring import evaluate, ground_quote, judge_with_retries

RUBRICS = {r.name: r for r in load_builtin_rubrics()}
ITEM = EvalItem(prompt="What is the warranty?", output="The kettle has a 2-year warranty.",
                context=["The K200 comes with a 2-year manufacturer's warranty."])


class FixedJudge(Judge):
    cacheable = False

    def __init__(self, scores, quote="2-year warranty", conf=0.9, ident="fixed"):
        self.scores, self.quote, self.conf, self.id = scores, quote, conf, ident

    def judge_dimension(self, rubric, dim, item):
        return JudgeVerdict(reasoning="r", evidence=[JudgeEvidence(quote=self.quote, issue="i")],
                            score=self.scores[dim.name], confidence=self.conf)


class FlakyJudge(Judge):
    id = "flaky"
    cacheable = False

    def __init__(self, fail_times):
        self.fail_times, self.calls = fail_times, 0

    def judge_dimension(self, rubric, dim, item):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise JudgeOutputError("bad json")
        return JudgeVerdict(reasoning="ok", evidence=[JudgeEvidence(quote="kettle", issue="x")], score=4, confidence=0.8)


def test_verdict_schema_rejects_bad_output():
    with pytest.raises(ValueError):
        JudgeVerdict(reasoning="x", evidence=[JudgeEvidence(quote="a", issue="b")], score=7, confidence=0.5)
    with pytest.raises(ValueError):
        JudgeVerdict(reasoning="x", evidence=[], score=3, confidence=0.5)  # evidence is mandatory
    with pytest.raises(ValueError):
        JudgeVerdict(reasoning="x", evidence=[JudgeEvidence(quote="a", issue="b")], score=3, confidence=1.5)


def test_retries_then_succeeds():
    rubric = RUBRICS["rag_faithfulness"]
    j = FlakyJudge(fail_times=2)
    r = judge_with_retries(j, rubric, rubric.dimensions[0], ITEM, max_retries=2)
    assert r.verdict.score == 4 and j.calls == 3
    assert r.usage.calls == 3  # failed attempts are counted toward cost


def test_retries_exhausted_fails_cleanly():
    rubric = RUBRICS["rag_faithfulness"]
    with pytest.raises(JudgeError):
        judge_with_retries(FlakyJudge(fail_times=10), rubric, rubric.dimensions[0], ITEM, max_retries=1)


def test_error_result_has_no_score():
    s = evaluate(ITEM, RUBRICS["rag_faithfulness"], FlakyJudge(fail_times=100))
    assert s.status == "error" and s.overall_score is None and s.passed is None and s.error


def test_weighted_overall_and_pass():
    s = evaluate(ITEM, RUBRICS["rag_faithfulness"], FixedJudge({"faithfulness": 5, "relevance": 4, "completeness": 4}))
    # weights 2, 1, 1.5 -> (10 + 4 + 6) / 4.5
    assert s.overall_score == pytest.approx(4.444, abs=1e-3)
    assert s.passed is True and s.status == "ok"


def test_critical_dimension_fails_item_despite_high_overall():
    s = evaluate(ITEM, RUBRICS["rag_faithfulness"], FixedJudge({"faithfulness": 3, "relevance": 5, "completeness": 5}))
    assert s.overall_score >= 3.5
    assert s.passed is False
    assert "critical_dimension_failed:faithfulness" in s.policy_warnings


def test_ungrounded_evidence_is_flagged_and_lowers_confidence():
    s = evaluate(ITEM, RUBRICS["rag_faithfulness"],
                 FixedJudge({"faithfulness": 5, "relevance": 5, "completeness": 5}, quote="text that is not there"))
    assert any(w.startswith("ungrounded_evidence") for w in s.policy_warnings)
    assert s.confidence <= 0.5


def test_secondary_judge_disagreement_flags_review():
    primary = FixedJudge({"faithfulness": 5, "relevance": 5, "completeness": 5})
    secondary = FixedJudge({"faithfulness": 2, "relevance": 5, "completeness": 5}, ident="other")
    s = evaluate(ITEM, RUBRICS["rag_faithfulness"], primary, secondary)
    assert "judge_disagreement:faithfulness" in s.policy_warnings
    assert s.needs_human_review


def test_ground_quote_tolerates_whitespace_and_case():
    out = "The  Kettle has\na 2-year warranty."
    start, end, ok = ground_quote("the kettle has a 2-year", out)
    assert ok and out[start:end].lower().split() == ["the", "kettle", "has", "a", "2-year"]
    assert ground_quote("lifetime warranty", out)[2] is False


def test_all_builtin_rubrics_load():
    assert {"summarization", "rag_faithfulness", "classification", "tone", "completeness", "safety"} <= set(RUBRICS)


def test_consistency_metrics_detect_unstable_and_gameable_judge():
    from evalsvc.consistency import compute_consistency

    def s(overall, passed, faith):
        return {"status": "ok", "overall": overall, "passed": passed, "scores": {"faithfulness": faith},
                "cost_usd": 0.0, "calls": 1, "tokens": 0}

    per_item = [
        # judge flips between 3 and 5 on identical input; injection turns a fail into a pass
        {"id": "x", "repeats": [s(3, False, 3), s(5, True, 5), s(3, False, 3)],
         "variants": {"injection": s(5, True, 5)}},
        {"id": "y", "repeats": [s(4, True, 4)] * 3, "variants": {"injection": s(4, True, 4)}},
    ]
    m = compute_consistency(per_item, ["faithfulness"], ["repeat", "injection"])
    assert m["checks"]["repeat"]["verdict"] == "unstable"
    assert m["checks"]["repeat"]["flipped_items"] == ["x"]
    assert m["checks"]["repeat"]["dimensions"]["faithfulness"]["stable_rate"] == 0.5
    assert m["checks"]["injection"]["verdict"] == "vulnerable"
    assert m["checks"]["injection"]["fail_to_pass"] == ["x"]


def test_heuristic_is_invariant_to_passage_order():
    # Regression: the context_order consistency probe found completeness swinging 1 -> 5 on reordering.
    from evalsvc.judges.heuristic_judge import HeuristicJudge

    rubric = RUBRICS["rag_faithfulness"]
    ctx = ["The K200 electric kettle holds 1.7 litres and has a 2,200 W element that boils a full kettle in about 4 minutes.",
           "Features: automatic shut-off when boiled, boil-dry protection, and a keep-warm mode that holds temperature for 30 minutes.",
           "Care: wipe the exterior with a damp cloth. The K200 is not dishwasher safe. The K200 comes with a 2-year manufacturer's warranty."]
    item = EvalItem(prompt="Is the K200 kettle dishwasher safe, and what warranty does it come with?",
                    output="No, the K200 is not dishwasher safe. It comes with a 2-year manufacturer's warranty.", context=ctx)
    a = evaluate(item, rubric, HeuristicJudge())
    b = evaluate(item.model_copy(update={"context": list(reversed(ctx))}), rubric, HeuristicJudge())
    assert {d.dimension: d.score for d in a.dimension_scores} == {d.dimension: d.score for d in b.dimension_scores}
