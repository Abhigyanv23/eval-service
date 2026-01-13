import math

from evalsvc import stats


def test_weighted_kappa_perfect_and_chance():
    a = [1, 2, 3, 4, 5, 5, 4, 3]
    assert stats.weighted_kappa(a, a) == 1.0
    # Systematic one-point disagreement is penalised much less than a reversal
    near = stats.weighted_kappa(a, [min(5, x + 1) for x in a])
    reversed_ = stats.weighted_kappa(a, [6 - x for x in a])
    assert near > 0.5 > reversed_


def test_weighted_kappa_known_value():
    # Hand-checked: observed weighted disagreement 1/16, expected 17/16 -> kappa = 1 - 1/17
    k = stats.weighted_kappa([1, 3, 5], [1, 4, 5])
    assert math.isclose(k, 16 / 17, abs_tol=1e-9)


def test_wilson_interval_bounds():
    lo, hi = stats.wilson_interval(8, 10)
    assert 0.44 < lo < 0.5 and 0.94 < hi < 0.97
    assert stats.wilson_interval(0, 0) == (0.0, 0.0)


def test_binary_agreement_counts_false_passes():
    m = stats.binary_agreement([True, True, False, False], [True, False, False, True])
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["false_pass_rate"] == 0.5


def test_threshold_sweep_finds_separating_cut():
    scores = [1.5, 2.0, 2.8, 3.2, 3.9, 4.4, 4.8]
    human = [False, False, False, False, True, True, True]
    r = stats.sweep_threshold(scores, human)
    assert 3.2 < r["best_threshold"] <= 3.9
    assert r["balanced_accuracy"] == 1.0


def test_ordinal_agreement_bias_direction():
    m = stats.ordinal_agreement([5, 5, 5], [3, 3, 3])
    assert m["mean_bias"] == 2.0 and m["too_generous"] == 3 and m["too_harsh"] == 0
