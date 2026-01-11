"""Agreement metrics and confidence intervals (numpy/scipy only)."""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats as sps


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Behaves well at small n and near 0/1."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                      seed: int = 0) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def weighted_kappa(a: Sequence[int], b: Sequence[int], categories: Sequence[int] = (1, 2, 3, 4, 5),
                   weights: str = "quadratic") -> float:
    """Cohen's kappa with linear or quadratic weights. Standard agreement metric for ordinal 1-5 scores:
    a 4-vs-5 disagreement costs much less than a 1-vs-5 one."""
    cats = list(categories)
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    obs = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        obs[idx[x], idx[y]] += 1
    n = obs.sum()
    if n == 0:
        return float("nan")
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    w = ((i - j) ** 2 if weights == "quadratic" else np.abs(i - j)) / (k - 1) ** (2 if weights == "quadratic" else 1)
    expected = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    denom = (w * expected).sum()
    if denom == 0:
        return 1.0  # both raters used a single identical category
    return float(1 - (w * obs).sum() / denom)


def cohen_kappa_binary(a: Sequence[bool], b: Sequence[bool]) -> float:
    a_arr, b_arr = np.asarray(a, bool), np.asarray(b, bool)
    n = a_arr.size
    if n == 0:
        return float("nan")
    po = float((a_arr == b_arr).mean())
    pa, pb = a_arr.mean(), b_arr.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else float((po - pe) / (1 - pe))


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    rho = sps.spearmanr(a, b).statistic
    return None if np.isnan(rho) else float(rho)


def ordinal_agreement(judge: Sequence[int], human: Sequence[int]) -> dict[str, float | int | None]:
    j, h = np.asarray(judge, float), np.asarray(human, float)
    diff = j - h
    return {
        "n": int(j.size),
        "exact_agreement": float((diff == 0).mean()) if j.size else None,
        "within_one": float((np.abs(diff) <= 1).mean()) if j.size else None,
        "mae": float(np.abs(diff).mean()) if j.size else None,
        "mean_bias": float(diff.mean()) if j.size else None,  # >0 judge too generous, <0 too harsh
        "weighted_kappa": weighted_kappa(list(map(int, judge)), list(map(int, human))) if j.size else None,
        "spearman": spearman(list(judge), list(human)),
        "too_generous": int((diff >= 2).sum()),
        "too_harsh": int((diff <= -2).sum()),
    }


def binary_agreement(judge_pass: Sequence[bool], human_pass: Sequence[bool]) -> dict[str, float | int | None]:
    jp, hp = np.asarray(judge_pass, bool), np.asarray(human_pass, bool)
    tp = int((jp & hp).sum()); tn = int((~jp & ~hp).sum())
    fp = int((jp & ~hp).sum()); fn = int((~jp & hp).sum())
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "n": n, "accuracy": (tp + tn) / n if n else None, "kappa": cohen_kappa_binary(jp, hp),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        # fp = judge passed something a human failed: the dangerous direction for a release gate
        "false_pass_rate": fp / (fp + tn) if fp + tn else None,
        "false_fail_rate": fn / (fn + tp) if fn + tp else None,
        "precision": precision, "recall": recall,
    }


def sweep_threshold(scores: Sequence[float], human_pass: Sequence[bool], lo: float = 1.0, hi: float = 5.0,
                    step: float = 0.05) -> dict[str, float | None]:
    """Find the overall-score threshold whose pass/fail decisions best match human labels.
    Maximises balanced accuracy, so a set with mostly-passing examples cannot be gamed by passing everything."""
    s, h = np.asarray(scores, float), np.asarray(human_pass, bool)
    if s.size == 0 or h.all() or (~h).all():
        return {"best_threshold": None, "balanced_accuracy": None}
    best_t, best_ba = None, -1.0
    for k in range(int(round((hi - lo) / step)) + 1):
        t = round(lo + k * step, 4)  # exact decimal thresholds, so the reported value reproduces the decisions
        p = s >= t
        tpr = (p & h).sum() / h.sum()
        tnr = (~p & ~h).sum() / (~h).sum()
        ba = (tpr + tnr) / 2
        if ba > best_ba + 1e-12:
            best_t, best_ba = float(round(t, 2)), float(ba)
    return {"best_threshold": best_t, "balanced_accuracy": best_ba}
