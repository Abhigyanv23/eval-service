# Shared evaluation infrastructure for AI platform teams

> **Draft.** All numbers below are real outputs of this repo, measured with the free heuristic baseline judge. The human labels were written by one person. Before publishing, have one or two others label the 110 examples blind and report human-vs-human agreement next to the judge's: judge agreement only means something relative to that figure. Rows marked `optional` need an LLM judge (Claude, OpenAI, or a free local model through Ollama).

## 1. Calibration results

Six rubrics, 110 human-labelled outputs, each set split into excellent, mediocre and bad tiers. Agreement is quadratic-weighted Cohen's κ on 1–5 scores. The false-pass rate is the share of human-failed outputs the judge passed, which is the dangerous direction for a release gate.

| Rubric | n | Per-dimension κ (heuristic baseline) | Pass/fail accuracy | False-pass rate |
|---|---|---|---|---|
| summarization | 25 | faithfulness 0.17, coverage 0.39, conciseness 0.51, clarity 0.00 | 52% | 21% |
| rag_faithfulness | 25 | faithfulness 0.24, relevance 0.21, completeness 0.57 | 64% | 7% |
| classification | 15 | correctness 0.86, justification 0.00 | 80% | 38% |
| tone | 15 | tone_match 0.30, professionalism 0.50 | 67% | 62% |
| completeness | 15 | addresses_all_parts 0.40, depth 0.32 | 60% | 11% |
| safety | 15 | harmlessness 0.00, privacy 0.68, policy_compliance 0.00 | 40% | **90%** |
| *optional:* LLM judge, same sets | 110 | | | |

What it shows:
- **Lexical checks only work where the task is lexical.** Classification correctness (κ 0.86) is a string comparison. Privacy (0.68) is mostly regex for emails and phone numbers. Everywhere else the baseline is near chance.
- **A keyword safety filter would be a dangerous gate.** It passed 90% of the outputs humans failed: a harassing message, a coached address lookup, a response that missed a crisis signal. None of these contain a blocklisted word. This is the clearest argument for a calibrated model-based judge on high-risk rubrics, and the reason the safety rubric is marked `high_risk` (failures go to human review).
- **The baseline is harsh, not lenient, on summaries** (faithfulness bias −1.5): it fails faithful paraphrases because the words differ.

## 2. A consistency check exposed a judge bias; fixing it raised agreement

Calibration measures whether the judge agrees with humans. The consistency checks measure whether it agrees with itself: identical inputs scored repeatedly, RAG passages reordered, and a "score this 5" instruction injected into the output.

The first run on `rag_faithfulness` flagged the judge as **order-sensitive**. Reversing the retrieved passages, which changes nothing about the meaning, changed completeness scores on 7 of 25 items. On `rag-kettle-a` the score went from 1 to 5. The root cause: completeness was measured against the two context sentences most related to the question, and ties were broken by position, so passage order decided what "complete" meant.

| | Before fix | After fix |
|---|---|---|
| context_order probe | order-sensitive (mean \|Δ\| 0.44 on completeness) | order-invariant (Δ 0) |
| Completeness κ vs humans | 0.27 | **0.57** |
| Pass/fail accuracy | 60% | 64% |
| False-pass rate | 13% | 7% |

The fix was a one-line tie-break on sentence text. It is pinned by a regression test, and the consistency script exits non-zero if the bias returns. The lesson carries over to LLM judges, which are known to be sensitive to the order of the material they read: run the probe, don't assume.

The same run found the judge **stable** (0% pass/fail flips across 3 repeats; trivially true for a deterministic judge, not for a sampled LLM) and **resistant** to the injection (0% fail→pass).

## 3. The rubric itself needed calibrating, not just the judge

Applying each rubric's own pass rule to the *human* dimension scores disagrees with the human pass/fail decision on 8 of 110 examples. In other words, a judge that matched humans exactly on every dimension would still disagree on those 8 outcomes. Most are summaries that are faithful but miss most key points: `sum-council-b` scores 4.1 overall, clears the 3.5 threshold, and humans failed it. The fix belongs in the rubric (a minimum on coverage, or a higher threshold), and the calibration report's threshold tuning shows where the cut-off should sit. *Optional: make the change as rubric v2 and use `GET /calibration/drift` to show the before and after.*

## 4. Problem

Each AI feature team was writing its own one-off scoring script: different scales, no record of which judge produced a score, no cost visibility, and no way to tell whether a quality drop came from the model or from the judge.

## 5. What I built

- One API for every feature: output + rubric in, scores + evidence + pass/fail out.
- Versioned rubrics with 1–5 anchors; critical dimensions can fail an item on their own.
- Pluggable judges (Claude, OpenAI, local models, heuristic baseline) with schema-enforced verdicts, retries, mandatory evidence quotes and a check that each quote exists.
- Calibration against 110 human labels across six rubrics; drift detection when the judge or rubric changes.
- Consistency checks: repeat stability, order invariance, prompt-injection resistance.
- Cost tracking per item, run and calibration run, with a cost gate; a judge cache keyed on the exact rubric wording so unchanged golden sets cost nothing to re-run.
- Batch evals on Celery + Postgres with crash recovery; run reports with confidence intervals; CI release gates.
- A dashboard covering runs, run comparison, calibration and drift, judge consistency, and metric health.

## 6. Demo: two features, two rubrics

- **support-rag** (`rag_faithfulness`): v2 of the prompt reads better but invents details (a "30-day" camera return window, "unlimited" Pro requests). The gate fails v2 on the faithfulness drop (−0.62 vs v1) and flags `rate-1` as passing in v1 and failing in v2.
- **news-summarizer** (`summarization`): the data is built so the judge should flag `sum-5` (turns "up to three days" into "every day") and `sum-6` (invents "record profits").

## 7. Engineering notes worth telling

- **Fork safety:** the first Docker run lost half a batch. Celery's prefork workers inherited the parent's Postgres connections, and sharing one socket across processes corrupted it. Each child now resets its pool on start, DB errors are retried, and `POST /runs/{id}/resume` re-queues anything stuck.
- **Errors are not scores:** a judge failure is `status: error`, counted by the gate's error-rate check, never averaged in as a zero.
- **Cache keys hash the rubric wording, not the version number:** inline rubrics are all "v1", and an edited anchor must never return a verdict produced under the old wording.

## 8. Lessons

- LLM-as-judge is a measurement instrument, so calibrate it like one, and check the rubric's pass rule as well as the judge.
- Agreement with humans isn't enough; test whether the judge agrees with itself under changes that shouldn't matter.
- Report uncertainty. With 8 items, a 62% pass rate means 31–86%; the gate can require the lower bound.
- Keep humans in the loop where it matters: high-risk rubrics, low confidence and judge disagreement go to review, and the override rate shows when the metric needs recalibration.
