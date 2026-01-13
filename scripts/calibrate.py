"""Run judge calibration against the human-labelled set and print an agreement report.

  python scripts/calibrate.py --rubric summarization --judge anthropic
  python scripts/calibrate.py --rubric rag_faithfulness --judge heuristic --markdown reports/cal.md
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx


def fmt(x, spec=".2f"):
    return "—" if x is None else format(x, spec)


def to_markdown(run: dict) -> str:
    m = run["metrics"]
    pf, th = m["pass_fail"], m["threshold_tuning"]
    lines = [f"## Calibration: `{run['judge']}` on `{run['rubric_name']}` v{run['rubric_version']}", "",
             f"{m['n_scored']} of {m['n_examples']} human-labelled examples scored ({m['n_errors']} judge errors).", "",
             "| Dimension | Weighted κ | Spearman | Exact | Within 1 | MAE | Bias | Too generous | Too harsh |",
             "|---|---|---|---|---|---|---|---|---|"]
    for d, v in m["dimensions"].items():
        lines.append(f"| {d} | {fmt(v['weighted_kappa'])} | {fmt(v['spearman'])} | {fmt(v['exact_agreement'], '.0%')} | "
                     f"{fmt(v['within_one'], '.0%')} | {fmt(v['mae'])} | {fmt(v['mean_bias'], '+.2f')} | "
                     f"{v['too_generous']} | {v['too_harsh']} |")
    lines += ["", f"**Pass/fail agreement:** accuracy {fmt(pf['accuracy'], '.0%')}, κ {fmt(pf['kappa'])}, "
                  f"false-pass rate {fmt(pf['false_pass_rate'], '.0%')} (judge passed what humans failed), "
                  f"false-fail rate {fmt(pf['false_fail_rate'], '.0%')}.",
              f"**Threshold tuning:** current pass threshold {m['pass_threshold']}; the cut-off that best matches "
              f"human pass/fail is {th['best_threshold']} (balanced accuracy {fmt(th['balanced_accuracy'], '.0%')}).",
              "", "| Tier | n | Human mean | Judge mean |", "|---|---|---|---|"]
    for t, v in m["by_tier"].items():
        lines.append(f"| {t} | {v['n']} | {v['human_mean']:.2f} | {v['judge_mean']:.2f} |")
    if m["largest_disagreements"]:
        lines += ["", "**Largest disagreements**", ""]
        for w in m["largest_disagreements"][:8]:
            lines.append(f"- `{w['id']}` {w['dimension']}: judge {w['judge']} vs human {w['human']} ({w['direction']}) — "
                         f"{w['judge_reasoning'][:200]}")
    return "\n".join(lines) + "\n"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    p = argparse.ArgumentParser()
    p.add_argument("--api", default=os.environ.get("EVALSVC_API_URL", "http://localhost:8000"))
    p.add_argument("--rubric", required=True)
    p.add_argument("--judge", default=None)
    p.add_argument("--markdown", default=None, help="also write the report to this .md file")
    a = p.parse_args()
    headers = {"X-API-Key": os.environ["EVALSVC_API_KEY"]} if os.environ.get("EVALSVC_API_KEY") else {}
    c = httpx.Client(base_url=a.api, headers=headers, timeout=60)

    run = c.post("/calibration/runs", json={"rubric": a.rubric, "judge": a.judge})
    run.raise_for_status()
    cal_id = run.json()["id"]
    print(f"calibration run {cal_id} started for {a.rubric} with {run.json()['judge']}")
    while (run := c.get(f"/calibration/runs/{cal_id}").json())["status"] == "pending":
        time.sleep(3)
    if run["status"] != "done":
        print(f"calibration failed: {run['error']}", file=sys.stderr)
        return 1
    md = to_markdown(run)
    print(md)
    if a.markdown:
        Path(a.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(a.markdown).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
