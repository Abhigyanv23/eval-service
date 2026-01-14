"""Release gate for CI. Submits a dataset of model outputs, waits for scoring, checks thresholds.
Exits 1 when the gate fails, so a pipeline step fails with it.

Example:
  python scripts/ci_gate.py --dataset data/demo/support_rag_v2.jsonl --feature support-rag \
      --rubric rag_faithfulness --min-pass-rate 0.8 --min-dim faithfulness=4.0 \
      --baseline previous --max-dimension-drop 0.2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default=os.environ.get("EVALSVC_API_URL", "http://localhost:8000"))
    p.add_argument("--dataset", required=True, help="JSONL of items: prompt, output, [reference], [context], [item_id]")
    p.add_argument("--feature", required=True)
    p.add_argument("--rubric", required=True)
    p.add_argument("--judge", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--git-sha", default=os.environ.get("GITHUB_SHA"))
    p.add_argument("--min-pass-rate", type=float, default=0.8)
    p.add_argument("--lower-bound", action="store_true", help="gate on the 95%% lower bound of the pass rate")
    p.add_argument("--min-dim", action="append", default=[], metavar="DIM=MEAN")
    p.add_argument("--max-error-rate", type=float, default=0.05)
    p.add_argument("--baseline", default=None, help="baseline run id, or 'previous'")
    p.add_argument("--max-pass-rate-drop", type=float, default=None)
    p.add_argument("--max-dimension-drop", type=float, default=None)
    p.add_argument("--max-cost-per-item", type=float, default=None, help="fail if mean judge cost per item (USD) exceeds this")
    p.add_argument("--timeout", type=int, default=1800, help="seconds to wait for scoring")
    p.add_argument("--report-out", default=None, help="write the gate result + report as JSON here")
    a = p.parse_args()

    headers = {"X-API-Key": os.environ["EVALSVC_API_KEY"]} if os.environ.get("EVALSVC_API_KEY") else {}
    client = httpx.Client(base_url=a.api, headers=headers, timeout=60)
    items = [json.loads(line) for line in Path(a.dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
    name = a.name or f"ci {a.feature} {(a.git_sha or 'local')[:8]}"

    r = client.post("/evals/batch", json={"name": name, "feature": a.feature, "rubric": a.rubric, "judge": a.judge,
                                          "git_sha": a.git_sha, "items": items})
    r.raise_for_status()
    run_id = r.json()["run_id"]
    print(f"submitted run {run_id} ({len(items)} items)")

    deadline = time.time() + a.timeout
    while True:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] == "done":
            break
        if time.time() > deadline:
            print(f"timed out waiting for run {run_id} ({run['completed']}/{run['total']})", file=sys.stderr)
            return 2
        time.sleep(2)

    thresholds = {"min_pass_rate": a.min_pass_rate, "use_lower_confidence_bound": a.lower_bound,
                  "min_dimension_means": {k: float(v) for k, v in (x.split("=", 1) for x in a.min_dim)},
                  "max_error_rate": a.max_error_rate, "max_pass_rate_drop": a.max_pass_rate_drop,
                  "max_dimension_drop": a.max_dimension_drop, "max_cost_per_item_usd": a.max_cost_per_item}
    gate = client.post("/gates/check", json={"run_id": run_id, "baseline_run_id": a.baseline,
                                             "thresholds": thresholds}).json()
    report = client.get(f"/runs/{run_id}/report").json()

    print(f"\n{gate['summary']}  (run {run_id}, pass rate {report['pass_rate']}, mean {report['overall']['mean']}, "
          f"judge cost ${report['cost']['total_usd']})")
    for c in gate["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['name']:<28} actual={c['actual']}  threshold={c['threshold']}  {c['detail']}")
    if report["failed_cases"]:
        print("\nLowest-scoring failures:")
        for fc in report["failed_cases"][:5]:
            print(f"  - {fc['item_id']}: {fc['explanation']}")
    if a.report_out:
        Path(a.report_out).write_text(json.dumps({"gate": gate, "report": report}, indent=2), encoding="utf-8")
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
