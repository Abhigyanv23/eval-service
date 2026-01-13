"""Run judge consistency checks (repeat stability, passage-order invariance, prompt-injection resistance).

  python scripts/consistency.py --rubric rag_faithfulness --judge heuristic
  python scripts/consistency.py --rubric summarization --judge local:qwen2.5:3b --repeats 5 --max-items 10

Checks bypass the judge cache: with a paid judge, calls = items x dimensions x (repeats + 2).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import httpx


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--api", default=os.environ.get("EVALSVC_API_URL", "http://localhost:8000"))
    p.add_argument("--rubric", required=True)
    p.add_argument("--judge", default=None)
    p.add_argument("--source", default="calibration", help="'calibration' or an eval run id")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-items", type=int, default=25)
    a = p.parse_args()
    headers = {"X-API-Key": os.environ["EVALSVC_API_KEY"]} if os.environ.get("EVALSVC_API_KEY") else {}
    c = httpx.Client(base_url=a.api, headers=headers, timeout=60)

    r = c.post("/consistency/runs", json={"rubric": a.rubric, "judge": a.judge, "source": a.source,
                                          "repeats": a.repeats, "max_items": a.max_items})
    r.raise_for_status()
    run_id = r.json()["id"]
    print(f"consistency run {run_id} for {a.rubric} with {r.json()['judge']}")
    while (run := c.get(f"/consistency/runs/{run_id}").json())["status"] == "pending":
        time.sleep(3)
    if run["status"] != "done":
        print(f"failed: {run['error']}", file=sys.stderr)
        return 1
    m = run["metrics"]
    print(f"{m['n_scored']} items, {m['cost']['judge_calls']} judge calls, cost {m['cost']['total_usd']}\n")
    for name, chk in m["checks"].items():
        print(f"[{chk['verdict']}] {name}")
        if "pass_flip_rate" in chk:
            print(f"    pass/fail flip rate {chk['pass_flip_rate']:.0%}  flipped: {chk['flipped_items'] or '-'}")
        if name == "injection" and chk.get("n"):
            print(f"    fail->pass {chk['attack_success_rate']:.0%}, any score inflated {chk['score_inflation_rate']:.0%}")
        for d, v in (chk.get("dimensions") or {}).items():
            print(f"    {d:<20} " + "  ".join(f"{k}={v[k]}" for k in v))
    bad = [n for n, chk in m["checks"].items() if chk["verdict"] in ("unstable", "order-sensitive", "vulnerable")]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
