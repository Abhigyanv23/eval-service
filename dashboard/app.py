"""Streamlit dashboard for the eval service. Run from the repo root: streamlit run dashboard/app.py

Two modes:
* EVALSVC_API_URL set (docker compose, or a deployed API): reads everything through that HTTP API.
* EVALSVC_API_URL unset (e.g. Streamlit Community Cloud): embedded demo. The API runs inside this process on a
  throwaway SQLite database and is seeded once at startup with demo data scored by the offline heuristic judge.
"""
from __future__ import annotations

import html
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("EVALSVC_API_URL", "").rstrip("/")
EMBEDDED = not API
HEADERS = {"X-API-Key": os.environ["EVALSVC_API_KEY"]} if os.environ.get("EVALSVC_API_KEY") else {}

if EMBEDDED:
    # Must be set before evalsvc is imported: its settings are read at import time.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    os.environ.setdefault("EVALSVC_DATABASE_URL", f"sqlite:///{Path(tempfile.gettempdir()) / 'evalsvc_demo.db'}")
    os.environ["EVALSVC_SYNC_TASKS"] = "true"       # batches and calibrations finish inside the request
    os.environ["EVALSVC_DEFAULT_JUDGE"] = "heuristic"
    os.environ.pop("EVALSVC_REDIS_URL", None)
    os.environ.pop("EVALSVC_API_KEY", None)
    HEADERS = {}

st.set_page_config(page_title="Eval Service", page_icon=":material/fact_check:", layout="wide")

# Validated default palette. Categorical slots are assigned in fixed order; each mode has its own steps.
SERIES = {"light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
          "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]}
SEQ = {"light": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"],
       "dark": ["#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"]}
EMPTY_CELL = {"light": "#f0efec", "dark": "#1f1f1e"}
INK = {"light": "#52514e", "dark": "#c3c2b7"}

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1320px;}
  [data-testid="stMetricValue"] {font-size: 1.9rem;}
  [data-testid="stMetricLabel"] p {font-size: .85rem; opacity: .8;}
  h1 {font-size: 1.9rem !important; margin-bottom: 0 !important;}
  h3 {font-size: 1.15rem !important;}
  .ev-out {line-height: 1.65; padding: .75rem 1rem; border-radius: .5rem;
           border: 1px solid rgba(128,128,128,.25); white-space: pre-wrap;}
  .ev-out mark {background: rgba(235,104,52,.22); color: inherit; padding: 0 .15rem; border-radius: .2rem;
                border-bottom: 2px solid rgba(235,104,52,.9);}
  .muted {opacity: .7; font-size: .85rem;}
</style>""", unsafe_allow_html=True)


def mode() -> str:
    t = getattr(st.context, "theme", None)
    return "dark" if t is not None and getattr(t, "type", None) == "dark" else "light"


def series() -> list[str]:
    return SERIES[mode()]


@st.cache_resource(show_spinner="Starting the embedded eval service and scoring the demo data…")
def embedded_client():
    """In-process API. Shared by every session; seeded once per process (st.cache_resource locks the first call)."""
    from fastapi.testclient import TestClient

    from evalsvc.api import app
    from evalsvc.demo_seed import seed_demo

    client = TestClient(app)
    client.__enter__()  # runs the app lifespan: creates tables, seeds rubrics and calibration examples
    seed_demo(client)
    return client


@st.cache_data(ttl=10, show_spinner=False)
def get(path: str, **params):
    params = {k: v for k, v in params.items() if v is not None}
    if EMBEDDED:
        r = embedded_client().get(path, params=params)
    else:
        r = httpx.get(f"{API}{path}", params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


# ----------------------------------------------------------------------------- formatting helpers

def pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def usd(x) -> str:
    if x is None:
        return "unpriced"
    return "$0" if x == 0 else f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def md(s: str) -> str:
    """Escape '$' so Streamlit markdown doesn't render dollar amounts as LaTeX."""
    return s.replace("$", "\\$")


def ago(ts: str) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:  # SQLite drops the offset; values are stored in UTC
        dt = dt.replace(tzinfo=timezone.utc)
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if secs >= n:
            return f"{int(secs // n)}{unit} ago"
    return "just now"


def gate_badge(passed):
    if passed is True:
        st.badge("Gate passed", icon=":material/check_circle:", color="green")
    elif passed is False:
        st.badge("Gate failed", icon=":material/cancel:", color="red")
    else:
        st.badge("No gate run", icon=":material/remove:", color="gray")


VERDICT = {"stable": ("green", ":material/check_circle:"), "order-invariant": ("green", ":material/check_circle:"),
           "resistant": ("green", ":material/shield:"), "not applicable": ("gray", ":material/remove:"),
           "unstable": ("orange", ":material/warning:"), "order-sensitive": ("orange", ":material/warning:"),
           "inflates scores": ("orange", ":material/warning:"), "vulnerable": ("red", ":material/gpp_bad:")}
VERDICT_TEXT = {"stable": "✅ stable", "order-invariant": "✅ order-invariant", "resistant": "✅ resistant",
                "unstable": "⚠️ unstable", "order-sensitive": "⚠️ order-sensitive",
                "inflates scores": "⚠️ inflates scores", "vulnerable": "❌ vulnerable", "not applicable": "— n/a"}


def header(title: str, subtitle: str):
    st.title(title)
    st.caption(subtitle)


def chart(c, height=240):
    st.altair_chart(c.properties(height=height), width="stretch")


def runs_df() -> pd.DataFrame:
    df = pd.DataFrame(get("/runs", limit=500))
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")
    return df


def calibrated_rubrics() -> list[str]:
    return sorted({e["rubric"] for e in get("/calibration/examples")})


def time_axis(ts: pd.Series) -> alt.Axis:
    span = (ts.max() - ts.min()).total_seconds() if len(ts) > 1 else 0
    fmt = "%H:%M:%S" if span < 3600 else "%H:%M" if span < 86400 else "%b %d"
    return alt.Axis(format=fmt, labelAngle=0, labelOverlap=True, tickCount=6, title=None)


# ----------------------------------------------------------------------------- overview

def page_overview():
    header("Overview", "Quality, cost and judge health for every AI feature that calls the eval service.")
    df = runs_df()
    if df.empty:
        st.info("No eval runs yet. Run `python scripts/demo.py` to create some.", icon=":material/info:")
        return
    done = df[df["status"] == "done"]

    with st.container(horizontal=True):
        st.metric("Features", df["feature"].nunique(), border=True)
        st.metric("Eval runs", len(df), border=True)
        st.metric("Outputs scored", f"{int(df['completed'].sum()):,}", border=True)
        gated = df[df["gate_passed"].notna()]
        st.metric("Gates failed", int((gated["gate_passed"] == False).sum()),  # noqa: E712
                  f"of {len(gated)} checked", delta_color="off", delta_arrow="off", border=True)
        st.metric("Judge spend", usd(df["cost_usd"].sum(min_count=1) if df["cost_usd"].notna().all() else None),
                  border=True)

    st.subheader("Features")
    features = sorted(done["feature"].unique())
    for row_start in range(0, len(features), 3):
        cols = st.columns(3)
        for col, feat in zip(cols, features[row_start:row_start + 3]):
            hist = done[done["feature"] == feat]
            last = hist.iloc[-1]
            prev = hist.iloc[-2] if len(hist) > 1 else None
            delta = None
            if prev is not None and last["pass_rate"] is not None and prev["pass_rate"] is not None:
                diff = round((last["pass_rate"] - prev["pass_rate"]) * 100)
                delta = f"{diff:+d} pts vs previous run" if diff else "no change vs previous run"
            with col.container(border=True):
                st.markdown(f"**{feat}**")
                st.caption(f"{last['rubric_name']} v{last['rubric_version']} · {last['judge']}")
                st.metric("Pass rate (latest run)", pct(last["pass_rate"]), delta,
                          delta_color="normal" if delta and "pts" in delta else "off",
                          delta_arrow="auto" if delta and "pts" in delta else "off",
                          chart_data=[x * 100 for x in hist["pass_rate"].dropna()] if len(hist) > 1 else None,
                          chart_type="line")
                gate_badge(None if pd.isna(last["gate_passed"]) else bool(last["gate_passed"]))
                st.caption(md(f"{last['name']} · {ago(last['created_at'].isoformat())} · mean "
                              f"{last['mean_score']:.2f}/5 · {len(hist)} runs · {usd(hist['cost_usd'].sum())} spend"))

    st.subheader("Judge health")
    st.caption("Latest calibration (agreement with human labels) and consistency check for each rubric.")
    rows = []
    cons = {}
    for c in get("/consistency/runs"):
        if c["status"] == "done" and c["rubric_name"] not in cons:
            cons[c["rubric_name"]] = c
    seen = set()
    for cal in get("/calibration/runs"):
        if cal["status"] != "done" or cal["rubric_name"] in seen:
            continue
        seen.add(cal["rubric_name"])
        m = cal["metrics"]
        kappas = [v["weighted_kappa"] for v in m["dimensions"].values() if v["weighted_kappa"] is not None]
        cr = cons.get(cal["rubric_name"])
        checks = (cr or {}).get("metrics", {}).get("checks", {}) if cr else {}
        rows.append({
            "Rubric": cal["rubric_name"], "Judge": cal["judge"], "Labelled examples": m["n_scored"],
            "Pass/fail accuracy": m["pass_fail"]["accuracy"], "False-pass rate": m["pass_fail"]["false_pass_rate"],
            "Mean κ": sum(kappas) / len(kappas) if kappas else None,
            "Repeat": VERDICT_TEXT.get(checks.get("repeat", {}).get("verdict"), "not run"),
            "Order": VERDICT_TEXT.get(checks.get("context_order", {}).get("verdict"), "not run"),
            "Injection": VERDICT_TEXT.get(checks.get("injection", {}).get("verdict"), "not run"),
        })
    if rows:
        hdf = pd.DataFrame(rows)
        for col in ("Pass/fail accuracy", "False-pass rate"):
            hdf[col] = hdf[col] * 100
        st.dataframe(hdf, hide_index=True, width="stretch", column_config={
            "Pass/fail accuracy": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
            "False-pass rate": st.column_config.NumberColumn(format="%.0f%%",
                                                             help="Share of human-failed outputs the judge passed"),
            "Mean κ": st.column_config.NumberColumn(format="%.2f", help="Mean quadratic-weighted κ across dimensions"),
        })
    else:
        st.caption("No calibration runs yet: `python scripts/calibrate.py --rubric <name>`")


# ----------------------------------------------------------------------------- runs

RUN_COLUMNS = {
    "name": st.column_config.TextColumn("Run"),
    "feature": st.column_config.TextColumn("Feature"),
    "status": st.column_config.TextColumn("Status"),
    "pass_rate": st.column_config.ProgressColumn("Pass rate", format="%.0f%%", min_value=0, max_value=100),
    "mean_score": st.column_config.NumberColumn("Mean score", format="%.2f"),
    "gate": st.column_config.TextColumn("Gate"),
    "items": st.column_config.TextColumn("Items"),
    "cost": st.column_config.TextColumn("Judge cost"),
    "rubric": st.column_config.TextColumn("Rubric"),
    "judge": st.column_config.TextColumn("Judge"),
    "created_at": st.column_config.DatetimeColumn("Created", format="MMM D, HH:mm"),
}


def _runs_table(df: pd.DataFrame):
    t = df.sort_values("created_at", ascending=False).assign(
        gate=df["gate_passed"].map(lambda g: "✅ passed" if g is True else "❌ failed" if g is False else "—"),
        items=df["completed"].astype(str) + " / " + df["total"].astype(str),
        cost=df["cost_usd"].map(usd), rubric=df["rubric_name"] + " v" + df["rubric_version"].astype(str),
        pass_rate=df["pass_rate"] * 100)
    cols = [c for c in RUN_COLUMNS if c != "status" or (df["status"] != "done").any()]
    st.dataframe(t[cols], hide_index=True, width="stretch", column_config=RUN_COLUMNS)


def _highlight(output: str, dims: list[dict]) -> str:
    spans = sorted({(e["start"], e["end"]) for d in dims for e in d["evidence"]
                    if e.get("grounded") and e.get("start") is not None and e["end"] <= len(output)})
    out, pos = [], 0
    for s, e in spans:
        if s < pos:
            continue
        out.append(html.escape(output[pos:s]))
        out.append(f"<mark>{html.escape(output[s:e])}</mark>")
        pos = e
    out.append(html.escape(output[pos:]))
    return f'<div class="ev-out">{"".join(out)}</div>'


def page_runs():
    header("Runs", "Every batch evaluation, with score distributions and the failures behind them.")
    df = runs_df()
    if df.empty:
        st.info("No eval runs yet. Run `python scripts/demo.py` to create some.", icon=":material/info:")
        return
    features = sorted(df["feature"].unique())
    chosen = st.pills("Feature", features, selection_mode="multi", default=features)
    view = df[df["feature"].isin(chosen or features)]

    done = view[view["pass_rate"].notna()]
    if not done.empty:
        with st.container(border=True):
            st.markdown("**Pass rate over time**")
            base = alt.Chart(done).encode(
                x=alt.X("created_at:T", axis=time_axis(done["created_at"])),
                y=alt.Y("pass_rate:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", tickCount=5)),
                color=alt.Color("feature:N", title=None, scale=alt.Scale(domain=features, range=series()),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("name:N", title="Run"), alt.Tooltip("feature:N", title="Feature"),
                         alt.Tooltip("pass_rate:Q", title="Pass rate", format=".0%"),
                         alt.Tooltip("mean_score:Q", title="Mean score", format=".2f"),
                         alt.Tooltip("created_at:T", title="Created", format="%b %d %H:%M")])
            chart(base.mark_line(strokeWidth=2) + base.mark_point(size=70, filled=True), 220)

    _runs_table(view)

    st.divider()
    ids = view.sort_values("created_at", ascending=False)["id"].tolist()
    names = df.set_index("id")["name"].to_dict()
    run_id = st.selectbox("Inspect run", ids, format_func=lambda i: f"{names[i]}  ·  {i[:8]}")
    rep = get(f"/runs/{run_id}/report")
    c = rep["counts"]

    title_col, badge_col = st.columns([3, 2], vertical_alignment="center")
    title_col.subheader(rep["name"])
    with badge_col.container(horizontal=True, horizontal_alignment="right"):
        gate_badge(rep["gate"]["passed"])
        st.badge(rep["rubric"], icon=":material/rule:", color="blue")
        st.badge(rep["judge"], icon=":material/gavel:", color="gray")

    ci = rep["pass_rate_ci95"]
    cost = rep["cost"]
    with st.container(horizontal=True):
        st.metric("Pass rate", pct(rep["pass_rate"]), f"95% CI {pct(ci[0])}–{pct(ci[1])}" if ci else None,
                  delta_color="off", delta_arrow="off", border=True)
        st.metric("Mean score", f"{rep['overall']['mean']:.2f}" if rep["overall"]["mean"] else "—",
                  "out of 5", delta_color="off", delta_arrow="off", border=True)
        st.metric("Failed", c["failed"], f"of {c['scored']} scored", delta_color="off", delta_arrow="off", border=True)
        st.metric("Needs review", c["needs_human_review"], f"{c['errors']} judge errors", delta_color="off",
                  delta_arrow="off", border=True)
        st.metric("Judge cost", usd(cost["total_usd"]), f"{usd(cost['per_1k_items_usd'])} per 1k items",
                  delta_color="off", delta_arrow="off", border=True,
                  help=f"{cost['judge_calls']} judge calls · {cost['input_tokens'] + cost['output_tokens']:,} tokens · "
                       f"cache hit rate {pct(cost['cache_hit_rate'])}")

    left, right = st.columns(2)
    with left.container(border=True):
        st.markdown("**Overall score distribution**")
        h = pd.DataFrame({"bin": list(rep["overall"]["histogram"]), "count": list(rep["overall"]["histogram"].values())})
        chart(alt.Chart(h).mark_bar(color=series()[0], cornerRadiusEnd=4).encode(
            x=alt.X("bin:N", title="Overall score", sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count:Q", title=None, axis=alt.Axis(tickMinStep=1)),
            tooltip=[alt.Tooltip("bin", title="Score range"), alt.Tooltip("count", title="Items")]))
    with right.container(border=True):
        weakest = rep["weakest_dimension"]
        st.markdown("**Dimension means** with 95% CI" + (f" · weakest: `{weakest}`" if weakest else ""))
        d = pd.DataFrame([{"dimension": k, "mean": v["mean"], "lo": v["mean_ci95"][0], "hi": v["mean_ci95"][1],
                           "fail_rate": v["fail_rate"]} for k, v in rep["dimensions"].items()])
        if not d.empty:
            enc_y = alt.Y("dimension:N", title=None, sort="-x", scale=alt.Scale(paddingInner=0.5))
            bars = alt.Chart(d).mark_bar(color=series()[0], cornerRadiusEnd=4).encode(
                y=enc_y, x=alt.X("mean:Q", title=None, scale=alt.Scale(domain=[0, 5])),
                tooltip=[alt.Tooltip("dimension", title="Dimension"), alt.Tooltip("mean:Q", title="Mean", format=".2f"),
                         alt.Tooltip("lo:Q", title="CI low", format=".2f"), alt.Tooltip("hi:Q", title="CI high", format=".2f"),
                         alt.Tooltip("fail_rate:Q", title="Fail rate", format=".0%")])
            err = alt.Chart(d).mark_rule(color=INK[mode()], strokeWidth=2).encode(y=enc_y, x="lo:Q", x2="hi:Q")
            chart(bars + err)

    st.subheader(f"Failed cases ({c['failed']})")
    if not rep["failed_cases"]:
        st.success("Every scored item passed.", icon=":material/check_circle:")
    for fc in rep["failed_cases"]:
        dims = fc["failing_dimensions"]
        label = f"**{fc['item_id']}** — {fc['overall_score']:.2f}/5 — " + ", ".join(
            f"{d['dimension']} {d['score']}/5" for d in dims)
        with st.expander(label):
            st.markdown(f'<div class="muted">Prompt</div>{html.escape(fc["prompt"])}', unsafe_allow_html=True)
            st.markdown('<div class="muted" style="margin-top:.8rem">Output — highlighted text is the evidence '
                        'the judge cited</div>' + _highlight(fc["output"], dims), unsafe_allow_html=True)
            st.write("")
            for d in dims:
                need = f" (needs {d['min_score']})" if d.get("min_score") else ""
                st.markdown(md(f"**{d['dimension']} · {d['score']}/5{need}** — {d['reasoning']}"))
                for e in d["evidence"]:
                    st.caption(md(f"“{e['quote']}” — {e['issue']}" + ("" if e.get("grounded") else
                                                                     "  ⚠️ quote not found in output")))
            if fc["warnings"]:
                with st.container(horizontal=True):
                    for w in fc["warnings"]:
                        st.badge(w, color="orange")


# ----------------------------------------------------------------------------- compare

def page_compare():
    header("Compare runs", "What changed between a baseline and a candidate release.")
    runs = get("/runs", limit=200)
    if len(runs) < 2:
        st.info("Need at least two runs.", icon=":material/info:")
        return
    label = {r["id"]: f"{r['name']} · {r['feature']} · {r['id'][:8]}" for r in runs}
    c1, c2 = st.columns(2)
    a = c1.selectbox("Baseline (A)", list(label), index=1, format_func=label.get)
    b = c2.selectbox("Candidate (B)", list(label), index=0, format_func=label.get)
    cmp = get("/runs/compare", a=a, b=b)
    if not (cmp["same_judge"] and cmp["same_rubric"]):
        st.warning("These runs used a different judge or rubric version, so the deltas mix a model change with a "
                   "judge change.", icon=":material/warning:")
    with st.container(horizontal=True):
        st.metric("Pass rate (B)", pct(cmp["b"]["pass_rate"]),
                  None if cmp["pass_rate_delta"] is None else f"{cmp['pass_rate_delta'] * 100:+.0f} pts", border=True)
        st.metric("Mean score (B)", f"{cmp['b']['mean']:.2f}" if cmp["b"]["mean"] else "—",
                  None if cmp["mean_delta"] is None else f"{cmp['mean_delta']:+.2f}", border=True)
        st.metric("Regressed items", len(cmp["regressed_items"]), "passed in A, fail in B", delta_color="off",
                  delta_arrow="off", border=True)
        st.metric("Fixed items", len(cmp["fixed_items"]), "failed in A, pass in B", delta_color="off",
                  delta_arrow="off", border=True)

    with st.container(border=True):
        st.markdown("**Dimension means**")
        rows = [{"dimension": d, "run": "A · baseline", "mean": v["a"]} for d, v in cmp["dimensions"].items()] + \
               [{"dimension": d, "run": "B · candidate", "mean": v["b"]} for d, v in cmp["dimensions"].items()]
        df = pd.DataFrame(rows).dropna()
        chart(alt.Chart(df).mark_bar(cornerRadiusEnd=4).encode(
            y=alt.Y("dimension:N", title=None, scale=alt.Scale(paddingInner=0.35)), yOffset="run:N",
            x=alt.X("mean:Q", title=None, scale=alt.Scale(domain=[0, 5])),
            color=alt.Color("run:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["A · baseline", "B · candidate"], range=series()[:2])),
            tooltip=[alt.Tooltip("dimension", title="Dimension"), alt.Tooltip("run", title="Run"),
                     alt.Tooltip("mean:Q", title="Mean", format=".2f")]), 70 * max(len(cmp["dimensions"]), 1))
        deltas = pd.DataFrame([{"Dimension": d, "A": v["a"], "B": v["b"], "Change": v["delta"]}
                               for d, v in cmp["dimensions"].items()])
        st.dataframe(deltas, hide_index=True, width="stretch",
                     column_config={k: st.column_config.NumberColumn(format="%.2f") for k in ("A", "B")} |
                     {"Change": st.column_config.NumberColumn(format="%+.2f")})
    if cmp["regressed_items"]:
        st.markdown("**Regressed items**")
        with st.container(horizontal=True):
            for i in cmp["regressed_items"]:
                st.badge(i, icon=":material/trending_down:", color="red")


# ----------------------------------------------------------------------------- calibration

def page_calibration():
    header("Calibration & drift", "How closely each judge agrees with human labels, and whether that changed.")
    rubric = st.pills("Rubric", calibrated_rubrics(), default="summarization") or "summarization"
    done = [r for r in get("/calibration/runs", rubric=rubric) if r["status"] == "done"]
    if not done:
        st.info(f"No calibration runs for {rubric}. Start one: `python scripts/calibrate.py --rubric {rubric}`",
                icon=":material/info:")
        return
    opts = {r["id"]: f"{r['judge']} · rubric v{r['rubric_version']} · {ago(r['created_at'])} · {r['id'][:8]}"
            for r in done}
    run_id = st.selectbox("Calibration run", list(opts), format_func=opts.get)
    run = get(f"/calibration/runs/{run_id}")
    m = run["metrics"]
    pf, th = m["pass_fail"], m["threshold_tuning"]

    with st.container(horizontal=True):
        st.metric("Pass/fail accuracy", pct(pf["accuracy"]), f"κ {pf['kappa']:.2f}", delta_color="off",
                  delta_arrow="off", border=True, help="Agreement between judge and human pass/fail decisions")
        st.metric("False-pass rate", pct(pf["false_pass_rate"]), "judge passed, human failed", delta_color="off",
                  delta_arrow="off", border=True, help="The dangerous direction for a release gate")
        st.metric("False-fail rate", pct(pf["false_fail_rate"]), "judge failed, human passed", delta_color="off",
                  delta_arrow="off", border=True)
        st.metric("Overall bias", f"{m['overall']['mean_bias']:+.2f}",
                  "too generous" if m["overall"]["mean_bias"] > 0 else "too harsh", delta_color="off",
                  delta_arrow="off", border=True)
        st.metric("Suggested threshold", th["best_threshold"] if th["best_threshold"] is not None else "—",
                  f"current {m['pass_threshold']}", delta_color="off", delta_arrow="off", border=True,
                  help="The overall-score cut-off that best reproduces human pass/fail (balanced accuracy)")

    st.markdown("**Per-dimension agreement**")
    dims = pd.DataFrame([{"Dimension": d, "κ (weighted)": v["weighted_kappa"], "Spearman": v["spearman"],
                          "Exact": 100 * v["exact_agreement"], "Within 1": 100 * v["within_one"], "MAE": v["mae"],
                          "Bias": v["mean_bias"], "Too generous": v["too_generous"], "Too harsh": v["too_harsh"]}
                         for d, v in m["dimensions"].items()])
    st.dataframe(dims, hide_index=True, width="stretch", column_config={
        "κ (weighted)": st.column_config.NumberColumn(format="%.2f", help="Quadratic-weighted Cohen's κ; 1 = perfect"),
        "Spearman": st.column_config.NumberColumn(format="%.2f"),
        "Exact": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "Within 1": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
        "MAE": st.column_config.NumberColumn(format="%.2f"),
        "Bias": st.column_config.NumberColumn(format="%+.2f", help="> 0 judge too generous, < 0 too harsh"),
    })

    left, right = st.columns(2)
    with left.container(border=True):
        dim = st.segmented_control("Judge vs human", list(m["dimensions"]), default=list(m["dimensions"])[0]) \
              or list(m["dimensions"])[0]
        pairs = [(e["judge_scores"][dim], e["human_scores"][dim]) for e in run["per_example"]
                 if e["status"] == "ok" and dim in e["judge_scores"]]
        grid = pd.DataFrame([{"judge": j, "human": h} for j in range(1, 6) for h in range(1, 6)])
        grid["count"] = [sum(1 for p in pairs if p == (j, h)) for j, h in zip(grid["judge"], grid["human"], strict=True)]
        grid["agree"] = grid["judge"] == grid["human"]
        nz = grid[grid["count"] > 0]
        empty = alt.Chart(grid).mark_rect(color=EMPTY_CELL[mode()], cornerRadius=3).encode(
            x=alt.X("human:O", title="Human score", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("judge:O", title="Judge score", sort="descending"))
        filled = alt.Chart(nz).mark_rect(cornerRadius=3).encode(
            x="human:O", y=alt.Y("judge:O", sort="descending"),
            color=alt.Color("count:Q", legend=None, scale=alt.Scale(range=SEQ[mode()][1:])),
            tooltip=[alt.Tooltip("human", title="Human"), alt.Tooltip("judge", title="Judge"),
                     alt.Tooltip("count", title="Examples")])
        text = alt.Chart(nz).mark_text(fontSize=13, fontWeight=600).encode(
            x="human:O", y=alt.Y("judge:O", sort="descending"), text="count:Q",
            color=alt.condition(f"datum.count >= {max(2, nz['count'].max() // 2) if not nz.empty else 2}",
                                alt.value("#ffffff"), alt.value("#0b0b0b")))
        chart(empty + filled + text, 270)
        st.caption("Diagonal = exact agreement. Above it the judge is too generous; below it, too harsh.")
    with right.container(border=True):
        st.markdown("**Mean overall score by quality tier**")
        tiers = pd.DataFrame([{"tier": t, "rater": r, "mean": v[f"{r}_mean"]} for t, v in m["by_tier"].items()
                              for r in ("human", "judge")])
        chart(alt.Chart(tiers).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("tier:N", sort=["excellent", "mediocre", "bad"], title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("rater:N", sort=["human", "judge"]),
            y=alt.Y("mean:Q", title=None, scale=alt.Scale(domain=[0, 5])),
            color=alt.Color("rater:N", title=None, legend=alt.Legend(orient="top"),
                            scale=alt.Scale(domain=["human", "judge"], range=series()[:2])),
            tooltip=[alt.Tooltip("tier", title="Tier"), alt.Tooltip("rater", title="Rater"),
                     alt.Tooltip("mean:Q", title="Mean", format=".2f")]), 298)

    if m["largest_disagreements"]:
        st.markdown("**Largest disagreements** — read these when tuning rubric prompts")
        st.dataframe(pd.DataFrame(m["largest_disagreements"]).rename(columns={
            "id": "Example", "dimension": "Dimension", "judge": "Judge", "human": "Human", "direction": "Direction",
            "judge_reasoning": "Judge reasoning"}), hide_index=True, width="stretch")

    if len(done) >= 2:
        with st.expander("Judge drift — compare two calibration runs", icon=":material/compare_arrows:"):
            ids = list(opts)
            c1, c2 = st.columns(2)
            a = c1.selectbox("From", ids, index=1, format_func=opts.get)
            b = c2.selectbox("To", ids, index=0, format_func=opts.get)
            drift = get("/calibration/drift", a=a, b=b)
            if drift["drifted"]:
                st.warning(f"Drift detected: {drift['pass_decision_flips']} of {drift['n_common']} pass/fail decisions "
                           "changed.", icon=":material/warning:")
            else:
                st.success(f"No material drift: {drift['pass_decision_flips']} of {drift['n_common']} pass/fail "
                           "decisions changed.", icon=":material/check_circle:")
            st.dataframe(pd.DataFrame([{"Dimension": d, "Mean shift": v["mean_shift"],
                                        "Share changed": 100 * (v["share_changed"] or 0), "κ before": v["kappa_before"],
                                        "κ after": v["kappa_after"], "κ change": v["kappa_change"]}
                                       for d, v in drift["dimensions"].items()]),
                         hide_index=True, width="stretch", column_config={
                             "Mean shift": st.column_config.NumberColumn(format="%+.2f"),
                             "Share changed": st.column_config.NumberColumn(format="%.0f%%"),
                             "κ before": st.column_config.NumberColumn(format="%.2f"),
                             "κ after": st.column_config.NumberColumn(format="%.2f"),
                             "κ change": st.column_config.NumberColumn(format="%+.2f")})


# ----------------------------------------------------------------------------- consistency

CHECK_INFO = {
    "repeat": ("Repeat stability", "Same input scored several times. Scores should not change."),
    "context_order": ("Passage order", "Retrieved passages reversed. Meaning is unchanged, so scores should be too."),
    "injection": ("Injection resistance", "A 'score this 5' note hidden in the output. Scores must not go up."),
}


def page_consistency():
    header("Judge consistency", "Does the judge agree with itself under changes that shouldn't matter?")
    rubric = st.pills("Rubric", calibrated_rubrics(), default="rag_faithfulness") or "rag_faithfulness"
    runs = [r for r in get("/consistency/runs", rubric=rubric) if r["status"] == "done"]
    if not runs:
        st.info(f"No consistency runs for {rubric}. Start one: `python scripts/consistency.py --rubric {rubric}`",
                icon=":material/info:")
        return
    opts = {r["id"]: f"{r['judge']} · {ago(r['created_at'])} · {r['id'][:8]}" for r in runs}
    run = get(f"/consistency/runs/{st.selectbox('Consistency run', list(opts), format_func=opts.get)}")
    m, cfg = run["metrics"], run["config"]
    st.caption(md(f"{m['n_scored']} items from **{cfg['source']}** · {m['cost']['judge_calls']} judge calls · "
                  f"cost {usd(m['cost']['total_usd'])} · cache bypassed"))

    checks = m["checks"]
    cols = st.columns(len(checks))
    for col, (name, c) in zip(cols, checks.items(), strict=True):
        title, desc = CHECK_INFO[name]
        with col.container(border=True, height=250):
            st.markdown(f"**{title}**")
            color, icon = VERDICT.get(c["verdict"], ("gray", None))
            st.badge(c["verdict"], icon=icon, color=color)
            st.caption(desc)
            if "pass_flip_rate" in c:
                st.metric("Pass/fail flips", pct(c["pass_flip_rate"]))
            if name == "injection" and c.get("n"):
                st.caption(f"Fail→pass: {pct(c['attack_success_rate'])} · any score inflated: "
                           f"{pct(c['score_inflation_rate'])}")

    if "repeat" in checks:
        st.markdown(f"**Repeat stability by dimension** — {checks['repeat']['repeats']} repeats per item")
        st.dataframe(pd.DataFrame([{"Dimension": d, "Identical every time": 100 * v["stable_rate"], "Mean std": v["mean_std"],
                                    "Max range": v["max_range"]} for d, v in checks["repeat"]["dimensions"].items()]),
                     hide_index=True, width="stretch", column_config={
                         "Identical every time": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
                         "Mean std": st.column_config.NumberColumn(format="%.2f")})
    for name in ("context_order", "injection"):
        c = checks.get(name)
        if c and c.get("dimensions"):
            st.markdown(f"**{CHECK_INFO[name][0]} by dimension** — change vs the unperturbed input")
            st.dataframe(pd.DataFrame([{"Dimension": d, "Mean change": v["mean_delta"], "Mean |change|": v["mean_abs_delta"],
                                        "Share increased": 100 * v["share_increased"]} for d, v in c["dimensions"].items()]),
                         hide_index=True, width="stretch", column_config={
                             "Mean change": st.column_config.NumberColumn(format="%+.2f"),
                             "Mean |change|": st.column_config.NumberColumn(format="%.2f"),
                             "Share increased": st.column_config.NumberColumn(format="%.0f%%")})


# ----------------------------------------------------------------------------- metric health

def page_health():
    header("Metric health", "Is the metric itself trustworthy enough to gate releases?")
    c1, c2 = st.columns(2)
    rubric = c1.selectbox("Rubric", [None] + [r["name"] for r in get("/rubrics")], format_func=lambda x: x or "All rubrics")
    feature = c2.selectbox("Feature", [None] + sorted({r["feature"] for r in get("/runs", limit=500)}),
                           format_func=lambda x: x or "All features")
    h = get("/metrics/health", rubric=rubric, feature=feature)
    with st.container(horizontal=True):
        st.metric("Scored results", f"{h['n_scored']:,}", border=True)
        st.metric("Judge error rate", pct(h["error_rate"]), border=True)
        st.metric("Flagged for review", pct(h["flagged_for_review_rate"]), border=True)
        st.metric("Human override rate", pct(h["human_override_rate"]), f"{h['human_reviewed']} reviewed",
                  delta_color="off", delta_arrow="off", border=True,
                  help="Share of human-reviewed results where the reviewer flipped the judge's pass/fail")
    rows = []
    for d, v in h["dimensions"].items():
        dis = v["judge_disagreement"] or {}
        rows.append({"Dimension": d, "n": v["n"], "Mean": v["mean"], "Variance": v["variance"],
                     "Share at most common score": 100 * v["share_at_mode"], "Mean judge confidence": v["mean_confidence"],
                     "2nd-judge pairs": dis.get("n"), "2nd-judge |Δ|": dis.get("mean_abs_diff"),
                     "2nd-judge κ": dis.get("weighted_kappa")})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Mean": st.column_config.NumberColumn(format="%.2f"),
        "Variance": st.column_config.NumberColumn(format="%.2f"),
        "Share at most common score": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                                                      help="Near 100% = the dimension isn't discriminating"),
        "Mean judge confidence": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
    })
    st.caption("Low variance with most scores at one value means a dimension isn't discriminating between outputs. "
               "High second-judge disagreement or override rate means the metric needs recalibration before it gates "
               "releases.")


# ----------------------------------------------------------------------------- navigation

nav = st.navigation({
    "": [st.Page(page_overview, title="Overview", icon=":material/space_dashboard:", default=True)],
    "Evaluations": [st.Page(page_runs, title="Runs", icon=":material/playlist_add_check:", url_path="runs"),
                    st.Page(page_compare, title="Compare runs", icon=":material/compare_arrows:", url_path="compare")],
    "Judge quality": [st.Page(page_calibration, title="Calibration & drift", icon=":material/tune:", url_path="calibration"),
                      st.Page(page_consistency, title="Consistency", icon=":material/repeat:", url_path="consistency"),
                      st.Page(page_health, title="Metric health", icon=":material/monitor_heart:", url_path="health")],
})
_HERE = os.path.dirname(os.path.abspath(__file__))
st.logo(os.path.join(_HERE, "logo.svg"), icon_image=os.path.join(_HERE, "logo_icon.svg"), size="large")
with st.sidebar:
    if EMBEDDED:
        st.info("**Demo mode.** The eval API runs inside this app on a temporary database, seeded at startup "
                "with sample data scored by the offline heuristic judge (no LLM calls).", icon=":material/science:")
    else:
        st.caption(f"Connected to {API}")
try:
    nav.run()
except httpx.HTTPError as e:
    st.error(f"Could not reach the eval API at {API}: {e}", icon=":material/cloud_off:")
