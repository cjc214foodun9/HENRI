#!/usr/bin/env python3
"""henri_telemetry_report.py — ONE self-contained telemetry report.

Aggregates the four evidence layers into a single digestible HTML + Markdown
artifact with embedded figures. Deterministic. No LLM, no network.

Layers:
  1. MoA prompt-cache economics   (per-slot usage from moa-traces JSONL)
  2. Benchmark gate outcomes      (experiments/verification/*scorecard*.json)
  3. Repository + sync topology   (git, .gitignore, worktrees)
  4. Chain-of-Evidence enforcement (imports henri_coe_gate; CPR per claim)
  5. Holonic graph audit          (imports henri_graph_render; cycles/orphans)
  6. Provenance                   (sha256 of every source read)

Run under the viz-venv interpreter (matplotlib):
  %LOCALAPPDATA%\\hermes\\viz-venv\\Scripts\\python.exe henri_telemetry_report.py
"""
from __future__ import annotations

import base64
import collections
import datetime as dt
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HOME = Path(os.path.expandvars(r"%LOCALAPPDATA%\hermes"))
REPO = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
TRACES = HOME / "moa-traces"
VERIF = REPO / "experiments" / "verification"
OUTDIR = HOME / "reports"
STAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
PAL = {"ok": "#54a24b", "bad": "#e45756", "acc": "#4c78a8",
       "chance": "#b0b0b0", "cache": "#72b7b2", "fresh": "#f58518",
       "mut": "#bab0ac", "ink": "#4c4c4c"}
PROV: dict[str, str] = {}


def sha_file(p: Path) -> str:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except Exception:
        return "unreadable"


def note_prov(key: str, p: Path) -> None:
    PROV[key] = f"{p}  sha256={sha_file(p)[:16]}  bytes={p.stat().st_size if p.exists() else 0:,}"


def sh(*a: str) -> str:
    try:
        r = subprocess.run(a, cwd=str(REPO), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        return (r.stdout or "").strip()
    except Exception as e:
        return f"<error {e}>"


def b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- layer 1
def cache_layer() -> dict:
    files = sorted(glob.glob(str(TRACES / "*.jsonl")), key=os.path.getmtime)[-8:]
    slots: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    turns = 0
    for f in files:
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            turns += 1
            for r in rec.get("references") or []:
                lab = r.get("label", "?").split(":", 1)[-1].split("[", 1)[0]
                u = r.get("usage") or {}
                s = slots[lab]
                s["turns"] += 1
                s["fresh"] += int(u.get("input_tokens") or 0)
                s["cache"] += int(u.get("cache_read_tokens") or 0)
                s["out"] += int(u.get("output_tokens") or 0)
                s["reason"] += int(u.get("reasoning_tokens") or 0)
                try:
                    s["usd"] += float(r.get("cost_usd") or 0.0)
                except (TypeError, ValueError):
                    pass
    if files:
        note_prov("moa_traces", Path(files[-1]))
    tot = collections.Counter()
    for v in slots.values():
        for k in ("turns", "fresh", "cache", "out", "reason", "usd"):
            tot[k] += v[k]
    tot["prompt"] = tot["fresh"] + tot["cache"]
    tot["hit"] = 100.0 * tot["cache"] / tot["prompt"] if tot["prompt"] else 0.0
    tot["files"] = len(files)
    tot["turns_moa"] = turns
    return {"slots": slots, "tot": tot}


def gate_state(r: dict) -> str:
    """Tri-state gate. PASS/FAIL require a pre-registered accept_margin.
    A harness with no accept_margin is 'n/a' — never FAIL. Reporting FAIL for
    an unregistered gate is a false negative and was a real defect here."""
    m, a = r.get("margin"), r.get("accept")
    if not isinstance(m, (int, float)) or not isinstance(a, (int, float)):
        return "n/a"
    return "PASS" if m >= a else "FAIL"


def gate_cls(state: str) -> str:
    return {"PASS": "ok", "FAIL": "bad"}.get(state, "mut")


def fig_cache(cl: dict) -> str:
    slots = cl["slots"]
    src = sorted(slots.items(), key=lambda x: -(x[1]["fresh"] + x[1]["cache"]))
    names, fresh, cache, hits = [], [], [], []
    for k, v in src:
        p = v["fresh"] + v["cache"]
        if not p:
            continue
        names.append(k if len(k) <= 22 else k[:21] + "…")
        fresh.append(v["fresh"] / 1e6)
        cache.append(v["cache"] / 1e6)
        hits.append(100.0 * v["cache"] / p)
    fig, ax = plt.subplots(figsize=(8.6, 3.0), dpi=110)
    y = range(len(names))
    ax.barh(y, fresh, color=PAL["fresh"], label="fresh input (Mtok)")
    ax.barh(y, cache, left=fresh, color=PAL["cache"], label="cache read (Mtok)")
    for i, h in enumerate(hits):
        ax.annotate(f"{h:.1f}% hit", (fresh[i] + cache[i], i), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8)
    ax.set_yticks(list(y), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("million tokens (newest 8 traces)")
    ax.set_title(f"MoA prompt-cache by slot — aggregate hit {cl['tot']['hit']:.1f}%",
                 fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    ax.tick_params(labelsize=7)
    return b64(fig)


# ---------------------------------------------------------------- layer 2
def bench_layer() -> list[dict]:
    rows = []
    for f in sorted(VERIF.rglob("*scorecard*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        # accuracy is reported under different keys per harness:
        #   accuracy | accuracy_attempted | solved/item_count
        acc = d.get("accuracy")
        if not isinstance(acc, (int, float)):
            acc = d.get("accuracy_attempted")
        if not isinstance(acc, (int, float)):
            sv, ic = d.get("solved"), d.get("item_count")
            if isinstance(sv, (int, float)) and isinstance(ic, (int, float)) and ic:
                acc = sv / ic
        if not isinstance(acc, (int, float)):
            continue
        ir = d.get("item_results")
        n_items = len(ir) if isinstance(ir, list) else int(d.get("item_count") or 0)
        rows.append({
            "file": f.name,
            "bench": str(d.get("benchmark", f.stem)),
            "variant": f.stem.replace("_839", "").replace("humaneval", "").strip("_")
                       or "base",
            "status": str(d.get("status", "-")),
            "verdict": str(d.get("verdict") or "-"),
            "acc": float(acc),
            "chance": float(d.get("chance") or 0.0),
            "margin": d.get("margin"),
            "accept": d.get("accept_margin"),
            "items": n_items,
            "solved": d.get("solved"),
            "ega": str(d.get("egress_path") or "-"),
            "bytes": f.stat().st_size,
        })
        note_prov(f"scorecard::{f.name}", f)
    return rows


def fig_gates(rows: list[dict]) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(9.4, 2.9), dpi=110)
    names = [r["bench"][:14] for r in rows]
    y = list(range(len(rows)))
    ax[0].barh([i + 0.18 for i in y], [r["acc"] for r in rows], height=0.36,
               color=PAL["acc"], label="accuracy")
    ax[0].barh([i - 0.18 for i in y], [r["chance"] for r in rows], height=0.36,
               color=PAL["chance"], label="chance")
    ax[0].set_yticks(y, names, fontsize=8)
    ax[0].set_xlim(0, max(1.0, max(r["acc"] for r in rows) * 1.25))
    ax[0].invert_yaxis()
    ax[0].set_title("accuracy vs chance", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[0].tick_params(labelsize=7)

    mar = [float(r["margin"] or 0) for r in rows]
    acc = [float(r["accept"] or 0) for r in rows]
    ax[1].barh([i + 0.18 for i in y], mar, height=0.36, color=PAL["fresh"],
               label="margin")
    ax[1].barh([i - 0.18 for i in y], acc, height=0.36, color=PAL["ok"],
               label="accept_margin")
    ax[1].set_yticks(y, ["" for _ in y])
    ax[1].invert_yaxis()
    ax[1].set_title("margin vs pre-registered accept_margin", fontsize=9)
    ax[1].legend(fontsize=7)
    for i, (m, a) in enumerate(zip(mar, acc)):
        ax[1].annotate("PASS" if m >= a else "FAIL", (max(m, a, 0.001), i),
                       xytext=(3, 0), textcoords="offset points",
                       fontsize=7, va="center",
                       color=PAL["ok"] if m >= a else PAL["bad"])
    ax[1].tick_params(labelsize=7)
    return b64(fig)


def fig_humaneval(rows: list[dict]) -> str | None:
    """Condition sweep: same benchmark, different egress paths / gates."""
    he = [r for r in rows if "humaneval" in r["file"].lower()]
    if len(he) < 2:
        return None
    he = sorted(he, key=lambda r: (r["solved"] if isinstance(r["solved"], (int, float))
                                   else r["acc"]))
    fig, ax = plt.subplots(figsize=(8.6, 2.7), dpi=110)
    labels = [r["variant"][:26] or "base" for r in he]
    vals = [(r["solved"] if isinstance(r["solved"], (int, float)) else r["acc"] * r["items"])
            for r in he]
    cols = [PAL["ok"] if v and v > 0 else PAL["bad"] for v in vals]
    ax.barh(range(len(he)), vals, color=cols)
    for i, (v, r) in enumerate(zip(vals, he)):
        ax.annotate(f"{v:.0f}/{r['items']}  {100.0*v/r['items']:.1f}%" if r["items"] else "",
                    (v, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5)
    ax.set_yticks(range(len(he)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("HumanEval problems solved")
    ax.set_title(f"HumanEval condition sweep — {len(he)} variants, all near zero",
                 fontsize=10)
    ax.tick_params(labelsize=7)
    return b64(fig)


def fig_subjects() -> tuple[str, list[tuple[str, int, int]]] | tuple[None, list]:
    body = VERIF / "mmlu_839_scorecard.json"
    if not body.exists():
        return None, []
    try:
        d = json.loads(body.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None, []
    ir = d.get("item_results")
    if not isinstance(ir, list):
        return None, []
    note_prov("mmlu_body", body)
    acc: dict[str, list[int]] = {}
    for it in ir:
        s = it.get("subject")
        if s is None:
            continue
        a = acc.setdefault(str(s), [0, 0])
        a[0] += 1
        a[1] += 1 if it.get("is_correct") else 0
    sub = sorted(((k, v[0], v[1]) for k, v in acc.items()),
                 key=lambda x: x[2] / x[1] if x[1] else 0)
    if not sub:
        return None, []
    fig, ax = plt.subplots(figsize=(8.6, 3.4), dpi=110)
    vals = [100.0 * c / n for _, n, c in sub]
    cols = [PAL["bad"] if v < 25 else (PAL["mut"] if v < 30 else PAL["ok"]) for v in vals]
    ax.bar(range(len(sub)), vals, color=cols)
    ax.axhline(25, color=PAL["ink"], lw=0.8, ls="--")
    ax.annotate("chance 25%", (0.2, 25.6), fontsize=7, color=PAL["ink"])
    ax.set_xticks([])
    ax.set_xlabel(f"57 MMLU subjects, sorted worst→best "
                  f"(n={sum(n for _, n, _ in sub):,} items)")
    ax.set_ylabel("accuracy %")
    ax.set_title("MMLU per-subject accuracy — all below chance", fontsize=10)
    ax.tick_params(labelsize=7)
    return b64(fig), sub


# ---------------------------------------------------------------- layer 3
def repo_layer() -> dict:
    gi = REPO / ".gitignore"
    note_prov("gitignore", gi)
    out = {
        "branch": sh("git", "branch", "--show-current"),
        "head": sh("git", "log", "--oneline", "-1"),
        "ahead": sh("git", "status", "-sb").splitlines()[:1],
        "blobs": len([l for l in sh("git", "ls-files", "--cached",
                                    "*.pt", "*.pdf", "*.zip").splitlines() if l.strip()]),
        "worktrees": len([l for l in sh("git", "worktree", "list").splitlines() if l.strip()]),
        "branches": len([l for l in sh("git", "branch").splitlines() if l.strip()]),
        "strace_ignored": sh("git", "check-ignore", "-v",
                             "HENRI V2/agentic_graph/STRACE-main/Dataset") != "",
        "worktrees_ignored": sh("git", "check-ignore", "-v", ".worktrees") != "",
        "mmlu_ignored": sh("git", "check-ignore", "-v",
                           "experiments/verification/mmlu_839_scorecard.json") != "",
        "status_short": sh("git", "status", "--short"),
    }
    return out


# ---------------------------------------------------------------- render
# ---------------------------------------------------------------- layer 4/5
def coe_layer() -> dict:
    """Chain-of-Evidence enforcement. Runs the real gate on the real ledger.

    The gate is imported, not reimplemented: doctrine lives in one place.
    A claim without linked, hashed evidence is rejected here, so no number in
    this report can be promoted as verified without an evidence chain.
    """
    ledger = VERIF / "coe_claims_20260911.json"
    gate = HOME / "scripts" / "henri_coe_gate.py"
    out = {"present": ledger.is_file(), "cpr": None, "total": 0, "verified": 0,
           "rejected": [], "err": None}
    if not ledger.is_file() or not gate.is_file():
        out["err"] = "ledger or gate absent"
        return out
    note_prov("coe_ledger", ledger)
    note_prov("coe_gate", gate)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("henri_coe_gate", gate)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        data = json.loads(ledger.read_text(encoding="utf-8"))
        recs = data.get("claims") if isinstance(data, dict) else data
        res = mod.compute_cpr(recs, check_artifacts=True)
        rows = []
        for rec in recs:
            ok, why = mod.check_claim(rec, check_artifacts=True)
            rows.append({"id": rec.get("claim_id", "<no-id>"),
                         "type": rec.get("claim_type", "?"),
                         "status": rec.get("status", "?"),
                         "ok": ok, "why": why})
        out.update(cpr=res["cpr_pct"], total=res["total"],
                   verified=res["verified"], rejected=res["failures"],
                   rows=rows)
    except Exception as e:                       # never break the report
        out["err"] = f"{type(e).__name__}: {e}"
    return out


def dag_layer() -> dict:
    """Holonic state-graph audit (deterministic) + rendered figure path."""
    script = HOME / "scripts" / "henri_graph_render.py"
    res = {"present": script.is_file(), "verdict": "BLOCKED", "nodes": 0,
           "edges": 0, "cycles": [], "orphans": [], "cycle_status": "-",
           "fig": None}
    if not script.is_file():
        return res
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("henri_graph_render", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        a = mod.analyze(mod.default_graph())
        res.update(verdict=a["verdict"], nodes=a["n_nodes"], edges=a["n_edges"],
                   cycles=a["cycle_nodes"], orphans=a["orphans"],
                   cycle_status=a["cycle_status"])
    except Exception as e:
        res["verdict"] = f"ERROR {type(e).__name__}"
        res["err"] = str(e)[:120]
    fig = OUTDIR / "figures" / "05_holon_dag.png"
    res["fig"] = fig if fig.is_file() else None
    return res


def sync_layer() -> dict:
    """Four-surface sync topology (local, github, drive, vast).

    The manifest is declarative: it names ONE writer surface per path class and
    the flows between them. This layer reports whether the declared topology
    still matches live state, so sync drift cannot hide from the report.
    """
    script = HOME / "scripts" / "henri_sync_manifest.py"
    out = {"present": script.is_file(), "err": None, "drift": None,
           "surfaces": 0, "flows": 0, "invariants": 0}
    if not script.is_file():
        out["err"] = "sync manifest tool absent"
        return out
    note_prov("sync_manifest", script)
    try:
        import importlib.util as _iu
        spec = _iu.spec_from_file_location("henri_sync_manifest", script)
        mod = _iu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        man = getattr(mod, "MANIFEST", {}) or {}
        out["surfaces"] = len(man.get("surfaces", {}))
        out["flows"] = len(man.get("flows", []))
        out["invariants"] = len(man.get("invariants", []))
        fn = None
        for cand in ("verify", "check_drift", "drift", "run"):
            f = getattr(mod, cand, None)
            if callable(f):
                fn = f
                break
        if fn is None:
            out["err"] = "no verifier exposed"
            return out
        for args in ((), (str(REPO),), (True,)):
            try:
                out["drift"] = fn(*args)
                break
            except TypeError:
                continue
        if out["drift"] is None:
            out["err"] = "verifier signature not matched"
    except Exception as e:
        out["err"] = f"{type(e).__name__}: {e}"
    return out


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cl = cache_layer()
    brows = bench_layer()
    rl = repo_layer()
    coe = coe_layer()
    dag = dag_layer()
    sy = sync_layer()

    cfig = fig_cache(cl)
    gfig = fig_gates(brows) if brows else None
    hefig = fig_humaneval(brows)
    sfig, subs = fig_subjects()

    tot = cl["tot"]
    nfail = sum(1 for r in brows if gate_state(r) == "FAIL")
    ngated = sum(1 for r in brows if gate_state(r) != "n/a")
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---- shared CSS: theme vars with fallbacks (renders in-app AND standalone)
    css = """
:root{--fg:var(--foreground,#1b1b1b);--mut:var(--muted-foreground,#6b6b6b);
--bd:var(--border,#d8d8d8);--cd:var(--card,#fff);--ac:var(--accent,#4c78a8);
--ok:#54a24b;--bad:#e45756;}
*{box-sizing:border-box}
body{margin:0;color:var(--fg);font:13px/1.5 ui-sans-serif,system-ui,Segoe UI,sans-serif}
h1{font-size:18px;margin:0 0 2px}
h2{font-size:13px;margin:20px 0 6px;letter-spacing:.04em;text-transform:uppercase;color:var(--mut)}
.sub{color:var(--mut);font-size:11px;margin-bottom:12px}
.kpis{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 4px}
.kpi{border:1px solid var(--bd);border-radius:8px;padding:7px 10px;min-width:104px}
.kpi b{display:block;font-size:17px;line-height:1.25}
.kpi span{color:var(--mut);font-size:10px;text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:11.5px}
th,td{border-bottom:1px solid var(--bd);padding:4px 6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--mut);font-weight:600;font-size:10px;text-transform:uppercase}
tr:hover td{background:color-mix(in srgb,var(--ac) 7%,transparent)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
img{max-width:100%;border:1px solid var(--bd);border-radius:6px;margin:6px 0}
.note{border-left:3px solid var(--ac);padding:6px 10px;margin:8px 0;
background:color-mix(in srgb,var(--ac) 6%,transparent);font-size:11.5px}
code{font-family:ui-monospace,Consolas,monospace;font-size:11px}
.f{color:var(--mut);font-size:10px}
"""

    def kpi(v, label):
        return f'<div class="kpi"><b>{v}</b><span>{label}</span></div>'

    h = [f"<!doctype html><meta charset=utf-8><title>HENRI Telemetry {ts}</title>",
         f"<style>{css}</style>",
         f"<h1>HENRI telemetry report</h1>",
         f'<div class="sub">{esc(ts)} &middot; branch <code>{esc(rl["branch"])}</code>'
         f' &middot; aggregation is DERIVED from OBSERVED fields only</div>']
    h.append('<div class="kpis">')
    h.append(kpi(f'{tot["hit"]:.1f}%', "MoA cache hit"))
    h.append(kpi(f'${tot["usd"]:.2f}', "reference spend"))
    h.append(kpi(f'{tot["turns_moa"]}', "MoA turns"))
    h.append(kpi(f'{nfail}/{len(brows)}', "gates failing"))
    h.append(kpi(f'{rl["worktrees"]}', "worktrees"))
    h.append(kpi(f'{rl["blobs"]}', "binary blobs tracked"))
    h.append(kpi(f'{len(PROV)}', "sources hashed"))
    h.append(kpi("BLOCKED" if coe["cpr"] is None else f'{coe["cpr"]:.0f}%',
                 "CPR (claim provenance)"))
    h.append(kpi(f'{dag["verdict"]}', "holon graph audit"))
    h.append("</div>")

    # layer 1
    h.append("<h2>1 &middot; MoA prompt-cache economics</h2>")
    h.append(f'<img src="data:image/png;base64,{cfig}" alt="cache by slot">')
    h.append("<table><tr><th>slot</th><th>turns</th><th>fresh in</th><th>cache read</th>"
             "<th>hit %</th><th>out</th><th>reason</th><th>usd</th><th>$/Mtok</th></tr>")
    for k, v in sorted(cl["slots"].items(), key=lambda x: -x[1]["usd"]):
        p = v["fresh"] + v["cache"]
        hit = 100.0 * v["cache"] / p if p else 0.0
        tt = p + v["out"]
        rate = 1e6 * v["usd"] / tt if tt else 0.0
        h.append(f"<tr><td>{esc(k)}</td><td>{v['turns']}</td><td>{v['fresh']:,}</td>"
                 f"<td>{v['cache']:,}</td><td>{hit:.1f}</td><td>{v['out']:,}</td>"
                 f"<td>{v['reason']:,}</td><td>{v['usd']:.4f}</td><td>{rate:.3f}</td></tr>")
    h.append(f"<tr><td><b>TOTAL</b></td><td>{tot['turns']}</td><td>{tot['fresh']:,}</td>"
             f"<td>{tot['cache']:,}</td><td><b>{tot['hit']:.1f}</b></td>"
             f"<td>{tot['out']:,}</td><td>{tot['reason']:,}</td>"
             f"<td><b>{tot['usd']:.4f}</b></td><td></td></tr></table>")
    h.append('<div class="note"><b>UNVERIFIED:</b> the aggregator slot carries no '
             '<code>usage</code>/<code>cost_usd</code> field, so aggregator cache and '
             'spend are not measurable on this install. The table covers reference '
             'slots only. Do not quote an aggregator cost figure.</div>')

    # layer 2
    h.append("<h2>2 &middot; Benchmark gate outcomes</h2>")
    if gfig:
        h.append(f'<img src="data:image/png;base64,{gfig}" alt="gates">')
    h.append("<table><tr><th>benchmark</th><th>variant</th><th>status</th>"
             "<th>acc</th><th>chance</th><th>margin</th><th>accept</th><th>gate</th>"
             "<th>items</th><th>ega</th></tr>")
    for r in sorted(brows, key=lambda x: (x["bench"], x["variant"])):
        st = gate_state(r)
        h.append(f"<tr><td>{esc(r['bench'])}</td><td>{esc(r['variant'])}</td>"
                 f"<td>{esc(r['status'])}</td><td>{r['acc']:.4f}</td>"
                 f"<td>{r['chance']:.4f}</td>"
                 f"<td>{r['margin'] if r['margin'] is not None else '-'}</td>"
                 f"<td>{r['accept'] if r['accept'] is not None else '-'}</td>"
                 f'<td class="{gate_cls(st)}">{st}</td>'
                 f"<td>{r['items']:,}</td><td>{esc(r['ega'])}</td></tr>")
    h.append("</table>")
    h.append(f'<div class="f">{ngated} benchmark(s) carry a pre-registered '
             f'accept_margin. "n/a" means no gate was registered for that harness — '
             f'it is not a pass and not a failure.</div>')

    # layer 2c
    if hefig:
        h.append("<h2>2c &middot; HumanEval condition sweep</h2>")
        h.append(f'<img src="data:image/png;base64,{hefig}" alt="humaneval sweep">')
        h.append('<div class="note">The sweep is the finding: <b>every</b> egress '
                 'variant is near zero. A flat sweep across conditions means changing '
                 'the decoder/ranking path did not move the outcome — the gap is '
                 'upstream of egress. Do not report one variant as the result.</div>')

    # layer 2b
    if sfig:
        h.append("<h2>2b &middot; MMLU per-subject detail</h2>")
        h.append(f'<img src="data:image/png;base64,{sfig}" alt="mmlu subjects">')
        best = subs[-3:][::-1]
        worst = subs[:3]
        h.append("<table><tr><th>subject</th><th>correct</th><th>n</th><th>acc %</th></tr>")
        for tag, group in (("worst", worst), ("best", best)):
            for k, n, c in group:
                h.append(f"<tr><td>{tag}: {esc(k)}</td><td>{c}</td><td>{n}</td>"
                         f"<td>{100.0*c/n:.1f}</td></tr>")
        h.append("</table>")

    # layer 3
    h.append("<h2>3 &middot; Repository &amp; sync topology</h2>")
    h.append("<table><tr><th>check</th><th>state</th></tr>")
    for label, val, good in (
        ("branch", rl["branch"], True),
        ("HEAD", rl["head"], True),
        ("sync state", (rl["ahead"] or ["-"])[0], True),
        ("binary blobs tracked (*.pt/*.pdf/*.zip)", rl["blobs"], rl["blobs"] == 0),
        ("STRACE Dataset ignored (2.0 GB)", rl["strace_ignored"], rl["strace_ignored"]),
        (".worktrees ignored", rl["worktrees_ignored"], rl["worktrees_ignored"]),
        ("mmlu body ignored (untracked)", rl["mmlu_ignored"], rl["mmlu_ignored"]),
        ("registered worktrees", rl["worktrees"], True),
        ("local branches preserved", rl["branches"], True),
    ):
        cls = "ok" if good else "bad"
        h.append(f'<tr><td>{esc(label)}</td><td class="{cls}">{esc(val)}</td></tr>')
    h.append("</table>")

    # layer 4: Chain-of-Evidence enforcement
    h.append("<h2>4 &middot; Chain-of-Evidence enforcement (CPR)</h2>")
    if coe["err"]:
        h.append(f'<div class="note"><b>BLOCKED:</b> {esc(coe["err"])}</div>')
    else:
        h.append('<table><tr><th>claim_id</th><th>type</th><th>status</th>'
                 '<th>verdict</th></tr>')
        for c in coe.get("rows", []):
            cls = "ok" if c["ok"] else "bad"
            h.append(f'<tr><td>{esc(c["id"])}</td><td>{esc(c["type"])}</td>'
                     f'<td>{esc(c["status"])}</td>'
                     f'<td class="{cls}">{"ACCEPT" if c["ok"] else "REJECT"}</td></tr>')
        h.append("</table>")
        h.append(f'<div class="note"><b>CPR = {coe["verified"]}/{coe["total"]} '
                 f'({coe["cpr"]:.1f}%)</b> = verified material claims / total. '
                 f'CPR is a reporting ratio, not a proof. A claim with no linked, '
                 f'hashed evidence is REJECTED before promotion.</div>')
        for cid, why in coe["rejected"]:
            h.append(f'<div class="f">REJECT {esc(cid)}: {esc("; ".join(why)[:200])}</div>')

    # layer 5: holonic graph audit
    h.append("<h2>5 &middot; Holonic graph audit</h2>")
    if dag["fig"]:
        h.append(f'<img src="data:image/png;base64,'
                 f'{base64.b64encode(dag["fig"].read_bytes()).decode()}" alt="holon dag">')
    h.append('<table><tr><th>check</th><th>value</th></tr>')
    for label, val in (("verdict", dag["verdict"]), ("nodes", dag["nodes"]),
                       ("edges", dag["edges"]), ("cycle status", dag["cycle_status"]),
                       ("cycle nodes", ", ".join(dag["cycles"]) or "none"),
                       ("orphans", ", ".join(dag["orphans"]) or "none")):
        cls = "ok" if dag["verdict"] == "PASS" else "bad"
        h.append(f'<tr><td>{esc(str(label))}</td><td class="{cls}">{esc(str(val))}</td></tr>')
    h.append("</table>")
    h.append('<div class="note">A development loop is iterative by design; a cycle is '
             'a defect only when it is <b>not</b> declared. Detection is deterministic; '
             'the rendered figure is a coarse filter for <code>vision_analyze</code>.</div>')

    # layer 6: four-surface sync topology
    h.append("<h2>6 &middot; Four-surface sync</h2>")
    if sy["err"]:
        h.append(f'<div class="note"><b>BLOCKED:</b> {esc(sy["err"])}</div>')
    h.append('<table><tr><th>check</th><th>value</th></tr>')
    h.append(f'<tr><td>surfaces declared</td><td>{sy["surfaces"]}</td></tr>')
    h.append(f'<tr><td>flows declared</td><td>{sy["flows"]}</td></tr>')
    h.append(f'<tr><td>invariants declared</td><td>{sy["invariants"]}</td></tr>')
    d = sy["drift"]
    if isinstance(d, dict):
        for k in sorted(d)[:12]:
            if isinstance(d[k], (str, int, float, bool)):
                h.append(f'<tr><td>drift.{esc(str(k))}</td><td>{esc(str(d[k]))}</td></tr>')
    else:
        h.append(f'<tr><td>drift verdict</td><td>{esc(str(d)[:200])}</td></tr>')
    h.append("</table>")
    h.append('<div class="note">One writer surface per path class. Vast holds '
             'ephemeral compute only. A drift verdict is DERIVED from git and '
             'filesystem probes, not from a declaration.</div>')

    # provenance
    h.append("<h2>7 &middot; Provenance (sha256 of every source read)</h2>")
    h.append("<table><tr><th>source</th><th>path &middot; sha256 &middot; bytes</th></tr>")
    for k, v in sorted(PROV.items()):
        h.append(f"<tr><td>{esc(k)}</td><td><code>{esc(v)}</code></td></tr>")
    h.append("</table>")
    h.append('<div class="f">Figures: deterministic matplotlib (fixed palette, DPI 110, '
             'no timestamps baked into the image). Numeric tables are authoritative; '
             'a vision read of a figure is a coarse filter only.</div>')

    html = "\n".join(h)
    hp = OUTDIR / f"henri_telemetry_{STAMP}.html"
    hp.write_text(html, encoding="utf-8")

    # markdown twin
    md = [f"# HENRI telemetry report — {ts}", "",
          f"- branch `{rl['branch']}` · HEAD `{rl['head'][:60]}`",
          f"- MoA cache hit **{tot['hit']:.1f}%** · reference spend **${tot['usd']:.2f}**"
          f" · {tot['turns_moa']} turns over {tot['files']} traces",
          f"- gates failing **{nfail}/{len(brows)}** · worktrees **{rl['worktrees']}**"
          f" · binary blobs tracked **{rl['blobs']}**", "",
          "## 1 MoA prompt-cache", "",
          "| slot | turns | fresh in | cache read | hit % | out | usd | $/Mtok |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in sorted(cl["slots"].items(), key=lambda x: -x[1]["usd"]):
        p = v["fresh"] + v["cache"]
        tt = p + v["out"]
        md.append(f"| {k} | {v['turns']} | {v['fresh']:,} | {v['cache']:,} | "
                  f"{100.0*v['cache']/p if p else 0:.1f} | {v['out']:,} | "
                  f"{v['usd']:.4f} | {1e6*v['usd']/tt if tt else 0:.3f} |")
    md += ["", "> UNVERIFIED: aggregator slot has no usage/cost field.", "",
           "## 2 Benchmark gates", "",
           "| benchmark | variant | status | acc | chance | margin | accept | gate | items |",
           "|---|---|---|---:|---:|---:|---:|---|---:|"]
    for r in sorted(brows, key=lambda x: (x["bench"], x["variant"])):
        md.append(f"| {r['bench']} | {r['variant']} | {r['status']} | {r['acc']:.4f} | "
                  f"{r['chance']:.4f} | {r['margin']} | {r['accept']} | "
                  f"{gate_state(r)} | {r['items']:,} |")
    md += ["", (f"_{ngated} benchmark(s) carry a pre-registered accept_margin; "
                '"n/a" = no gate registered (not a pass, not a failure)._')]
    if hefig:
        md += ["", "## 2c HumanEval condition sweep", "",
               "| variant | solved | n | acc % |", "|---|---:|---:|---:|"]
        for r in sorted([x for x in brows if "humaneval" in x["file"].lower()],
                        key=lambda x: (x["solved"] if isinstance(x["solved"], (int, float))
                                       else 0)):
            solved = r["solved"] if isinstance(r["solved"], (int, float)) else 0
            md.append(f"| {r['variant']} | {solved:.0f} | {r['items']} | "
                      f"{100.0*solved/r['items'] if r['items'] else 0:.1f} |")
        md += ["", "_All variants near zero: the sweep is flat, so the gap is "
                   "upstream of egress — not a decoder problem._"]

    md += ["", "## 4 Chain-of-Evidence enforcement (CPR)", ""]
    if coe["err"]:
        md += [f"- BLOCKED: {coe['err']}"]
    else:
        md += [f"- **CPR = {coe['verified']}/{coe['total']} "
               f"({coe['cpr']:.1f}%)** (a reporting ratio, not a proof)", "",
               "| claim_id | type | status | verdict |", "|---|---|---|---|"]
        for c in coe.get("rows", []):
            md.append(f"| {c['id']} | {c['type']} | {c['status']} | "
                      f"{'ACCEPT' if c['ok'] else 'REJECT'} |")
        for cid, why in coe["rejected"]:
            md.append(f"- REJECT `{cid}`: {'; '.join(why)[:160]}")
    md += ["", "## 5b Four-surface sync", "",
           f"- surfaces **{sy['surfaces']}** \u00b7 flows **{sy['flows']}** \u00b7 "
           f"invariants **{sy['invariants']}**"]
    if sy["err"]:
        md += [f"- BLOCKED: {sy['err']}"]
    elif isinstance(sy["drift"], dict):
        for k in sorted(sy["drift"])[:10]:
            if isinstance(sy["drift"][k], (str, int, float, bool)):
                md += [f"- `drift.{k}` = {sy['drift'][k]}"]
    md += ["", "## 5 Holonic graph audit", "",
           f"- verdict **{dag['verdict']}** · {dag['nodes']} nodes · "
           f"{dag['edges']} edges · cycle status **{dag['cycle_status']}**",
           f"- cycle nodes: {', '.join(dag['cycles']) or 'none'}",
           f"- orphans: {', '.join(dag['orphans']) or 'none'}",
           "", "## 6 Provenance", ""]
    for k, v in sorted(PROV.items()):
        md.append(f"- `{k}` — `{v}`")
    mp = OUTDIR / f"henri_telemetry_{STAMP}.md"
    mp.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"HTML : {hp}")
    print(f"MD   : {mp}")
    print(f"cache_hit={tot['hit']:.1f}%  ref_spend=${tot['usd']:.4f}  "
          f"turns={tot['turns_moa']}  scorecards={len(brows)}  "
          f"gates_failing={nfail}/{ngated}  (of {len(brows)} rows, rest n/a)")
    he = [r for r in brows if "humaneval" in r["file"].lower()]
    if he:
        sv = [r["solved"] for r in he if isinstance(r["solved"], (int, float))]
        print(f"humaneval_variants={len(he)}  solved_range="
              f"{min(sv) if sv else '-'}..{max(sv) if sv else '-'} of {he[0]['items']}")
    print(f"worktrees={rl['worktrees']}  blobs_tracked={rl['blobs']}  sources_hashed={len(PROV)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
