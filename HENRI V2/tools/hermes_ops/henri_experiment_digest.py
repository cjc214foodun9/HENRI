#!/usr/bin/env python3
"""henri_experiment_digest.py — compact scorecard digest + optional render.

Replaces inline reading of multi-MB benchmark scorecards. A human or LLM needs
the headline row and a figure, not 196k JSON lines.

Evidence class: DERIVED from OBSERVED scorecard fields. No LLM, no network.

Usage:
  python henri_experiment_digest.py --root "<repo>/experiments/verification"
  python henri_experiment_digest.py --root ... --figure out.png
  python henri_experiment_digest.py --root ... --subjects   # per-subject breakdown
  python henri_experiment_digest.py --root ... --md out.md --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HEADLINE = [
    "benchmark", "status", "verdict", "egress_path", "device",
    "item_count", "attempted", "correct", "solved", "accuracy",
    "accuracy_attempted", "chance", "margin", "accept_margin",
    "checkpoint_used", "wall_clock_sec", "avg_latency_ms_item", "timestamp_utc",
]


def load_one(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:  # malformed / truncated scorecard
        print(f"  SKIP {path.name}: {type(e).__name__}", file=sys.stderr)
        return None
    return d if isinstance(d, dict) else None


def row_of(d: dict, path: Path) -> dict:
    r = {"file": path.name}
    for k in HEADLINE:
        if k in d:
            v = d[k]
            if k == "commit" and isinstance(v, str):
                v = v[:10]
            if k == "dataset_sha256" and isinstance(v, str):
                v = v[:12]
            r[k] = v
    r["_bytes"] = path.stat().st_size
    ir = d.get("item_results")
    r["_items"] = len(ir) if isinstance(ir, list) else 0
    # keep the raw body out of the digest on purpose
    return r


def subjects_of(d: dict) -> list[tuple[str, int, int]]:
    """(subject, n, correct) from MMLU-style item_results."""
    ir = d.get("item_results")
    if not isinstance(ir, list):
        return []
    acc: dict[str, list[int]] = {}
    for it in ir:
        if not isinstance(it, dict):
            continue
        s = it.get("subject")
        if s is None:
            continue
        a = acc.setdefault(str(s), [0, 0])
        a[0] += 1
        a[1] += 1 if it.get("is_correct") else 0
    return sorted(((s, v[0], v[1]) for s, v in acc.items()),
                  key=lambda x: (x[2] / x[1] if x[1] else 0), reverse=True)


def render(rows: list[dict], out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("RENDER BLOCKED: matplotlib not importable in this interpreter.")
        print("  fix: uv venv <dir> && uv pip install matplotlib --python <dir>/Scripts/python.exe")
        return False

    usable = [r for r in rows if isinstance(r.get("accuracy"), (int, float))]
    if not usable:
        print("RENDER BLOCKED: no scorecard with a numeric 'accuracy' field.")
        return False

    names = [str(r["benchmark"] or r["file"])[:18] for r in usable]
    accs = [float(r["accuracy"]) for r in usable]
    chances = [float(r.get("chance") or 0.0) for r in usable]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=110)
    y = range(len(usable))
    axes[0].barh([i + 0.18 for i in y], accs, height=0.36, label="accuracy", color="#4c78a8")
    axes[0].barh([i - 0.18 for i in y], chances, height=0.36, label="chance", color="#b0b0b0")
    axes[0].set_yticks(list(y), names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(1.0, max(accs) * 1.15))
    axes[0].set_title("accuracy vs chance")
    axes[0].legend(fontsize=8)

    mar = [float(r.get("margin") or 0.0) for r in usable]
    acc_m = [float(r.get("accept_margin") or 0.0) for r in usable]
    axes[1].barh([i + 0.18 for i in y], mar, height=0.36, label="margin", color="#f58518")
    axes[1].barh([i - 0.18 for i in y], acc_m, height=0.36, label="accept_margin", color="#54a24b")
    axes[1].set_yticks(list(y), ["" for _ in y])
    axes[1].invert_yaxis()
    axes[1].set_title("margin vs pre-registered accept_margin")
    axes[1].legend(fontsize=8)
    for i, (m, a) in enumerate(zip(mar, acc_m)):
        axes[1].annotate("PASS" if m >= a else "FAIL", (max(m, a), i),
                         fontsize=7, va="center", ha="left")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote figure: {out}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing *_scorecard.json")
    ap.add_argument("--glob", default="*scorecard*.json")
    ap.add_argument("--figure", default=None)
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--subjects", action="store_true", help="print per-subject accuracy")
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    root = Path(a.root)
    files = sorted(root.rglob(a.glob))
    if not files:
        print(f"BLOCKED: no files matching {a.glob} under {root}")
        return 2

    rows, subject_rows = [], []
    for f in files:
        d = load_one(f)
        if d is None:
            continue
        rows.append(row_of(d, f))
        if a.subjects:
            sub = subjects_of(d)
            if sub:
                subject_rows.append((d.get("benchmark", f.name), sub))

    print("=" * 96)
    print(f"EXPERIMENT DIGEST  (DERIVED from OBSERVED scorecards)  root={root}")
    print(f"scorecards={len(rows)}  inline bytes avoided={sum(r['_bytes'] for r in rows):,}")
    print("=" * 96)
    hdr = f"{'benchmark':22s} {'status':10s} {'verdict':10s} {'acc':>7s} {'chance':>7s} " \
          f"{'margin':>7s} {'accept':>7s} {'gate':>5s} {'items':>7s} {'ega':>3s}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: str(x.get("benchmark"))):
        acc = r.get("accuracy")
        acc_s = f"{acc:.4f}" if isinstance(acc, (int, float)) else "-"
        ch = r.get("chance")
        ch_s = f"{ch:.4f}" if isinstance(ch, (int, float)) else "-"
        m, am = r.get("margin"), r.get("accept_margin")
        m_s = f"{m:.4f}" if isinstance(m, (int, float)) else "-"
        am_s = f"{am:.4f}" if isinstance(am, (int, float)) else "-"
        gate = "-"
        if isinstance(m, (int, float)) and isinstance(am, (int, float)):
            gate = "PASS" if m >= am else "FAIL"
        print(f"{str(r.get('benchmark'))[:22]:22s} {str(r.get('status'))[:10]:10s} "
              f"{str(r.get('verdict') or '-')[:10]:10s} {acc_s:>7s} {ch_s:>7s} "
              f"{m_s:>7s} {am_s:>7s} {gate:>5s} {r['_items']:7d} "
              f"{'Y' if r.get('checkpoint_used') else 'N':>3s}")

    if subject_rows:
        for bench, subs in subject_rows:
            print(f"\nPER-SUBJECT (best) — {bench}  [n_subjects={len(subs)}]")
            for s, n, c in subs[: a.top]:
                print(f"   {s:34s} {c:5d}/{n:<5d} {100.0*c/n:6.2f}%")
            print("  ... worst:")
            for s, n, c in subs[-a.top:][::-1]:
                print(f"   {s:34s} {c:5d}/{n:<5d} {100.0*c/n:6.2f}%")

    if a.figure:
        render(rows, Path(a.figure))

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps(
            {"root": str(root), "rows": rows,
             "evidence_class": "DERIVED from OBSERVED scorecard fields"},
            indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {a.json_out}")

    if a.md:
        lines = [f"# Experiment digest", "",
                 f"Root: `{root}` — {len(rows)} scorecards, "
                 f"{sum(r['_bytes'] for r in rows):,} bytes not inlined.", "",
                 "| benchmark | status | acc | chance | margin | accept | gate | items |",
                 "|---|---|---:|---:|---:|---:|---|---:|"]
        for r in sorted(rows, key=lambda x: str(x.get("benchmark"))):
            lines.append("| {b} | {s} | {a} | {c} | {m} | {am} | {g} | {n} |".format(
                b=r.get("benchmark", "-"), s=r.get("status", "-"),
                a=r.get("accuracy", "-"), c=r.get("chance", "-"),
                m=r.get("margin", "-"), am=r.get("accept_margin", "-"),
                g=("PASS" if (isinstance(r.get("margin"), (int, float))
                              and isinstance(r.get("accept_margin"), (int, float))
                              and r["margin"] >= r["accept_margin"]) else "FAIL"),
                n=r["_items"]))
        Path(a.md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {a.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
