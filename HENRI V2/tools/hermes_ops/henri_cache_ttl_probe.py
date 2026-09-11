#!/usr/bin/env python3
"""DECIDE the cache hit-killer: is it TTL eviction or prefix change?

Two competing hypotheses for the 21.7% hit rate:
  H-TTL    : the provider cache expires between turns, so long gaps miss and
             back-to-back turns hit. PREDICTION: hit% correlates NEGATIVELY with
             the wall-clock gap since the previous turn.
  H-PREFIX : the prompt prefix changes between turns (roster/content churn), so
             cache misses are independent of timing. PREDICTION: no correlation.

This prints the per-turn table needed to choose, plus a rank correlation.

If the trace has no timestamps, report BLOCKED rather than guessing.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

TRACE_DIR = Path(r"C:\Users\chan\AppData\Local\hermes\moa-traces")


def slot_short(label: str) -> str:
    return label.split(":", 1)[-1].split("[", 1)[0]


def find_time(rec: dict):
    """Return a float epoch/time-like value if the record carries one."""
    for k in ("ts", "timestamp", "time", "created_at", "start_time", "at"):
        v = rec.get(k)
        if isinstance(v, (int, float)):
            # crude epoch normalisation: seconds vs milliseconds
            return float(v) / (1000.0 if v > 1e11 else 1.0)
        if isinstance(v, str) and len(v) >= 19:
            return v
    return None


def spearman(xs: list[float], ys: list[float]) -> float:
    """Rank correlation without scipy."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main() -> int:
    files = sorted(TRACE_DIR.glob("*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:3]
    if not files:
        print("BLOCKED: no traces")
        return 3

    # ---- schema discovery: does a turn record carry a timestamp? ----
    keysets: dict[tuple, int] = {}
    sample = None
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ks = tuple(sorted(r))
                keysets[ks] = keysets.get(ks, 0) + 1
                if sample is None:
                    sample = r
    print("=== turn-record key-sets ===")
    for ks, n in sorted(keysets.items(), key=lambda kv: -kv[1])[:5]:
        print(f"  x{n:<4} {list(ks)}")
    if sample:
        print(f"\nfirst-record keys: {sorted(sample)}")
        t = find_time(sample)
        print(f"timestamp field detected: {t!r}"
              f"{'  (string form)' if isinstance(t, str) else ''}")

    has_ts = any(find_time(
        json.loads(l)) is not None
        for f in files for l in open(f, encoding="utf-8", errors="replace")
        if l.strip()) if False else None

    # ---- per-turn rows ----
    rows = []
    for f in files:
        prev_t = None
        with open(f, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = find_time(rec)
                gap = None
                if isinstance(t, float) and isinstance(prev_t, float):
                    gap = max(0.0, t - prev_t)
                if isinstance(t, float):
                    prev_t = t
                rd = fr = 0.0
                for r in rec.get("references") or []:
                    u = r.get("usage") or {}
                    rd += float(u.get("cache_read_tokens") or 0)
                    fr += float(u.get("input_tokens") or 0)
                if rd + fr <= 0:
                    continue
                hit = 100 * rd / (rd + fr)
                rows.append((f.name[:18], i, gap, rd, fr, hit))

    print(f"\nturns with usage = {len(rows)}")
    have_gap = [r for r in rows if r[2] is not None]
    print(f"turns with a measurable gap = {len(have_gap)}")
    print()
    print(f"{'trace':18s} {'turn':>4} {'gap_s':>9} {'read':>10} {'fresh':>10} {'hit%':>6}")
    for name, i, gap, rd, fr, hit in rows[:30]:
        g = f"{gap:,.0f}" if gap is not None else "-"
        print(f"{name:18s} {i:>4} {g:>9} {rd:>10,.0f} {fr:>10,.0f} {hit:>5.1f}%")

    if len(have_gap) >= 5:
        gaps = [r[2] for r in have_gap]
        hits = [r[5] for r in have_gap]
        rho = spearman(gaps, hits)
        print(f"\nSpearman(gap, hit%) = {rho:+.3f}  (n={len(gaps)})")
        short = [h for g, h in zip(gaps, hits) if g < st.median(gaps)]
        long_ = [h for g, h in zip(gaps, hits) if g >= st.median(gaps)]
        print(f"  mean hit% for SHORT gaps (<median) = {st.mean(short):.1f}%  (n={len(short)})")
        print(f"  mean hit% for LONG  gaps (>=median) = {st.mean(long_):.1f}%  (n={len(long_)})")
        if rho < -0.3:
            print("  VERDICT: H-TTL SUPPORTED — hits decay with the inter-turn gap. "
                  "Lever = reduce wall-clock gap (batch tool calls, avoid long "
                  "local compute between turns) and keep one session alive.")
        elif abs(rho) < 0.3:
            print("  VERDICT: H-TTL NOT SUPPORTED — gap and hit% are unrelated. "
                  "Lever = prefix stability (roster/ordering/session identity), "
                  "not cadence.")
        else:
            print("  VERDICT: positive correlation is unexplained; inspect manually.")
    else:
        print("\nBLOCKED: too few turns carry a timestamp to decide H-TTL vs "
              "H-PREFIX. Report 'undecided' rather than guessing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
