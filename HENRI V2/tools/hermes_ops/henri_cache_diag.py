#!/usr/bin/env python3
"""Cache-hit diagnosis + arithmetic ceiling. Read-only.

WHY: the aggregate MoA prompt-cache hit rate is 25.5% against a p50 prompt of
~137k tokens. Before promising >90% this measures the ACTUAL ceiling and
locates the hit-killer.

Schema (moa-traces/*.jsonl, one JSON object per line = one MoA turn):
  rec["references"] = [{"label": "openrouter:z-ai/glm-5.3-flash[reasoning=high]",
                        "usage": {"input_tokens": N, "cache_read_tokens": N,
                                  "cache_write_tokens": N, "output_tokens": N,
                                  "reasoning_tokens": N},
                        "cost_usd": F}, ...]

Two denominators are reported because the choice changes the headline number:
  fresh  = input_tokens        uncached prompt tokens
  write  = cache_write_tokens  prefix written to cache (billed at a premium)
  read   = cache_read_tokens   served from cache
  prompt = fresh + write + read
  hit_A  = read / (read + fresh)     convention used by henri_cache_audit.py
  hit_B  = read / prompt             strict share of all prompt tokens

Ceiling model: hit_B <= 1 - irreducible/prompt, where irreducible is the
minimum (fresh+write) observed for that slot, i.e. the least NEW content the
slot ever had to process. If that floor is large relative to the prompt, >90%
is arithmetically unreachable and the honest answer is "not reachable at this
prompt composition" rather than a protocol that games the metric.

Usage: python henri_cache_diag.py [--newest 3] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

TRACE_DIR = Path(r"C:\Users\chan\AppData\Local\hermes\moa-traces")


def slot_short(label: str) -> str:
    s = label.split(":", 1)[-1]
    return s.split("[", 1)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--newest", type=int, default=3)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    files = sorted(TRACE_DIR.glob("*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[: a.newest]
    if not files:
        print(f"BLOCKED: no .jsonl traces under {TRACE_DIR}")
        return 3
    print(f"traces examined: {len(files)}")
    for f in files:
        print(f"  {f.name}  ({f.stat().st_size:,} B)")

    # slot -> counters ; slot -> per-turn-index hit_B samples
    S: dict[str, dict] = defaultdict(lambda: {"read": 0, "fresh": 0, "write": 0,
                                              "out": 0, "reason": 0, "n": 0,
                                              "usd": 0.0})
    byturn: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    turn_prompt: list[float] = []
    turn_new: list[float] = []
    n_turns = 0
    used_metric = defaultdict(int)

    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_turns += 1
                for r in rec.get("references") or []:
                    lab = slot_short(r.get("label") or "?")
                    u = r.get("usage") or {}
                    if not u:
                        used_metric[f"no-usage:{lab}"] += 1
                        continue
                    fresh = float(u.get("input_tokens") or 0)
                    rd = float(u.get("cache_read_tokens") or 0)
                    wr = float(u.get("cache_write_tokens") or 0)
                    if fresh + rd + wr <= 0:
                        continue
                    g = S[lab]
                    g["read"] += rd
                    g["fresh"] += fresh
                    g["write"] += wr
                    g["out"] += float(u.get("output_tokens") or 0)
                    g["reason"] += float(u.get("reasoning_tokens") or 0)
                    g["n"] += 1
                    try:
                        g["usd"] += float(r.get("cost_usd") or 0.0)
                    except (TypeError, ValueError):
                        pass
                    p = fresh + rd + wr
                    byturn[lab][min(i, 24)].append(100 * rd / p)
                    turn_prompt.append(p)
                    turn_new.append(fresh + wr)

    print(f"\nturns parsed with usage: {n_turns}")
    if used_metric:
        print(f"turns WITHOUT a usage block: {dict(used_metric)}")

    hdr = (f"{'slot':26s} {'n':>4} {'read':>12} {'fresh':>12} {'write':>11} "
           f"{'hitA%':>6} {'hitB%':>6} {'avg_out':>8} {'$':>8} {'$/Mtok':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    T = defaultdict(float)
    order = sorted(S.items(), key=lambda kv: -(kv[1]["read"] + kv[1]["fresh"]))
    for lab, g in order:
        p = g["fresh"] + g["read"] + g["write"]
        hitA = 100 * g["read"] / (g["read"] + g["fresh"]) if g["read"] + g["fresh"] else 0
        hitB = 100 * g["read"] / p if p else 0
        print(f"{lab[:26]:26s} {g['n']:>4} {g['read']:>12,.0f} {g['fresh']:>12,.0f} "
              f"{g['write']:>11,.0f} {hitA:>5.1f}% {hitB:>5.1f}% "
              f"{g['out']/max(g['n'],1):>8,.0f} {g['usd']:>8.2f} "
              f"{(g['usd']/ (p/1e6)) if p else 0:>7.3f}")
        for k in ("read", "fresh", "write", "out", "usd"):
            T[k] += g[k]
    p = T["fresh"] + T["read"] + T["write"]
    hitA = 100 * T["read"] / (T["read"] + T["fresh"]) if T["read"] + T["fresh"] else 0
    hitB = 100 * T["read"] / p if p else 0
    print("-" * len(hdr))
    print(f"{'TOTAL':26s} {'':>4} {T['read']:>12,.0f} {T['fresh']:>12,.0f} "
          f"{T['write']:>11,.0f} {hitA:>5.1f}% {hitB:>5.1f}% {'':>8} {T['usd']:>8.2f} "
          f"{(T['usd']/(p/1e6)) if p else 0:>7.3f}")

    print("\n--- hit_B% by turn index (prefix-stability test) ---")
    print("  a rising curve = prefix reuses; a flat low curve = prefix is broken "
          "every turn")
    for lab, d in sorted(byturn.items(), key=lambda kv: -len(kv[1]))[:6]:
        pts = ", ".join(f"t{k}={st.mean(v):.0f}" for k, v in sorted(d.items())[:10])
        print(f"  {lab[:24]:24s} {pts}")

    print("\n--- arithmetic ceiling (per slot) ---")
    ceiling = {}
    for lab, g in order:
        p_tot = g["fresh"] + g["read"] + g["write"]
        if not p_tot:
            continue
        new_min = g.get("_min", None)
        # per-turn floors
        floors = [min(x) for x in [[1.0]]]  # placeholder, replaced below
        ceiling[lab] = None
    # compute using the raw per-turn arrays
    raw_new: dict[str, list[float]] = defaultdict(list)
    raw_prompt: dict[str, list[float]] = defaultdict(list)
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for r in rec.get("references") or []:
                    lab = slot_short(r.get("label") or "?")
                    u = r.get("usage") or {}
                    fresh = float(u.get("input_tokens") or 0)
                    rd = float(u.get("cache_read_tokens") or 0)
                    wr = float(u.get("cache_write_tokens") or 0)
                    if fresh + rd + wr <= 0:
                        continue
                    raw_new[lab].append(fresh + wr)
                    raw_prompt[lab].append(fresh + rd + wr)
    for lab in sorted(raw_prompt, key=lambda l: -st.median(raw_prompt[l]))[:8]:
        news, proms = raw_new[lab], raw_prompt[lab]
        floor = min(news)
        med = st.median(proms)
        ceil = 100 * (1 - floor / med) if med else 0
        print(f"  {lab[:24]:24s} floor_new={floor:>10,.0f}  median_prompt={med:>10,.0f}"
              f"  -> ceiling {ceil:>5.1f}%")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {k: dict(v) for k, v in S.items()}, indent=2, default=str),
            encoding="utf-8")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
