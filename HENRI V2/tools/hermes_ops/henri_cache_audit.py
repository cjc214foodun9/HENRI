#!/usr/bin/env python3
"""henri_cache_audit.py — deterministic MoA prompt-cache audit.

Reads MoA trace JSONL and reports per-slot cache economics.
Evidence class: DERIVED from OBSERVED per-slot usage fields.
Writes no file unless --json is given. No LLM, no network, no matplotlib.

Usage:
  python henri_cache_audit.py                 # all traces
  python henri_cache_audit.py --newest 8      # newest N trace files
  python henri_cache_audit.py --json out.json # also write machine artifact
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from pathlib import Path

TRACE_DIR = Path(os.path.expandvars(r"%LOCALAPPDATA%\hermes\moa-traces"))


def slot_short(label: str) -> str:
    """'openrouter:z-ai/glm-5.3-flash[reasoning=high]' -> 'z-ai/glm-5.3-flash'"""
    s = label.split(":", 1)[-1]
    return s.split("[", 1)[0]


def load(files):
    """Yield (file, turn_index, record)."""
    for f in files:
        n = 0
        for line in open(f, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            yield f, n, rec


def audit(files):
    tot = collections.Counter()
    slots = collections.defaultdict(collections.Counter)
    first_turn = collections.Counter()   # cache reads on the FIRST fanout of a file
    per_file_turns = collections.Counter()
    prompt_lens = []
    reason_tokens = collections.Counter()
    unverified = collections.Counter()

    for f, turn, rec in load(files):
        per_file_turns[f] += 1
        for r in rec.get("references") or []:
            lab = slot_short(r.get("label", "?"))
            u = r.get("usage") or {}
            fresh = int(u.get("input_tokens") or 0)
            rd = int(u.get("cache_read_tokens") or 0)
            wr = int(u.get("cache_write_tokens") or 0)
            out = int(u.get("output_tokens") or 0)
            rea = int(u.get("reasoning_tokens") or 0)
            s = slots[lab]
            s["turns"] += 1
            s["fresh_in"] += fresh
            s["cache_rd"] += rd
            s["cache_wr"] += wr
            s["out"] += out
            s["reason"] += rea
            reason_tokens[lab] += rea
            try:
                s["usd"] += float(r.get("cost_usd") or 0.0)
            except (TypeError, ValueError):
                pass
            tot["fresh_in"] += fresh
            tot["cache_rd"] += rd
            tot["cache_wr"] += wr
            tot["out"] += out
            tot["reason"] += rea
            # prompt size actually sent (all message chars)
            chars = sum(len(m.get("content") or "") for m in (r.get("input_messages") or []))
            prompt_lens.append(chars // 4)
            if turn == 1:
                first_turn[lab] += rd
            if rd == 0:
                unverified[lab] += 1
        agg = rec.get("aggregator") or {}
        if isinstance(agg, dict) and "usage" not in agg:
            unverified["aggregator:usage_field_absent"] += 1
    return tot, slots, first_turn, per_file_turns, prompt_lens, unverified


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--newest", type=int, default=0, help="only the N newest trace files")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--trace-dir", default=str(TRACE_DIR))
    a = ap.parse_args()

    tdir = Path(a.trace_dir)
    files = sorted(glob.glob(str(tdir / "*.jsonl")), key=os.path.getmtime)
    if not files:
        print(f"BLOCKED: no trace JSONL under {tdir}")
        return 2
    if a.newest:
        files = files[-a.newest:]

    tot, slots, first_turn, pft, plens, unver = audit(files)
    prompt = tot["fresh_in"] + tot["cache_rd"]
    hit = 100.0 * tot["cache_rd"] / prompt if prompt else 0.0

    print("=" * 78)
    print("MoA PROMPT-CACHE AUDIT  (DERIVED from OBSERVED per-slot usage)")
    print(f"traces={len(files)}  turns={sum(pft.values())}  dir={tdir}")
    print("=" * 78)
    print(f"{'slot':30s} {'turns':>5s} {'fresh_in':>10s} {'cache_rd':>10s} {'hit%':>6s} "
          f"{'out':>8s} {'reason':>8s} {'usd':>8s}")
    for k, v in sorted(slots.items(), key=lambda x: -(x[1]['fresh_in'] + x[1]['cache_rd'])):
        p = v["fresh_in"] + v["cache_rd"]
        h = 100.0 * v["cache_rd"] / p if p else 0.0
        print(f"{k:30s} {v['turns']:5d} {v['fresh_in']:10d} {v['cache_rd']:10d} {h:6.1f} "
              f"{v['out']:8d} {v['reason']:8d} {v['usd']:8.4f}")
    print("-" * 78)
    print(f"{'TOTAL':30s} {sum(v['turns'] for v in slots.values()):5d} {tot['fresh_in']:10d} "
          f"{tot['cache_rd']:10d} {hit:6.1f} {tot['out']:8d} {tot['reason']:8d} "
          f"{sum(v['usd'] for v in slots.values()):8.4f}")

    if plens:
        plens_sorted = sorted(plens)
        print(f"\nprompt size est (tokens): min={plens_sorted[0]} "
              f"p50={plens_sorted[len(plens_sorted)//2]} max={plens_sorted[-1]} "
              f"(n={len(plens)})  [sum of message chars / 4]")

    print("\nFIRST-FANOUT CACHE READS (turn 1 of each trace; structural writes, not hits):")
    for k, v in sorted(first_turn.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v:10d}")

    if unver:
        print("\nVERIFICATION GAPS:")
        for k, v in sorted(unver.items(), key=lambda x: -x[1]):
            print(f"  {k:34s} {v}d")
    agg_usage = tot.get("agg_usage", 0)
    print(f"\n  aggregator usage fields present: {'YES' if agg_usage else 'NO'}"
          "  -> aggregator cache accounting is UNVERIFIED on this install")

    if a.json_out:
        Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json_out).write_text(json.dumps({
            "trace_files": len(files),
            "turns": sum(pft.values()),
            "total_fresh_input_tokens": tot["fresh_in"],
            "total_cache_read_tokens": tot["cache_rd"],
            "total_cache_write_tokens": tot["cache_wr"],
            "hit_rate_pct": round(hit, 2),
            "slots": {k: dict(v) for k, v in slots.items()},
            "prompt_tokens_min": plens_sorted[0] if plens else None,
            "prompt_tokens_p50": plens_sorted[len(plens_sorted) // 2] if plens else None,
            "prompt_tokens_max": plens_sorted[-1] if plens else None,
            "evidence_class": "DERIVED from OBSERVED per-slot usage fields",
        }, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
