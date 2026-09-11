#!/usr/bin/env python3
"""M-2 kill experiment: WHICH slot misses, and does the prefix stabilise?

The 21.7% aggregate hit rate could be:
  (a) GLOBAL prefix churn  -> every slot misses together, curves overlap
  (b) PER-SLOT routing     -> some slots hit, others never do
  (c) TTL expiry           -> already FALSIFIED (Spearman -0.118)

Discriminator: per-slot per-turn cache_read on the CURRENT session trace.
If the curves overlap and all collapse together, it is (a) global churn.
If one slot holds a high curve while others stay at 0, it is (b) routing.

Read-only. Prints the aligned matrix so the pattern is visible, not inferred.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

H = Path(r"C:\Users\chan\AppData\Local\hermes")
TRACES = H / "moa-traces"


def short(l):
    return l.split(":", 1)[-1].split("[", 1)[0]


def main() -> int:
    files = sorted(TRACES.glob("*.jsonl"),
                   key=lambda p: p.stat().st_mtime, reverse=True)[:3]
    if not files:
        print("BLOCKED: no traces")
        return 3

    cur = files[0]
    print(f"current-session trace: {cur.name} ({cur.stat().st_size:,} B)")

    # turn -> slot -> (read, fresh)
    turns: dict[int, dict[str, tuple[float, float]]] = {}
    with open(cur, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for r in rec.get("references") or []:
                u = r.get("usage") or {}
                rd = float(u.get("cache_read_tokens") or 0)
                fr = float(u.get("input_tokens") or 0)
                if rd + fr <= 0:
                    continue
                turns.setdefault(i, {})[short(r.get("label") or "?")] = (rd, fr)

    if not turns:
        print("BLOCKED: no usage records in current trace")
        return 3

    slots = sorted({s for t in turns.values() for s in t})
    print(f"\nturns with usage = {len(turns)}   slots = {len(slots)}\n")

    print("=== aligned per-turn matrix (hit% per slot) ===")
    hdr = f"{'turn':>4} " + " ".join(f"{s[:11]:>11s}" for s in slots) + "   turn_hit%"
    print(hdr)
    print("-" * len(hdr))
    for t in sorted(turns):
        row, tr = [], 0.0
        tf = 0.0
        for s in slots:
            rd, fr = turns[t].get(s, (0.0, 0.0))
            row.append(f"{100*rd/(rd+fr):>10.0f}%" if rd + fr else f"{'-':>11s}")
            tr += rd
            tf += fr
        th = f"{100*tr/(tr+tf):>8.1f}%" if tr + tf else "-"
        print(f"{t:>4} " + " ".join(row) + f"   {th}")

    print("\n=== per-slot summary ===")
    for s in slots:
        rs = [turns[t].get(s, (0.0, 0.0)) for t in turns]
        cr = sum(x[0] for x in rs)
        cf = sum(x[1] for x in rs)
        hits = sum(1 for x in rs if x[0] > 0)
        print(f"  {s[:22]:22s} read={cr:>10,.0f} fresh={cf:>10,.0f} "
              f"hit={100*cr/(cr+cf) if cr+cf else 0:>5.1f}%  "
              f"turns_with_hit={hits}/{len(turns)}")

    # does the whole turn move together? (global churn) or split? (routing)
    print("\n=== pattern test ===")
    both = [t for t in turns if len(turns[t]) >= 2]
    if not both:
        print("  insufficient multi-slot turns to discriminate")
        return 0
    together = split = 0
    for t in both:
        vals = [1 if turns[t][s][0] > 0 else 0 for s in turns[t]]
        if all(vals) or not any(vals):
            together += 1
        else:
            split += 1
    print(f"  turns where ALL slots agree (hit or all miss) = {together}")
    print(f"  turns where slots DISAGREE                    = {split}")
    if split > together:
        print("  VERDICT: PER-SLOT ROUTING. Slots behave independently, so a single")
        print("  global protocol cannot reach 90%; each slot needs its own prefix.")
    elif together:
        print("  VERDICT: GLOBAL CHURN. Slots move together, so the whole prompt")
        print("  prefix is being invalidated - a stable-prefix protocol addresses it.")
    else:
        print("  VERDICT: undecided")

    # how many turns are the read=0 -> some hit -> read=0 sawtooth?
    seq = [1 if sum(turns[t][s][0] for s in turns[t]) > 0 else 0 for t in sorted(turns)]
    flips = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
    print(f"\n  hit-present sequence: {''.join(str(x) for x in seq)}")
    print(f"  flips = {flips} over {len(seq)} turns "
          f"({100*flips/len(seq):.0f}% of turns change state)")
    print("  a sawtooth (many flips) = the prefix is rebuilt, not reused")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
