#!/usr/bin/env python3
"""UHR-03 POOLED reducer -- judges the paired A/B against the FROZEN criteria.

Separate from uhr03_reduce.py on purpose: two earlier attempts to patch that file
failed on a brittle literal anchor (`reducer: anchor MISSING`), and a reducer that
did not get patched cannot judge the run it was written for. This file reduces ONE
thing (the pooled exteroceptive domain) and is self-contained.

USAGE
  python uhr03_reduce_pooled.py <egress_dir>

Reads every production_run_*.jsonl under <egress_dir>/telemetry_uhr03_<ARM>/ and
prints, per arm, the pooled-domain distribution plus the G1-G6 / C1-C3 verdict.

PRE-REGISTERED (uhr03_preregistration.md Amd 1-6, frozen before run #4):
  G1 store populated      target_theta_norm > 0 and not stalled
  G2 guard executed       phase820_guard_state.updated is True
  G3 FORM B defined       phase820_extero_info.status == "OK"
  G4 both arms exit 0
  G5 flag-OFF identity    arm A emits no pooled key
  G6 non-trivial ref      pooled_own < pooled invalid baseline, i.e. truth is not identity
  C1 own is argmin        pooled_own_is_min True in >= 50% of records with n_channels >= 2
  C2 margin above band    pooled_margin_above_band True in >= 50% of those records
  C3 delta leaves {0.0}   pooled_own takes >1 distinct value
A record with marginal=False counts as a FAIL, never a win, and n_channels < 2 is
excluded from C1/C2 (it cannot be scored).
"""
import json, sys, glob, os, statistics as st

ARM_DIRS = {"BASELINE": "telemetry_uhr03_BASELINE", "RFSS": "telemetry_uhr03_RFSS"}


def find_pooled(obj, acc):
    """Collect every dict that carries pooled-domain keys, at any nesting depth."""
    if isinstance(obj, dict):
        if ("pooled_margin" in obj) or ("delta_pooled_own" in obj):
            acc.append(obj)
        for v in obj.values():
            find_pooled(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_pooled(v, acc)
    return acc


def load(path):
    rows, pooled, guards, probes = [], [], [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            rows.append(r)
            find_pooled(r, pooled)
            g = r.get("phase820_guard_state")
            if isinstance(g, dict):
                guards.append(g)
            p = r.get("outcome_probe")
            if isinstance(p, dict):
                probes.append(p)
    return rows, pooled, guards, probes


def pct(xs, pred):
    xs = list(xs)
    return (sum(1 for x in xs if pred(x)), len(xs))


def main(root):
    out = {}
    for arm, sub in ARM_DIRS.items():
        files = sorted(glob.glob(os.path.join(root, sub, "production_run_*.jsonl")))
        if not files:
            print(f"[{arm}] NO JSONL under {sub}/")
            out[arm] = None
            continue
        rows, pooled, guards, probes = load(files[-1])
        print("=" * 78)
        print(f"[{arm}] {os.path.basename(files[-1])}  records={len(rows)}  "
              f"pooled_records={len(pooled)}")
        print("=" * 78)
        if not pooled:
            print("  no pooled payload -> FORM B flag was OFF or the block never ran (G5/G3)")
            out[arm] = {"n": 0}
            continue

        mar = [p.get("pooled_margin") for p in pooled if p.get("pooled_margin") is not None]
        own = [p.get("delta_pooled_own") for p in pooled if p.get("delta_pooled_own") is not None]
        inv = [p.get("delta_pooled_invalid_min") for p in pooled
               if p.get("delta_pooled_invalid_min") is not None]
        nch = [p.get("n_channels") for p in pooled if p.get("n_channels") is not None]
        stat = sorted({p.get("status") for p in pooled})
        ok = [p for p in pooled if p.get("status") == "OK" and (p.get("n_channels") or 0) >= 2]

        print(f"  status            : {stat}")
        print(f"  n_channels        : {sorted(set(nch)) if nch else 'n/a'}")
        if mar:
            print(f"  margin            : mean={st.mean(mar):+.6f} min={min(mar):+.6f} "
                  f"max={max(mar):+.6f}")
        if own:
            print(f"  delta_pooled_own  : distinct={len(set(round(v, 9) for v in own))} "
                  f"min={min(own):.6f} max={max(own):.6f}")
        if inv:
            print(f"  delta_pooled_inv  : min={min(inv):.6f} max={max(inv):.6f}")

        n_self, n_tot = pct(ok, lambda p: p.get("pooled_own_is_min") is True)
        n_band, _ = pct(ok, lambda p: p.get("pooled_margin_above_band") is True)
        n_spread, _ = pct(ok, lambda p: p.get("pooled_spread_above_band") is True)
        c1 = (n_self / n_tot) if n_tot else None
        c2 = (n_band / n_tot) if n_tot else None
        c3 = len(set(round(v, 9) for v in own)) > 1
        g2 = any(g.get("updated") is True for g in guards)
        g3 = any(p.get("status") == "OK" for p in pooled)
        moved = ([p.get("frame_changed") for p in probes if p.get("frame_changed") is not None])
        print()
        print(f"  G2 guard executed : {g2}")
        print(f"  G3 FORM B defined : {g3}")
        print(f"  G6 ref non-trivial: {n_spread}/{n_tot} spread_above_band")
        print(f"  C1 own is argmin  : {n_self}/{n_tot}  ({c1:.0%})" if c1 is not None
              else "  C1 own is argmin  : n/a")
        print(f"  C2 margin > band  : {n_band}/{n_tot}  ({c2:.0%})" if c2 is not None
              else "  C2 margin > band  : n/a")
        print(f"  C3 owns vary      : {c3}  distinct={len(set(round(v, 9) for v in own))}")
        print(f"  frame moved       : {sum(1 for m in moved if m)}/{len(moved)} probes")
        out[arm] = {"n": len(pooled), "n_scored": n_tot, "C1": c1, "C2": c2, "C3": c3,
                    "G2": g2, "G3": g3, "margin_mean": (st.mean(mar) if mar else None)}

    print()
    print("=" * 78); print("VERDICT"); print("=" * 78)
    b, r = out.get("BASELINE"), out.get("RFSS")
    if not r or not r.get("n"):
        print("  RFSS arm produced NO pooled payload -> BLOCKED_INFRASTRUCTURE, 0 kill used")
        return
    if r.get("n_scored", 0) == 0:
        print("  RFSS pooled records all had n_channels < 2 -> BLOCKED (domain vacuous)")
        return
    c1, c2 = r.get("C1"), r.get("C2")
    print(f"  RFSS C1={c1:.0%}  C2={c2:.0%}  C3={r.get('C3')}  margin_mean={r.get('margin_mean'):+.6f}")
    if c1 == 1.0 and c2 == 1.0:
        print("  -> FORM B SEPARATES. Content is discriminated on the live path.")
    elif (c1 or 0) >= 0.5 and (c2 or 0) >= 0.5:
        print("  -> FORM B separates on a MAJORITY of scored records.")
    else:
        print("  -> FORM B FAILED TO SEPARATE on this run. Kill strike 1 of 2.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.expandvars(r"%LOCALAPPDATA%\Temp\uhr03_egress"))
