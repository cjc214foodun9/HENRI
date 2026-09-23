#!/usr/bin/env python3
"""UHR-03 reducer — None-safe, pin-and-print. Judged ONLY against the frozen
pre-registration (uhr03_preregistration.md + Amendments 1-3).

DEFECT CLASS THIS FILE IS WRITTEN AGAINST: my earlier reducer did
`sorted({...})` on a set of nested-JSON values that included None, and died with
`TypeError: '<' not supported between instances of 'int' and 'NoneType'`. Because
it died BEFORE printing FORM B, the mechanism's own output was hidden twice. Every
sort here uses a None-safe key, every section prints a count, and nothing is
computed from a field that did not appear.
"""
import json, glob, os, pathlib, re, sys, collections

D = pathlib.Path(os.path.expandvars(r"%LOCALAPPDATA%\Temp\uhr03_egress"))


def nkey(v):
    return (v is None, 0.0 if v is None else float(v))


def load(arm):
    fs = sorted(glob.glob(str(D / f"telemetry_uhr03_{arm}" / "production_run_*.jsonl")))
    if not fs:
        return [], None
    f = max(fs, key=os.path.getmtime)
    rows, bad = [], 0
    for line in open(f, encoding="utf-8", errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            bad += 1
    return rows, (pathlib.Path(f).name, len(rows), bad)


def dist(rows, extract):
    c = collections.Counter()
    for r in rows:
        v = extract(r)
        key = "None" if v is None else (round(v, 6) if isinstance(v, float) else v)
        c[key] += 1
    return dict(sorted(c.items(), key=lambda kv: str(kv[0])))


def main():
    print("=" * 78)
    print("UHR-03 REDUCTION — frozen criteria: tau=0.3500, min_norm=1e-5, ft09-0d8bbf25")
    print("=" * 78)
    for arm in ("BASELINE", "RFSS"):
        rows, meta = load(arm)
        print(f"\n########## ARM {arm} ##########")
        if meta is None:
            print("  NO JSONL FOUND")
            continue
        name, n, bad = meta
        print(f"  file={name}  rows={n}  malformed={bad}")

        # G-witnesses
        moved = sum(1 for r in rows if (r.get("outcome_probe") or {}).get("frame_changed"))
        changed = sum(1 for r in rows
                      if ((r.get("outcome_probe") or {}).get("changed_cells") or 0) > 0)
        g = [(r.get("phase820_guard_state") or {}) for r in rows]
        gupd = sum(1 for x in g if x.get("updated"))
        print(f"  MOVEMENT  frame_changed={moved}/{n}  changed_cells>0={changed}/{n}")
        print(f"  G2 guard.updated_true={gupd}/{n}")
        print(f"  guard distinct={dist(rows, lambda r: (r.get('phase820_guard_state') or {}).get('updated'))}")

        # G1: the recorded transition must be NON-TRIVIAL
        th = [ (r.get("phase820_update_info") or {}).get("target_theta_norm") for r in rows ]
        thv = sorted({v for v in th if v is not None}, key=nkey)
        print(f"  G1 target_theta_norm: n_present={sum(1 for v in th if v is not None)}"
              f"/{n}  distinct={[round(v,6) for v in thv[:8]]}")
        stalled = {(r.get("phase820_update_info") or {}).get("stalled") for r in rows}
        print(f"  G1 stalled distinct={sorted(stalled, key=str)}")

        # G3/G4: FORM B (default OFF in BASELINE by construction)
        e = [(r.get("phase820_extero_info") or {}) for r in rows]
        e = [x for x in e if x]
        print(f"  G3 extero_info records={len(e)}  status={dist(rows, lambda r: (r.get('phase820_extero_info') or {}).get('status'))}")
        if e:
            print(f"     delta_pred   ={[round(v,6) for v in sorted({x.get('delta_pred') for x in e if x.get('delta_pred') is not None}, key=nkey)][:8]}")
            print(f"     delta_extero ={[round(v,6) for v in sorted({x.get('delta_extero') for x in e if x.get('delta_extero') is not None}, key=nkey)][:8]}")
            print(f"     argmin_hits_truth={dist(rows, lambda r: (r.get('phase820_extero_info') or {}).get('argmin_hits_truth'))}")
            print(f"     n_recorded  ={dist(rows, lambda r: (r.get('phase820_extero_info') or {}).get('n_recorded'))}")
            print(f"     invalid_minus_own={[round(v,6) for v in sorted({x.get('invalid_minus_own') for x in e if x.get('invalid_minus_own') is not None}, key=nkey)][:8]}")
            print(f"     magnitude_only_risk={dist(rows, lambda r: (r.get('phase820_extero_info') or {}).get('magnitude_only_risk'))}")
            print(f"     truth_gen_frobenius={[round(v,6) for v in sorted({x.get('truth_gen_frobenius') for x in e if x.get('truth_gen_frobenius') is not None}, key=nkey)][:8]}")
            print(f"     nontrivial_transition={dist(rows, lambda r: (r.get('phase820_extero_info') or {}).get('nontrivial_transition'))}")
            print(f"     role_content={dist(rows, lambda r: (r.get('phase820_extero_info') or {}).get('role_content'))}")

        # C1 / C2 computed on QUALIFYING records only (n_recorded >= 2)
        ok = [x for x in e if (x.get("n_recorded") or 0) >= 2 and x.get("status") == "OK"]
        print(f"  --- C-criteria on qualifying records (status=OK and n_recorded>=2): n={len(ok)} ---")
        if ok:
            c1 = sum(1 for x in ok if (x.get("argmin_hits_truth") or 0) >= 1)
            c2 = sum(1 for x in ok if (x.get("invalid_minus_own") or 0) > 0)
            print(f"  C1 argmin_hits_truth>=1 : {c1}/{len(ok)}  ({100.0*c1/len(ok):.1f}%)  PASS>=50%: {c1*2 >= len(ok)}")
            print(f"  C2 invalid_minus_own>0  : {c2}/{len(ok)}  ({100.0*c2/len(ok):.1f}%)  PASS>=50%: {c2*2 >= len(ok)}")
            c3set = {round(x.get("delta_extero"), 6) for x in ok if x.get("delta_extero") is not None}
            print(f"  C3 delta_extero distinct={sorted(c3set, key=nkey)[:8]}  leaves-{{0.0}}: {c3set - {0.0} != set()}")
        else:
            print("  C1/C2/C3 NOT COMPUTABLE (no qualifying record) -> BLOCKED_INFRASTRUCTURE")

        # determinism / provenance
        print(f"  terminal={dist(rows, lambda r: (r.get('outcome_probe') or {}).get('terminal') or r.get('terminal'))}")
        print(f"  demo_pair_count={dist(rows, lambda r: (r.get('in_context') or {}).get('demo_pair_count') or r.get('demo_pair_count'))}")


if __name__ == "__main__":
    main()
