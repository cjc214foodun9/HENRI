#!/usr/bin/env python3
"""UHR-03 POOLED reducer -- judges the paired A/B against the FROZEN criteria.

DESIGN NOTE -- the defect this file now carries a guard against.
The helper `pooled_domain_statistic` returns UNPREFIXED keys (`own_is_min`,
`margin`, `margin_above_band`). The runner emits a PREFIXED copy alongside it
(`pooled_own_is_min`, `pooled_margin`, ...). Version 1 of this reducer selected the
dict that has `n_channels` (the helper's return) and then read the PREFIXED names,
which exist only on the parent -- so every record read as `None` and the verdict was
a FALSE NEGATIVE ("C1 0/16, kill strike 1 of 2") while the payload actually said
`own_is_min: true` and `margin_above_band: true` in 16/16 records.

Two rules now enforced:
  R1  Record ONLY the helper's return dict (`n_channels` present). This also fixes the
      double-count where parent + child both matched (32 dicts for 16 records).
  R2  Read each field by ACCEPTING BOTH names, and report which name supplied it.

A reducer that misreads its own payload is a verification-harness defect of the same
class as a gate that cannot fail.
"""
import json, sys, glob, os, re, collections, statistics as st

ARM_DIRS = {"BASELINE": "telemetry_uhr03_BASELINE", "RFSS": "telemetry_uhr03_RFSS"}


def find_pooled(obj, acc):
    """R1: only the helper's return dict -- it is the one carrying `n_channels`."""
    if isinstance(obj, dict):
        if "n_channels" in obj and ("margin" in obj or "own_is_min" in obj):
            acc.append(obj)
            return acc            # do not descend into the helper's own payload
        for v in obj.values():
            find_pooled(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_pooled(v, acc)
    return acc


def field(d, base, sources=None):
    """R2: accept the unprefixed helper name and the runner's prefixed name."""
    for name in (base, "pooled_" + base):
        if d.get(name) is not None:
            if sources is not None:
                sources.add(name)
            return d.get(name)
    return None


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


def main(root):
    out = {}
    for arm, sub in ARM_DIRS.items():
        files = sorted(glob.glob(os.path.join(root, sub, "production_run_*.jsonl")))
        if not files:
            print("[%s] NO JSONL under %s/" % (arm, sub))
            out[arm] = None
            continue
        rows, pooled, guards, probes = load(files[-1])
        print("=" * 78)
        print("[%s] %s  records=%d  pooled_records=%d"
              % (arm, os.path.basename(files[-1]), len(rows), len(pooled)))
        print("=" * 78)
        if not pooled:
            print("  no pooled payload -> FORM B OFF or the block never ran")
            out[arm] = {"n": 0}
            continue

        src = set()
        mar  = [v for v in (field(p, "margin", src) for p in pooled) if v is not None]
        own  = [v for v in (field(p, "delta_pooled_own", src) for p in pooled) if v is not None]
        inv  = [v for v in (field(p, "delta_pooled_invalid_min", src) for p in pooled) if v is not None]
        mind = [v for v in (field(p, "invalid_minus_own", src) for p in pooled) if v is not None]
        band = [v for v in (field(p, "band", src) for p in pooled) if v is not None]
        nch  = [v for v in (field(p, "n_channels", src) for p in pooled) if v is not None]
        stat = sorted({str(field(p, "status")) for p in pooled})

        ok = [p for p in pooled
              if field(p, "status") == "OK" and (field(p, "n_channels") or 0) >= 2]
        print("  field names resolved from: %s" % sorted(src))
        print("  status            : %s" % stat)
        print("  n_channels        : %s" % (sorted(set(nch)) if nch else "n/a"))
        if band:
            print("  band              : %.6e" % band[0])
        if mar:
            print("  margin            : mean=%+.6f min=%+.6f max=%+.6f" % (st.mean(mar), min(mar), max(mar)))
        if own:
            print("  delta_pooled_own  : distinct=%d min=%.6f max=%.6f"
                  % (len(set(round(v, 9) for v in own)), min(own), max(own)))
        if inv:
            # the COMPARATOR NATURE: if every invalid action shares one value, the
            # competitor set is identity-only (support overlap, not content).
            vals = collections.Counter(round(v, 6) for v in inv)
            print("  delta_pooled_inv  : distinct=%d  min=%.6f  max=%.6f" % (len(vals), min(inv), max(inv)))
            print("  invalid value mode: %s%s"
                  % (vals.most_common(1), "  <- SINGLE VALUE: competitor set is identity-only"
                     if len(vals) == 1 else ""))

        n_self = sum(1 for p in ok if field(p, "own_is_min") is True)
        n_band = sum(1 for p in ok if field(p, "margin_above_band") is True)
        n_mono = sum(1 for p in ok
                     if (field(p, "delta_pooled_own") is not None
                         and field(p, "delta_pooled_invalid_min") is not None
                         and field(p, "delta_pooled_own") < field(p, "delta_pooled_invalid_min")))
        n_tot = len(ok)
        c1 = (n_self / n_tot) if n_tot else None
        c2 = (n_band / n_tot) if n_tot else None
        c3 = len(set(round(v, 9) for v in own)) > 1
        g2 = any(g.get("updated") is True for g in guards)
        g3 = any(field(p, "status") == "OK" for p in pooled)
        moved = [p.get("frame_changed") for p in probes if p.get("frame_changed") is not None]
        print()
        print("  G2 guard executed : %s" % g2)
        print("  G3 FORM B defined : %s" % g3)
        print("  G1 frame moved    : %d/%d probes" % (sum(1 for m in moved if m), len(moved)))
        print("  C1 own is argmin  : %d/%d  (%s)" % (n_self, n_tot, "%.0f%%" % (100 * c1) if c1 is not None else "n/a"))
        print("  C2 margin > band  : %d/%d  (%s)" % (n_band, n_tot, "%.0f%%" % (100 * c2) if c2 is not None else "n/a"))
        print("  C2' own < inv_min : %d/%d  (independent recomputation from raw values)" % (n_mono, n_tot))
        print("  C3 owns vary      : %s  distinct=%d" % (c3, len(set(round(v, 9) for v in own))))
        out[arm] = {"n": len(pooled), "n_scored": n_tot, "C1": c1, "C2": c2, "C2p": n_mono,
                    "C3": c3, "G2": g2, "G3": g3, "margin_mean": (st.mean(mar) if mar else None),
                    "inv_distinct": (len(set(round(v, 6) for v in inv)) if inv else None)}

    print()
    print("=" * 78); print("VERDICT"); print("=" * 78)
    b, r = out.get("BASELINE"), out.get("RFSS")
    if not r or not r.get("n"):
        print("  RFSS produced NO pooled payload -> BLOCKED_INFRASTRUCTURE, 0 kill used")
        return
    if b and b.get("n"):
        print("  WARNING: BASELINE emitted pooled keys too -> G5 (flag-OFF identity) FAILS")
    else:
        print("  G5 flag-OFF identity: BASELINE emitted 0 pooled keys -> PASS")
    if r.get("n_scored", 0) == 0:
        print("  RFSS pooled records all had n_channels < 2 -> BLOCKED (domain vacuous)")
        return
    print("  RFSS C1=%.0f%%  C2=%.0f%%  C2'=%d/%d  C3=%s  margin_mean=%+.6f  invalid_distinct=%s"
          % (100 * r["C1"], 100 * r["C2"], r["C2p"], r["n_scored"], r["C3"],
             r["margin_mean"], r["inv_distinct"]))
    if r["C1"] == 1.0 and r["C2"] == 1.0:
        print("  -> FORM B SEPARATES: the true action is the strict argmin in every scored record.")
        if r["inv_distinct"] == 1:
            print("  -> SCOPE: the live competitor set is IDENTITY-ONLY (support overlap), so this")
            print("     demonstrates action IDENTIFICATION on the live path, not content")
            print("     discrimination. Content isolation is the local hard-comparator test.")
    elif (r["C1"] or 0) >= 0.5 and (r["C2"] or 0) >= 0.5:
        print("  -> FORM B separates on a MAJORITY of scored records.")
    else:
        print("  -> FORM B FAILED TO SEPARATE on this run. Kill strike 1 of 2.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.expandvars(r"%LOCALAPPDATA%\Temp\uhr03_egress"))
