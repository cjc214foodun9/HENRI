#!/usr/bin/env python3
"""Audit the FULL-SCALE run (num_blocks=8192, d_model=65536).

Guard: this run used `offline://surrogate` (HENRI_OFFLINE_DIAG=1), which the runner's
own comment marks "Diagnostic only - never score-eligible". So the audit reports the
WIDTH/TELEMETRY mechanism and the raw external outcome separately. No score claim.
"""
import collections
import json
import math
import pathlib
import re

D = pathlib.Path(r"C:\Users\chan\HENRI_telemetry_exports\fullscale_52161444")

print("=== A. RECEIPTS (pulled before stop) ===")
for p in sorted(D.glob("*")):
    print("  %-46s %9d B" % (p.name, p.stat().st_size))

tel = sorted(D.glob("*.jsonl"))
scs = sorted(D.glob("*scorecards.json"))
log = (D / "fullscale_diag.log")
logtxt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""

# ---------------------------------------------------------------- telemetry
pk = collections.Counter()
gs = collections.Counter()
hv = collections.Counter()
dax = []
dep = []
types = collections.Counter()
records = 0


def walk(o):
    if isinstance(o, dict):
        t = o.get("event_type") or o.get("type")
        if t:
            types[str(t)] += 1
        v = o.get("sagnac_veto")
        if isinstance(v, dict):
            pk[tuple(sorted(v.keys()))] += 1
            if v.get("gate_status") is not None:
                gs[str(v["gate_status"])] += 1
            if "hard_vetoed" in v:
                hv[str(bool(v["hard_vetoed"]))] += 1
            for key, acc in (("delta_axiom", dax), ("delta_epistemic", dep)):
                x = v.get(key)
                if isinstance(x, (int, float)) and math.isfinite(float(x)):
                    acc.append(float(x))
        for x in o.values():
            walk(x)
    elif isinstance(o, list):
        for x in o:
            walk(x)


for p in tel:
    with p.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records += 1
            try:
                walk(json.loads(line))
            except Exception:
                pass

n = sum(pk.values())
print("\n=== B. TELEMETRY (full scale) ===")
print("  files=%d  records=%d  veto_payloads=%d" % (len(tel), records, n))
for k, c in pk.most_common(6):
    print("    %4dx keys=%s" % (c, list(k)))
print("  gate_status = %s" % (dict(gs) or "{}  <- ABSENT, i.e. the veto RAN (no shape error)"))
print("  hard_vetoed = %s" % dict(hv))
for nm, arr in (("delta_axiom", dax), ("delta_epistemic", dep)):
    if arr:
        print("  %s: n=%d min=%.6f max=%.6f mean=%.6f"
              % (nm, len(arr), min(arr), max(arr), sum(arr) / len(arr)))
    else:
        print("  %s: ABSENT" % nm)
print("  record_types = %s" % dict(types.most_common(6)))

# ---------------------------------------------------------------- log
print("\n=== C. LOG ===")
print("  log_lines=%d  steps_logged=%d"
      % (len(logtxt.splitlines()), len(re.findall(r"^  step ", logtxt, re.M))))
einsum_n = logtxt.count("einsum()")
opine_unavail = len(re.findall(r"\[opine\] unavailable", logtxt))
print("  einsum_errors=%d   opine_unavailable=%d" % (einsum_n, opine_unavail))
checks = (
    (r"device=cuda scale=\{.*?\}", "device/scale"),
    (r"\[opine\] macro-option[^\n]*", "opine"),
    (r"\[env summary\][^\n]*", "env summary"),
    (r"^  step .*", "step"),
)
for pat, label in checks:
    hits = re.findall(pat, logtxt, re.M)
    for s in hits[:2]:
        print("  %-12s -> %s" % (label, s[:150]))

# ---------------------------------------------------------------- scorecards
print("\n=== D. SCORECARDS (external outcome, verbatim) ===")
for p in scs:
    try:
        d = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:  # noqa: BLE001
        print("  %s: parse error %s" % (p.name, e))
        continue
    print("  file=%s" % p.name)

    def show(o, ind="    "):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(v, (dict, list)):
                    print("%s%s:" % (ind, k))
                    show(v, ind + "  ")
                else:
                    print("%s%s = %r" % (ind, k, v))
        elif isinstance(o, list):
            for i, v in enumerate(o[:3]):
                print("%s[%d]" % (ind, i))
                show(v, ind + "  ")

    show(d)

summ = D / "p823_gauntlet_summary.json"
print("\n=== E. GAUNTLET VERDICT FILE ===")
if summ.exists():
    print(json.dumps(json.loads(summ.read_text(encoding="utf-8", errors="replace")), indent=1)[:800])
else:
    print("  (absent)")

print("\n=== F. INTERPRETATION GUARD ===")
print("  Zone C path = offline://surrogate (HENRI_OFFLINE_DIAG=1) -> DIAGNOSTIC ONLY.")
print("  This is width/mechanism evidence at production scale, NOT a capability claim.")
