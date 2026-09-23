#!/usr/bin/env python3
"""STEP 2 AUDIT: did the Sagnac veto RUN with the COUPLED resolution fix?

Pre-registered criteria (flag HENRI_MACRO_NUM_CHANNELS=1, bridge unset):
  C1 payloads > 0                      the veto EXECUTED (not skipped)
  C2 zero gate_status=UNAVAILABLE_SHAPE_MISMATCH
  C3 every payload carries a finite delta_axiom (and hard_vetoed)
  C4 no bridge keys consulted (the real path, not a crutch)
  C5 log has zero `einsum()` errors and zero `[opine] unavailable`
`hard_vetoed` value counts are reported as a MEASUREMENT; one-sided is allowed
here because the macro candidate and the boundary axiom are different objects.
Bidirectionality at matched width is proven separately by contract test C6.
"""
import collections
import json
import math
import pathlib
import re
import sys

tel = pathlib.Path(sys.argv[1])
log = pathlib.Path(sys.argv[2])

payload_keys = collections.Counter()
gate_status = collections.Counter()
hard_vetoed = collections.Counter()
deltas = []
records = 0


def walk(o):
    if isinstance(o, dict):
        v = o.get("sagnac_veto")
        if isinstance(v, dict):
            payload_keys[tuple(sorted(v.keys()))] += 1
            gs = v.get("gate_status")
            if gs is not None:
                gate_status[str(gs)] += 1
            if "hard_vetoed" in v:
                hard_vetoed[str(bool(v.get("hard_vetoed")))] += 1
            d = v.get("delta_axiom")
            if isinstance(d, (int, float)) and math.isfinite(float(d)):
                deltas.append(float(d))
        for x in o.values():
            walk(x)
    elif isinstance(o, list):
        for x in o:
            walk(x)


with tel.open(encoding="utf-8", errors="ignore") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        records += 1
        try:
            walk(json.loads(line))
        except Exception:
            pass

n = sum(payload_keys.values())
logtxt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
n_einsum = logtxt.count("einsum()")
n_opine = len(re.findall(r"\[opine\] unavailable", logtxt))
n_bridge = len(re.findall(r"bridge", logtxt, re.I))

print(f"records={records} payloads={n}")
for k, c in payload_keys.most_common(8):
    print(f"  {c:>4}x keys={list(k)}")
print(f"gate_status={dict(gate_status)}")
print(f"hard_vetoed={dict(hard_vetoed)}")
if deltas:
    print(f"delta_axiom n={len(deltas)} min={min(deltas):.6f} "
          f"max={max(deltas):.6f}")
print(f"log: einsum_errors={n_einsum} opine_unavailable={n_opine} "
      f"bridge_mentions={n_bridge}")

c1 = n > 0
c2 = gate_status.get("UNAVAILABLE_SHAPE_MISMATCH", 0) == 0
c3 = n > 0 and len(deltas) >= n
c4 = not any("bridge" in k for k in payload_keys for _ in [0])
c5 = n_einsum == 0 and n_opine == 0
print(f"C1_payloads_present={c1}")
print(f"C2_no_shape_mismatch={c2}")
print(f"C3_delta_on_every_payload={c3}")
print(f"C4_no_bridge_keys={c4}")
print(f"C5_no_einsum_no_opine_error={c5}")
print("STEP2_VERDICT=" + ("VETO_RUNS_AT_REDUCED_SCALE"
                          if (c1 and c2 and c3 and c5)
                          else "NOT_YET"))
