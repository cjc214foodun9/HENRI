#!/usr/bin/env python3
"""Name the post-fix error. One datum decides: WORKED or DISPLACED."""

import collections
import json
import pathlib

EXP = pathlib.Path(r"C:\Users\chan\HENRI_telemetry_exports")
files = sorted(EXP.glob("production_run_*.jsonl"),
               key=lambda p: p.stat().st_mtime, reverse=True)[:3]

for f in files:
    msgs = collections.Counter()
    where = collections.Counter()
    sv = collections.Counter()
    n = 0

    def walk(o, path="root"):
        if isinstance(o, dict):
            if isinstance(o.get("error"), str):
                msgs[o["error"][:200]] += 1
                where[path] += 1
            if "sagnac_veto" in o:
                v = o["sagnac_veto"]
                sv[type(v).__name__] += 1
            for k, v in o.items():
                walk(v, path + "." + str(k))
        elif isinstance(o, list):
            for i, v in enumerate(o[:6]):
                walk(v, path + "[]")

    for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        n += 1
        try:
            walk(json.loads(line))
        except Exception:
            pass

    print("=" * 74)
    print(f"{f.name}  records={n}")
    print(f"  sagnac_veto value types : {dict(sv) or '{}'}")
    print(f"  error-dict locations    : {dict(where) or '{}'}")
    if msgs:
        print("  ERROR MESSAGES:")
        for k, v in msgs.most_common(5):
            print(f"    {v:>3}x  {k}")
    else:
        print("  (no error strings)")

print("=" * 74)
print("VERDICT RULE")
print("  If the message names a CHANNEL/RESOLUTION problem in the macro-option path")
print("  (e.g. an index or shape mismatch consuming a 64-channel field), then my fix")
print("  DISPLACED the failure: the width mismatch is gone but the next consumer")
print("  cannot handle the run-matched resolution.")
print("  If it instead names a width/shape mismatch in dual_channel_sagnac_veto, the")
print("  fix did NOT take effect and the earlier gate_status payloads were from a")
print("  stale run.")
