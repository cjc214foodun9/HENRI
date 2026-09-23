#!/usr/bin/env python3
"""Settle the rendering discrepancy by reading BYTES, in BOTH trees.

THE CONTRADICTION
    My receipts and the advisor rendering disagree on three checkable facts about
    `production_arc_run.py`:

        fact                      my receipt        advisor rendering
        handler match form        isinstance (2327) `_name == "..."` (2280)
        extra fields              ABSENT            candidate_elems/veto_stage/hint
        file length               3486 lines        3426 lines
        opine_info sites          2337, 2350        2300, 2323

    A stale revision cannot explain a 60-line difference in the same file, so one
    rendering is wrong. Only a byte read decides it. This also checks two adjacent
    existence claims (a file the rendering cites, and `def field_to_wave` placement).

WHY A SCRIPT AND NOT AN INLINE COMMAND
    The inline version of this check was rejected by the command blocklist for being
    an oversized payload. A file plus `bash <file>` is the supported form.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TOP = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM")
TREES = {"worktree": TOP / ".worktrees" / "semantic-backbone" / "HENRI V2",
         "main": TOP / "HENRI V2"}

NEEDLES = {
    # the two competing handler forms
    "isinstance_form": "isinstance(_veto_exc, SagnacGateUnavailable)",
    "name_assign": "_name = type(_veto_exc).__name__",
    "name_cmp": '_name == "SagnacGateUnavailable"',
    # fields only the rendering shows
    "candidate_elems": "candidate_elems",
    "veto_stage": "veto_stage",
    "shape_candidate": "shape_candidate",
    "hint_field": '"hint"',
    # things my patches added
    "gate_status": "gate_status",
    "unavailable_label": "UNAVAILABLE_SHAPE_MISMATCH",
    "width_bridge": "HENRI_SAGNAC_WIDTH_BRIDGE",
    "veto_debug": "HENRI_ARC_VETO_DEBUG",
    "explicit_epsilon": "epsilon_hard=sagnac_planner.tau_veto",
}

CONTESTED_FILE = "synthesize_sagnac_guarded_macro_options.py"

out: dict = {"trees": {}}

for label, base in TREES.items():
    rec: dict = {"root": str(base), "exists": base.exists()}
    if not base.exists():
        out["trees"][label] = rec
        continue
    f = base / "production_arc_run.py"
    if not f.exists():
        rec["runner_exists"] = False
        out["trees"][label] = rec
        continue
    blob = f.read_bytes()
    txt = blob.decode("utf-8", errors="replace")
    lines = txt.splitlines()
    rec.update({
        "runner_exists": True,
        "bytes": len(blob),
        "lines": len(lines),
        "crlf": blob.count(b"\r\n"),
        "sha256_16": hashlib.sha256(blob).hexdigest()[:16],
        "needle_counts": {k: txt.count(v) for k, v in NEEDLES.items()},
        "opine_info_lines": [i for i, l in enumerate(lines, 1)
                             if l.strip().startswith("opine_info = {")],
        "except_veto_lines": [i for i, l in enumerate(lines, 1)
                              if l.strip().startswith("except Exception as _veto_exc")],
        "isinstance_lines": [i for i, l in enumerate(lines, 1)
                             if "isinstance(_veto_exc" in l],
        "name_cmp_lines": [i for i, l in enumerate(lines, 1)
                           if '_name == "SagnacGateUnavailable"' in l],
    })
    # the contested file
    cf = base / CONTESTED_FILE
    rec["contested_file"] = {"exists": cf.exists()}
    if cf.exists():
        ct = cf.read_text(encoding="utf-8", errors="replace")
        rec["contested_file"].update({
            "lines": len(ct.splitlines()),
            "has_su3_class": "class SU3FieldWaveTransducer" in ct,
            "su3_class_lines": [i for i, l in enumerate(ct.splitlines(), 1)
                                if "class SU3FieldWaveTransducer" in l],
        })
    # efe_planner: is field_to_wave DEFINED there?
    ep = base / "efe_planner.py"
    if ep.exists():
        et = ep.read_text(encoding="utf-8", errors="replace")
        rec["efe_planner"] = {
            "lines": len(et.splitlines()),
            "def_field_to_wave_count": et.count("def field_to_wave"),
            "mention_count": et.count("field_to_wave"),
            "def_lines": [i for i, l in enumerate(et.splitlines(), 1)
                          if l.strip().startswith("def field_to_wave")],
        }
    out["trees"][label] = rec

# ------------------------------------------------------------------ verdict
w = out["trees"].get("worktree", {})
counts = w.get("needle_counts", {})
verdicts = {
    "handler_form": ("ISINSTANCE" if counts.get("isinstance_form", 0) > 0
                     and counts.get("name_cmp", 0) == 0 else
                     "NAME_COMPARISON" if counts.get("name_cmp", 0) > 0
                     and counts.get("isinstance_form", 0) == 0 else
                     "BOTH" if counts.get("name_cmp", 0) > 0 else "NEITHER"),
    "extra_fields_present": any(counts.get(k, 0) > 0 for k in
                                ("candidate_elems", "veto_stage", "shape_candidate")),
    "runner_lines": w.get("lines"),
    "opine_info_lines": w.get("opine_info_lines"),
    "efe_planner_defines_field_to_wave": (w.get("efe_planner", {})
                                          .get("def_field_to_wave_count")),
    "contested_file_exists": w.get("contested_file", {}).get("exists"),
}
out["verdict"] = verdicts
out["rendering_supported"] = bool(
    counts.get("name_cmp", 0) > 0 or verdicts["extra_fields_present"]
    or (w.get("lines") == 3426))
out["conclusion"] = (
    "ADVISOR RENDERING IS SUPPORTED by the bytes" if out["rendering_supported"] else
    "ADVISOR RENDERING IS NOT SUPPORTED by the bytes: my receipts are correct and the "
    "rendering is stale or fabricated")

receipt = w.get("root")
if receipt:
    p = Path(receipt) / "experiments" / "verification" / "rendering_discrepancy.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("RECEIPT=" + str(p))

print("RENDERING_SUPPORTED=" + str(out["rendering_supported"]))
print("CONC=" + out["conclusion"][:90])
for k, v in verdicts.items():
    print(f"  {k} = {v}")
for label in ("worktree", "main"):
    t = out["trees"].get(label, {})
    if not t.get("runner_exists"):
        print(f"  [{label}] runner missing")
        continue
    print(f"  [{label}] lines={t.get('lines')} bytes={t.get('bytes')} "
          f"sha16={t.get('sha256_16')} crlf={t.get('crlf')}")
    print(f"  [{label}] counts={json.dumps(t.get('needle_counts'))}")
    print(f"  [{label}] except={t.get('except_veto_lines')} "
          f"isinstance={t.get('isinstance_lines')} namecmp={t.get('name_cmp_lines')}")
    print(f"  [{label}] contested_file={json.dumps(t.get('contested_file'))}")
    print(f"  [{label}] efe_planner={json.dumps(t.get('efe_planner'))}")
