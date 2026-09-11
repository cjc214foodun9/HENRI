#!/usr/bin/env python
"""Gate 1 discrimination analysis — payload-engagement vs bare-enum control.

Reads ARC_ACTION_PAYLOAD events + ARC_EPISODE_TRACE + scorecards from the
g1 factorial telemetry (cells A/B/C/D x lf52/tn36/sc25, payloads ON).
Prints a compact JSON verdict for the pre-registered gate:
- payload-complete calls observed
- meaningful frame changes vs bare-enum control (1/4096 = 0.000244)
- no nonzero arm exits
"""
import glob
import json
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
CELLS = ["A", "B", "C", "D"]
ENVS = ["lf52", "tn36", "sc25"]

report = {"cells": {}}
payload_total = 0
payload_complete_total = 0
action6_payload_events = 0
bare_enum_events = 0
changed_cells_all = []
changed_fraction_all = []
exact_pass_any = False
terminal_states = {}
arm_errors = []

for cell in CELLS:
    cell_changed = []
    cell_payload_events = 0
    cell_payload_complete = 0
    cell_action6 = 0
    cell_scores = []
    for env in ENVS:
        jl = glob.glob(os.path.join(ROOT, cell, env, "*.jsonl"))
        for path in jl:
            try:
                for line in open(path, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    et = rec.get("event_type")
                    if et == "ARC_ACTION_PAYLOAD":
                        cell_payload_events += 1
                        payload_total += 1
                        for a in rec.get("payload_actions", []):
                            if a.get("payload_present"):
                                cell_payload_complete += 1
                                payload_complete_total += 1
                                if a.get("action_enum") == "ACTION6":
                                    cell_action6 += 1
                                    action6_payload_events += 1
                            else:
                                bare_enum_events += 1
                        cc = rec.get("changed_cells")
                        cf = rec.get("changed_fraction")
                        if cc is not None:
                            cell_changed.append(cc)
                            changed_cells_all.append(cc)
                        if cf is not None:
                            changed_fraction_all.append(cf)
                    elif et == "ARC_EPISODE_TRACE":
                        tr = rec.get("trace", {})
                        ts = tr.get("evaluator_status", "?")
                        terminal_states[f"{cell}/{env}"] = ts
                        if tr.get("exact_pass"):
                            exact_pass_any = True
                        cell_scores.append({
                            "terminal": ts,
                            "exact_pass": tr.get("exact_pass"),
                            "ext_delta": tr.get("external_state_delta"),
                            "policy": tr.get("policy"),
                            "frozen": tr.get("learning_frozen"),
                            "diag": tr.get("diagnostic_only"),
                        })
                    elif et == "ENV_STEP_ERROR":
                        arm_errors.append(f"{cell}/{env}: {rec.get('error', '')[:80]}")
            except Exception as exc:
                arm_errors.append(f"{cell}/{env}: READ {type(exc).__name__}: {exc}")
    n = len(cell_changed)
    report["cells"][cell] = {
        "payload_events": cell_payload_events,
        "payload_complete": cell_payload_complete,
        "action6_with_data": cell_action6,
        "changed_cells_mean": round(sum(cell_changed) / n, 3) if n else None,
        "changed_cells_max": max(cell_changed) if cell_changed else None,
        "envs": cell_scores,
    }

n_all = len(changed_cells_all)
report["summary"] = {
    "payload_events_total": payload_total,
    "payload_complete_total": payload_complete_total,
    "action6_with_data_total": action6_payload_events,
    "bare_enum_events_total": bare_enum_events,
    "changed_cells_mean": round(sum(changed_cells_all) / n_all, 3) if n_all else None,
    "changed_fraction_mean": round(sum(changed_fraction_all) / len(changed_fraction_all), 6) if changed_fraction_all else None,
    "bare_enum_control_fraction": 0.000244,  # observed probe baseline (1/4096)
    "exact_pass_any": exact_pass_any,
    "terminal_states": terminal_states,
    "arm_errors": arm_errors,
}
# Pre-registered discrimination gate:
# 1. payload-complete calls observed
# 2. meaningful frame changes exceed the bare-enum control
# 3. no arm exits nonzero
gate = {
    "payload_complete_observed": payload_complete_total > 0,
    "action6_payloads_observed": action6_payload_events > 0,
    "changed_exceeds_bare_control": (
        n_all > 0 and
        (sum(changed_cells_all) / n_all) > 1.0  # mean changed cells > 1 = beyond bare 1/4096
    ),
    "no_arm_errors": len(arm_errors) == 0,
}
report["gate_verdict"] = gate
report["gate_pass"] = all(gate.values())
print(json.dumps(report, indent=1))
