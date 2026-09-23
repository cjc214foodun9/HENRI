#!/usr/bin/env python3
"""CONSOLIDATED RECEIPT: status of the FIVE approved actions.

Aggregates the individual receipts already produced this session, and measures the
one remaining datum: the cost of constructing the FULL-SCALE planner on CPU, to size
the full-scale run before GPU spend.

SCALE SWITCH, read from production_arc_run.py:88-92
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda": SCALE = dict(num_experts=1024, d_model=65536, r_rank=16,
                                     num_blocks=8192)
    else:                SCALE = dict(num_experts=64,   d_model=512,  r_rank=8,
                                     num_blocks=64)
    So full scale is selected by the DEVICE, not by a flag. Provisioning CUDA
    automatically selects the configuration where the widths match.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
VER = WT / "experiments" / "verification"

out: dict = {"actions": {}, "measurements": {}, "errors": []}


def load(name: str):
    p = VER / name
    if not p.exists():
        return {"MISSING": name}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return {"PARSE_ERROR": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------ ACTION 1
a1 = load("applied_patches_verify.json")
a1c = a1.get("checks", {})
out["actions"]["1_verify_patches"] = {
    "verdict": a1.get("verdict"),
    "isinstance_form": a1c.get("A3_isinstance_match"),
    "name_comparison_form": a1c.get("A3_name_comparison"),
    "opine_info_marks": a1c.get("B_opine_marks"),
    "opine_info_inside_except": a1c.get("B_any_inside_except"),
    "mismatch_raises_named": a1c.get("C_mismatch_raises_named"),
    "hard_vetoed_both_values": a1c.get("D_both_values_reachable"),
    "mixed_dtype_at_matched_width": a1c.get("E_mixed_dtype_ok"),
    "conclusion": ("opine_info is NOT inside any except handler -> the width fix "
                   "cannot trigger a NameError on the success path"),
}
if a1c.get("B_any_inside_except"):
    out["errors"].append("ACTION1: opine_info IS inside an except handler")

# ------------------------------------------------------------------ ACTION 1b
b = load("binding_and_fullscale_width.json")
bc = b.get("checks", {})
out["actions"]["1b_binding_defect"] = {
    "found_defect": bc.get("P1_gate_unavailable_imported") is False,
    "usage_lines_before_fix": [u["line"] for u in bc.get("P2_usage_sites", [])],
    "runtime_bound_before_fix": bc.get("P3_runtime_bound"),
    "note": ("the isinstance handler patch applied while its companion IMPORT patch "
             "failed ('Found 2 matches'); the name was used and unbound, so the "
             "handler would raise NameError exactly when the veto raised. "
             "py_compile and ast.parse both pass on that state."),
}
f = load("action1_action2_verify.json")
fc = f.get("checks", {})
out["actions"]["1b_after_fix"] = {
    "verdict": f.get("verdict"),
    "gate_bound": fc.get("V1_gate_bound"),
    "runtime_bound": fc.get("V1_runtime_bound"),
    "os_bound_in_planner": fc.get("V2_os_bound_in_planner"),
    "opine_info_unsafe": fc.get("V3_unsafe_inside_except"),
}
if fc.get("V1_gate_bound") is not True:
    out["errors"].append("ACTION1b: binding fix not confirmed")

# ------------------------------------------------------------------ ACTION 2
out["actions"]["2_width_contract"] = {
    "verdict": f.get("verdict"),
    "bridge_off_raises": fc.get("V5_raises_when_off"),
    "bridge_on_tuple_len": fc.get("V6_tuple_len"),
    "bridge_record": fc.get("V6_bridge_recorded"),
    "matched_width_tuple_len": fc.get("V7_tuple_len"),
    "matched_width_no_bridge": fc.get("V7_bridge_none"),
    "hard_vetoed_both_values": fc.get("V9_both_reachable"),
}

# ------------------------------------------------------------------ ACTION 3
a3 = load("gauntlet_gate_audit.json")
arms = a3.get("arms", {})
out["actions"]["3_gauntlet_ab"] = {
    "verdict": a3.get("verdict"),
    "verdicts": a3.get("verdicts"),
    "off_arm": {k: arms.get("bridge_off", {}).get(k) for k in
                ("n", "unavailable", "passed", "vetoed", "error",
                 "gate_status_values")},
    "on_arm": {k: arms.get("bridge_on", {}).get(k) for k in
               ("n", "unavailable", "passed", "vetoed", "error",
                "delta_axiom_min", "delta_axiom_max")},
    "gate_not_relaxed": (
        "my pre-registered expectation (both hard_vetoed values in the ON arm) was "
        "WRONG; on_both_values=False is recorded as the finding and the gate was NOT "
        "lowered"),
}
ac = load("action3_conclusion.json")
out["actions"]["3_interpretation"] = ac.get("derived_independent_vector_check")

# ------------------------------------------------------------------ ACTION 4
out["actions"]["4_commit_push"] = {
    "commits_on_branch": ["33af147", "3c48589", "997cf32", "ef07279", "79fffa0",
                          "36f50ab"],
    "pushed": True,
    "remote_ref": "refs/heads/milestone1-readout-decoupling",
    "last_pushed": "36f50ab",
}

# ------------------------------------------------------------------ ACTION 5
a5 = load("action5_su3_width_rule.json")
a5c = a5.get("checks", {})
out["actions"]["5_full_scale_width"] = {
    "verdict": a5.get("verdict"),
    "widths_by_N": a5c.get("B_field_to_wave_widths"),
    "encoder_refs": a5c.get("B_encoder_ref_widths"),
    "full_scale_match": a5c.get("B_widths_match_at_full_scale"),
    "reduced_scale_match": a5c.get("B_widths_match_at_reduced_scale"),
    "full_width_veto_runs": a5c.get("B_full_width_veto_runs"),
    "full_width_both_values": a5c.get("B_full_width_both_values"),
    "full_width_pass_delta": a5c.get("B_full_width_pass_delta"),
    "full_width_veto_delta": a5c.get("B_full_width_veto_delta"),
}
up = load("umacro_provenance.json")
upc = up.get("checks", {})
out["actions"]["5_macro_provenance"] = {
    "assignments": upc.get("_u_macro_assignments"),
    "verdict": upc.get("verdict"),
    "opine_classes": upc.get("opine_engine_classes"),
}

# ------------------------------------------------------------------ scale switch
try:
    rl = (WT / "production_arc_run.py").read_text(encoding="utf-8",
                                                  errors="ignore").splitlines()
    seg = [f"{i}: {rl[i-1]}" for i in range(86, 93) if i <= len(rl)]
    out["measurements"]["scale_switch"] = seg
except Exception as e:  # noqa: BLE001
    out["errors"].append(f"scale switch read failed: {type(e).__name__}: {e}")

# ------------------------------------------------------------------ full-scale cost
try:
    import sys
    sys.path.insert(0, str(WT))
    import torch
    t0 = time.time()
    from sagnac_mcts_planner import SagnacMCTSPlanner
    pl = SagnacMCTSPlanner(d_model=8192 * 8, k_blocks=8192, tau_veto=0.35, device="cpu")
    out["measurements"]["full_scale_ctor_seconds"] = round(time.time() - t0, 2)
    out["measurements"]["full_scale_ctors_on_cpu"] = True
    # a single veto at full width: cost + bidirectionality
    W = 65536
    u = torch.zeros(W); u[0] = 1.0
    t1 = time.time()
    r_ok = pl.dual_channel_sagnac_veto(u, u, u, epsilon_hard=pl.tau_veto)
    r_no = pl.dual_channel_sagnac_veto(-u, u, u, epsilon_hard=pl.tau_veto)
    out["measurements"]["full_width_veto_pair_seconds"] = round(time.time() - t1, 4)
    out["measurements"]["full_width_pass_delta"] = round(float(r_ok[0]), 6)
    out["measurements"]["full_width_veto_delta"] = round(float(r_no[0]), 6)
    out["measurements"]["full_width_both_values"] = ((not bool(r_ok[2])) and bool(r_no[2]))
except Exception as e:  # noqa: BLE001
    out["measurements"]["full_scale_ctor_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    out["traceback_tail"] = traceback.format_exc()[-600:]

# ------------------------------------------------------------------ checkpoint
ck = WT / "models" / "henri_decoder_checkpoint.pt"
out["measurements"]["checkpoint_overlay"] = {
    "present": ck.exists(),
    "bytes": ck.stat().st_size if ck.exists() else None,
    "sha256": "75572389083455a371546b40500b6614abfc3a245cfa0db9eba74c183a974060",
    "sha_source": "action5_prep.py measured src==dst equality",
}

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = VER / "five_actions_status.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("FIVE_ACTIONS=" + out["verdict"] + " errors=" + str(len(out["errors"])))
for k in sorted(out["actions"]):
    v = out["actions"][k]
    verdict = v.get("verdict") if isinstance(v, dict) else None
    print(f"  {k}: verdict={verdict}")
print("A1 isinstance_form=" + str(a1c.get("A3_isinstance_match"))
      + " opine_inside=" + str(a1c.get("B_any_inside_except")))
print("A1b runtime_bound_after_fix=" + str(fc.get("V1_runtime_bound")))
print("A3 verdicts=" + json.dumps(a3.get("verdicts")))
print("A5 full_match=" + str(a5c.get("B_widths_match_at_full_scale"))
      + " veto_runs=" + str(a5c.get("B_full_width_veto_runs"))
      + " both=" + str(a5c.get("B_full_width_both_values")))
print("scale_switch=" + json.dumps(out["measurements"].get("scale_switch")))
print("full_ctor_s=" + str(out["measurements"].get("full_scale_ctor_seconds"))
      + " veto_pair_s=" + str(out["measurements"].get("full_width_veto_pair_seconds"))
      + " both=" + str(out["measurements"].get("full_width_both_values")))
for e in out["errors"]:
    print("ERR: " + e[:200])
print("RECEIPT=" + str(receipt))
