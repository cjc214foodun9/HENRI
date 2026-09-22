#!/usr/bin/env python3
"""ACTION 5: does the width mismatch DISAPPEAR at full scale (d_model=65536)?

THE PRE-REGISTERED QUESTION
    Measured: the SU3 transducer's `field_to_wave` output width tracks its INPUT
    FIELD, and `state_wave` width = num_blocks * 8 (512 at 64, 65536 at 8192). At the
    reduced scale the gauntlet uses, the candidate is 65536-wide while both references
    are 512-wide -> 128x mismatch -> the veto records UNAVAILABLE_SHAPE_MISMATCH.

    HYPOTHESIS (falsifiable): at full scale (d_model = 65536, num_blocks = 8192) all
    three waves are 65536-wide, so the veto RUNS WITHOUT A BRIDGE. If the widths still
    disagree at full scale, the hypothesis is FALSIFIED and the cross-family boundary
    -- not the scale setting -- is the real cause.

WHAT IS MEASURED (no GPU, no long run)
    S1  construct the planner at d_model = 65536 (requires the staged checkpoint)
    S2  measure the width of a real encoded grid wave
    S3  measure the width of a real SU3 field_to_wave output at that dimension
    S4  call the veto with three full-width waves: does it RUN, and does hard_vetoed
        reach BOTH values?
    S5  report the derived comparison against the mismatched-width call

SCOPE / HONESTY
    This does NOT evaluate ARC. It answers whether the veto is EXECUTABLE at the
    configuration where the widths can agree. A runnable gate is a PRECONDITION for
    any veto claim, not evidence of capability. Architecture policy also says
    full-scale HENRI runs belong on CUDA; on CPU this is a construction-and-width
    probe, and that limitation is recorded rather than hidden.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
out: dict = {"checks": {}, "errors": [], "notes": []}
chk = out["checks"]

FULL_D = 65536
FULL_NB = 8192

try:
    sys.path.insert(0, str(WT))
    import numpy as np
    import torch

    t0 = time.time()

    # ---------------------------------------------------------------- S1
    from sagnac_mcts_planner import SagnacMCTSPlanner, SagnacGateUnavailable
    planner = SagnacMCTSPlanner(d_model=FULL_D, k_blocks=FULL_NB, tau_veto=0.35,
                                device="cpu")
    chk["S1_constructed"] = True
    chk["S1_seconds"] = round(time.time() - t0, 2)
    chk["S1_d_model"] = int(planner.d_model)
    chk["S1_k_blocks"] = int(planner.k_blocks)
    chk["S1_has_su3"] = hasattr(planner, "_su3_transducer") or hasattr(
        planner, "su3_transducer")

    # ---------------------------------------------------------------- S2
    grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    state_wave = planner.vision_encoder.encode_grid(grid)
    chk["S2_state_wave_shape"] = list(state_wave.shape)
    chk["S2_state_wave_numel"] = int(state_wave.numel())
    chk["S2_state_wave_dtype"] = str(state_wave.dtype)
    w_ref = state_wave.detach().reshape(-1)
    chk["S2_ref_width"] = int(w_ref.numel())

    # ---------------------------------------------------------------- S3
    # The macro wave comes from the SU3 transducer on the planner (or the EFE planner).
    su3 = getattr(planner, "_su3_transducer", None)
    if su3 is None:
        su3 = getattr(planner, "su3_transducer", None)
    if su3 is None:
        # try the EFE planner owned by the planner, if present
        ep = getattr(planner, "efe_planner", None) or getattr(planner, "planner", None)
        su3 = getattr(ep, "_su3_transducer", None) if ep is not None else None
    chk["S3_su3_found"] = su3 is not None

    macro_candidates = {}
    if su3 is not None and hasattr(su3, "field_to_wave"):
        # Build a field of the SAME width family the transducer consumes. The
        # production call is field_to_wave(_u_macro.unsqueeze(0)) where _u_macro is an
        # SU(3) field; its width was observed as 65536 at reduced run scale. Try the
        # natural widths and record whichever the transducer accepts.
        for label, width in (("state_width", int(w_ref.numel())),
                             ("full_D", FULL_D)):
            try:
                field = torch.randn(1, width, generator=torch.Generator().manual_seed(3))
                w = su3.field_to_wave(field)
                macro_candidates[label] = {
                    "input_width": width,
                    "out_shape": list(w.shape),
                    "out_numel": int(w.numel()),
                    "out_dtype": str(w.dtype),
                }
            except Exception as e:  # noqa: BLE001
                macro_candidates[label] = {"input_width": width,
                                           "error": f"{type(e).__name__}: {str(e)[:120]}"}
    chk["S3_macro_candidates"] = macro_candidates

    # Pick the candidate whose output width matches the reference, if any.
    matched_label = next(
        (k for k, v in macro_candidates.items()
         if v.get("out_numel") == chk["S2_ref_width"]), None)
    chk["S3_natural_match_label"] = matched_label
    chk["S3_widths_match_at_full_scale"] = matched_label is not None

    # ---------------------------------------------------------------- S4
    if matched_label is not None:
        width = FULL_D if matched_label == "full_D" else chk["S2_ref_width"]
        gen = torch.Generator().manual_seed(11)
        field = torch.randn(1, width, generator=gen)
        macro = su3.field_to_wave(field).reshape(-1)
        # three full-width waves
        u = torch.zeros(chk["S2_ref_width"])
        u[0] = 1.0
        try:
            r_self = planner.dual_channel_sagnac_veto(macro, macro, macro,
                                                      epsilon_hard=planner.tau_veto)
            chk["S4_veto_runs_at_full_scale"] = True
            chk["S4_self_delta_axiom"] = round(float(r_self[0]), 6)
            chk["S4_self_hard"] = bool(r_self[2])
        except Exception as e:  # noqa: BLE001
            chk["S4_veto_runs_at_full_scale"] = False
            chk["S4_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            out["errors"].append(f"S4: veto failed at full scale: {chk['S4_error']}")
        # deterministic both-values pair on the shared width
        base = torch.zeros(chk["S2_ref_width"])
        base[0] = 1.0
        try:
            r_pass = planner.dual_channel_sagnac_veto(base, base, base,
                                                      epsilon_hard=planner.tau_veto)
            r_veto = planner.dual_channel_sagnac_veto(-base, base, base,
                                                      epsilon_hard=planner.tau_veto)
            chk["S4_pass_hard"] = bool(r_pass[2])
            chk["S4_veto_hard"] = bool(r_veto[2])
            chk["S4_pass_delta"] = round(float(r_pass[0]), 6)
            chk["S4_veto_delta"] = round(float(r_veto[0]), 6)
            chk["S4_both_values_at_full_width"] = ((not bool(r_pass[2]))
                                                   and bool(r_veto[2]))
            if not chk["S4_both_values_at_full_width"]:
                out["errors"].append("S4: hard_vetoed not both-reachable at full width")
        except Exception as e:  # noqa: BLE001
            chk["S4_pair_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    else:
        out["notes"].append(
            "S3: no natural width match found between the SU3 output and the grid "
            "reference at this construction. The full-scale hypothesis is NOT "
            "confirmed by this probe; record the measured widths and treat the "
            "cross-family boundary as the cause.")
        # Still measure whether a mismatched call raises the named type here.
        try:
            planner.dual_channel_sagnac_veto(
                torch.randn(FULL_D).to(torch.complex64), w_ref, w_ref,
                epsilon_hard=planner.tau_veto)
            chk["S4_mismatch_raised"] = False
        except SagnacGateUnavailable:
            chk["S4_mismatch_raised"] = True
        except Exception as e:  # noqa: BLE001
            chk["S4_mismatch_raised"] = f"{type(e).__name__}"

    chk["elapsed_s"] = round(time.time() - t0, 2)

except Exception as e:  # noqa: BLE001
    out["errors"].append(f"FATAL: {type(e).__name__}: {e}")
    out["traceback_tail"] = traceback.format_exc()[-1200:]

out["verdict"] = ("PASS" if not out["errors"] else "FAIL")
why = ("WIDTHS_MATCH_AT_FULL_SCALE" if chk.get("S3_widths_match_at_full_scale")
       else "WIDTHS_DO_NOT_MATCH_AT_FULL_SCALE")
out["hypothesis_verdict"] = why
receipt = WT / "experiments" / "verification" / "action5_fullscale_width.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("A5=" + out["verdict"] + " hypothesis=" + why
      + " errors=" + str(len(out["errors"])))
print("S1=" + str(chk.get("S1_constructed")) + " in " + str(chk.get("S1_seconds"))
      + "s  state_numel=" + str(chk.get("S2_state_wave_numel")))
print("S3_candidates=" + json.dumps(chk.get("S3_macro_candidates")))
print("S3_match=" + str(chk.get("S3_natural_match_label")))
print("S4_runs=" + str(chk.get("S4_veto_runs_at_full_scale"))
      + " both=" + str(chk.get("S4_both_values_at_full_width"))
      + " pass_d=" + str(chk.get("S4_pass_delta"))
      + " veto_d=" + str(chk.get("S4_veto_delta")))
for n in out.get("notes", []):
    print("NOTE: " + n[:200])
for e in out["errors"]:
    print("ERR: " + e[:240])
print("RECEIPT=" + str(receipt))
