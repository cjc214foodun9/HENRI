#!/usr/bin/env python3
"""ACTION 5: measure the WIDTH ARITHMETIC at full scale, isolated to the object under test.

WHY ISOLATED, NOT THE FULL PLANNER
    The question is a WIDTH question: at d_model=65536 / num_blocks=8192, do the
    candidate and both references share a width? That is answered by the SU(3)
    transducer plus the vision encoder -- the two objects that PRODUCE the widths.
    Constructing the whole SagnacMCTSPlanner at d_model=65536 on CPU additionally
    builds the egress transducer (763 MB overlay), the REPL, and the INTACT head,
    which is slow and memory-heavy without adding width information. Project policy
    also reserves full-scale SCORE-bearing runs for CUDA. So this measures the
    arithmetic and leaves a full-scale RUN as the separately gated step.

WHY IT MATTERS
    MEASURED earlier: state_wave width = num_blocks * 8 (512 at 64, 65536 at 8192).
    The live mismatch at reduced scale is candidate 65536 vs refs 512 = 128x.
    HYPOTHESIS: at num_blocks=8192 the refs become 65536-wide, so the mismatch
    DISAPPEARS and the veto runs with no bridge. FALSIFIABLE: if field_to_wave's
    output is NOT 65536 at that setting, or if it tracks d_model downward, the
    hypothesis fails and the cross-family boundary is the cause instead of scale.
"""
from __future__ import annotations

import inspect
import json
import sys
import time
import traceback
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
out: dict = {"checks": {}, "errors": [], "notes": []}
chk = out["checks"]

try:
    sys.path.insert(0, str(WT))
    import numpy as np
    import torch

    # ---------------------------------------------------------------- source
    src = (WT / "universal_data_transducer.py").read_text(encoding="utf-8", errors="ignore")
    lines = src.splitlines()
    chk["source_file"] = "universal_data_transducer.py"
    chk["source_total_lines"] = len(lines)
    chk["class_and_def_lines"] = [
        {"line": i, "text": l.strip()[:100]}
        for i, l in enumerate(lines, 1)
        if l.strip().startswith(("class SU3FieldWaveTransducer", "def field_to_wave",
                                 "def __init__"))
    ]
    # store the field_to_wave body verbatim (bounded) as the width-rule evidence
    fw_body = []
    started = False
    for i, l in enumerate(lines, 1):
        if l.strip().startswith("def field_to_wave"):
            started = True
        elif started and l.strip().startswith(("def ", "class ")) and i > 1:
            break
        if started:
            fw_body.append(f"{i}: {l}")
        if len(fw_body) > 44:
            break
    chk["field_to_wave_body"] = fw_body

    # ---------------------------------------------------------------- ctor sig
    from universal_data_transducer import SU3FieldWaveTransducer
    chk["A1_class_imported"] = True
    try:
        chk["A1_ctor_signature"] = str(inspect.signature(SU3FieldWaveTransducer.__init__))
    except Exception as e:  # noqa: BLE001
        chk["A1_ctor_signature_error"] = f"{type(e).__name__}: {e}"

    # ---------------------------------------------------------------- S3 transducer
    t0 = time.time()
    su3 = None
    try:
        su3 = SU3FieldWaveTransducer(d_model=65536)
        chk["S3_ctor_d_model_65536_ok"] = True
    except Exception as e:  # noqa: BLE001
        chk["S3_ctor_d_model_65536_error"] = f"{type(e).__name__}: {str(e)[:180]}"
        # try a no-arg / keyword-free construction as a fallback
        try:
            su3 = SU3FieldWaveTransducer()
            chk["S3_ctor_default_ok"] = True
        except Exception as e2:  # noqa: BLE001
            chk["S3_ctor_default_error"] = f"{type(e2).__name__}: {str(e2)[:180]}"
    chk["S3_ctor_seconds"] = round(time.time() - t0, 2)

    produced = {}
    if su3 is not None:
        for label, width in (("field_65536", 65536), ("field_512", 512)):
            try:
                f = torch.randn(1, width, generator=torch.Generator().manual_seed(5))
                w = su3.field_to_wave(f)
                produced[label] = {"input_width": width, "out_shape": list(w.shape),
                                   "out_numel": int(w.numel()), "out_dtype": str(w.dtype)}
            except Exception as e:  # noqa: BLE001
                produced[label] = {"input_width": width,
                                   "error": f"{type(e).__name__}: {str(e)[:140]}"}
    chk["S3_field_to_wave_outputs"] = produced

    # ---------------------------------------------------------------- S2 encoder
    enc_widths = {}
    from henri_vision_encoder import HENRIVisionEncoder
    grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for nb in (64, 8192):
        try:
            enc = HENRIVisionEncoder(d_model=nb * 8, k_blocks=nb, device="cpu")
            w = enc.encode_grid(grid)
            enc_widths[f"num_blocks_{nb}"] = {"numel": int(w.numel()),
                                              "shape": list(w.shape),
                                              "dtype": str(w.dtype)}
        except Exception as e:  # noqa: BLE001
            enc_widths[f"num_blocks_{nb}"] = {"error": f"{type(e).__name__}: {str(e)[:140]}"}
    chk["S2_encoder_widths"] = enc_widths

    # ---------------------------------------------------------------- verdict
    ref_full = enc_widths.get("num_blocks_8192", {}).get("numel")
    cand_full = produced.get("field_65536", {}).get("out_numel")
    chk["V_ref_width_at_8192"] = ref_full
    chk["V_candidate_width_at_65536_input"] = cand_full
    chk["V_widths_match_at_full_scale"] = bool(ref_full and cand_full
                                              and ref_full == cand_full)
    if chk["V_widths_match_at_full_scale"]:
        out["notes"].append(
            f"CONFIRMED: at num_blocks=8192 the reference is {ref_full}-wide and the "
            f"SU3 candidate is {cand_full}-wide -> EQUAL, so the veto runs with no "
            f"bridge. The reduced-scale mismatch is a SCALE artifact.")
    else:
        out["notes"].append(
            f"NOT CONFIRMED: reference={ref_full} candidate={cand_full}. If these "
            f"differ at full scale, the scale setting is not the cause and the "
            f"cross-family boundary is. Recorded, not assumed.")

    chk["elapsed_s"] = round(time.time() - t0, 2)

except Exception as e:  # noqa: BLE001
    out["errors"].append(f"FATAL: {type(e).__name__}: {e}")
    out["traceback_tail"] = traceback.format_exc()[-1000:]

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "action5_width_arithmetic.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("A5WIDTH=" + out["verdict"] + " errors=" + str(len(out["errors"])))
print("ctor_sig=" + str(chk.get("A1_ctor_signature"))[:120])
print("field_out=" + json.dumps(chk.get("S3_field_to_wave_outputs")))
print("enc_widths=" + json.dumps(chk.get("S2_encoder_widths")))
print("V_match=" + str(chk.get("V_widths_match_at_full_scale"))
      + " ref=" + str(chk.get("V_ref_width_at_8192"))
      + " cand=" + str(chk.get("V_candidate_width_at_65536_input")))
print("class_defs=" + json.dumps(chk.get("class_and_def_lines")))
for n in out.get("notes", []):
    print("NOTE: " + n[:220])
for e in out["errors"]:
    print("ERR: " + e[:220])
print("RECEIPT=" + str(receipt))
