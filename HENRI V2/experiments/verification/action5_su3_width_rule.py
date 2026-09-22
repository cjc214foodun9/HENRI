#!/usr/bin/env python3
"""ACTION 5, CORRECTED AND DECISIVE. Measure the SU3 width rule with the RIGHT input.

WHAT MY OWN READS ESTABLISHED
    `universal_data_transducer.py:93` defines:
        class SU3FieldWaveTransducer(torch.nn.Module):
            def __init__(self, gell_mann_basis: torch.Tensor)      # [8,3,3] complex
            def field_to_wave(self, su3_field) -> torch.Tensor:
                '''su3_field: [B, N, 3, 3] complex SU(3). Returns [B, N*8] ...'''
    So the OUTPUT WIDTH IS N*8, where N is the number of SU(3) blocks in the INPUT
    FIELD -- NOT d_model. That single line answers the Action 5 question and explains
    the observed reduced-scale mismatch.

    My previous attempt returned `field_out={}` because I passed a 2-D REAL tensor
    (`torch.randn(1, 65536)`) to a method expecting `[B, N, 3, 3]` COMPLEX. That was a
    probe defect, and it also let the probe report PASS while measuring nothing. Both
    defects are fixed here: the input is built to the DOCUMENTED CONTRACT, and a
    missing measurement is a FAILURE, not a pass.

THE PREDICTION BEING TESTED (falsifiable)
    Reduced scale (num_blocks=64):  refs = 512       ; SU3 field N=64  -> out 512
    Full scale   (num_blocks=8192): refs = 65536     ; SU3 field N=8192 -> out 65536
    => at FULL scale the widths MATCH and the veto runs with no bridge. If any
    measured width deviates from N*8, this prediction is FALSIFIED.

ALSO: classify BOTH `except Exception as _veto_exc` sites by reading the try BODY,
    so I do not report a false defect on a site that guards a different call.
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
out: dict = {"checks": {}, "errors": [], "notes": []}
chk = out["checks"]

# ============================================================ A. both handler sites
rl = (WT / "production_arc_run.py").read_text(encoding="utf-8", errors="ignore").splitlines()
sites = [i for i, l in enumerate(rl, 1)
         if l.strip().startswith("except Exception as _veto_exc")]
chk["A_sites"] = sites
site_info = []
for s in sites:
    # the try body is the run of lines ABOVE the except at the same indent
    ind_e = len(rl[s - 1]) - len(rl[s - 1].lstrip())
    body = []
    j = s - 2
    while j >= 0:
        line = rl[j]
        if line.strip() and (len(line) - len(line.lstrip())) < ind_e:
            break
        body.append(line)
        j -= 1
    body.reverse()
    body_txt = "\n".join(body)
    site_info.append({
        "site_line": s,
        "guards_planner_veto": "dual_channel_sagnac_veto" in body_txt,
        "guards_sidecar_evaluate_veto": "evaluate_veto" in body_txt,
        "guards_advisory_rerank": "apply_advisory_rerank" in body_txt,
        "body_head": [f"{s - len(body) + k}: {body[k]}" for k in range(min(6, len(body)))],
        "has_gate_status_after": "gate_status" in "\n".join(rl[s - 1: s + 34]),
        "records_message": "{_veto_exc}" in "\n".join(rl[s - 1: s + 30]),
    })
chk["A_site_info"] = site_info
# A site that guards the PLANNER veto must distinguish UNAVAILABLE from a real error.
need = [si["site_line"] for si in site_info
        if si["guards_planner_veto"] and not si["has_gate_status_after"]]
chk["A_planner_sites_missing_gate_status"] = need
if need:
    out["errors"].append(
        f"A: planner-veto handler site(s) {need} lack the gate_status branch")
# Report (not fail) sidecar sites: the sidecar returns VETO_UNAVAILABLE rather than
# raising, so a gate_status branch may be unnecessary there -- stated, not assumed.
sidecar = [si["site_line"] for si in site_info if si["guards_sidecar_evaluate_veto"]]
chk["A_sidecar_sites"] = sidecar

# ============================================================ B. width rule
try:
    sys.path.insert(0, str(WT))
    import torch
    from universal_data_transducer import SU3FieldWaveTransducer

    # --- obtain the basis the constructor requires -------------------------
    basis, src = None, None
    try:
        import efe_planner as ep
        if hasattr(ep, "GELL_MANN_BASIS"):
            basis, src = ep.GELL_MANN_BASIS, "efe_planner.GELL_MANN_BASIS"
    except Exception as e:  # noqa: BLE001
        chk["B_efe_import_error"] = f"{type(e).__name__}: {str(e)[:120]}"
    if basis is None:
        s3 = 1.0 / math.sqrt(3.0)
        lm = [
            [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
            [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
            [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
            [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
            [[s3, 0, 0], [0, s3, 0], [0, 0, -2 * s3]],
        ]
        # build as COMPLEX (the previous attempt divided a LIST by a float -> TypeError)
        basis = torch.tensor([[complex(v) for v in row] for row in lm],
                             dtype=torch.complex64) if False else torch.tensor(
            lm, dtype=torch.complex64)
        src = "constructed_standard_gell_mann"
    chk["B_basis_source"] = src
    chk["B_basis_shape"] = list(basis.shape)

    trans = SU3FieldWaveTransducer(basis)
    chk["B_constructed"] = True

    def su3_field(n_blocks: int, seed: int = 7) -> torch.Tensor:
        """[1, N, 3, 3] complex UNITARY field -- the documented input contract."""
        g = torch.Generator().manual_seed(seed)
        a = torch.randn(1, n_blocks, 3, 3, generator=g, dtype=torch.complex64)
        a = a - a.conj().transpose(-2, -1)          # anti-Hermitian
        return torch.matrix_exp(0.05 * a)           # unitary

    widths = {}
    for nb in (64, 512, 8192):
        try:
            f = su3_field(nb)
            w = trans.field_to_wave(f)
            widths[nb] = {"out_shape": list(w.shape), "out_numel": int(w.numel()),
                          "out_dtype": str(w.dtype),
                          "predicted_n_times_8": nb * 8,
                          "matches_prediction": int(w.numel()) == nb * 8}
        except Exception as e:  # noqa: BLE001
            widths[nb] = {"error": f"{type(e).__name__}: {str(e)[:140]}"}
    chk["B_field_to_wave_widths"] = {str(k): v for k, v in widths.items()}

    # --- encoder reference widths -----------------------------------------
    import numpy as np
    from henri_vision_encoder import HENRIVisionEncoder
    grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    enc = {}
    for nb in (64, 8192):
        try:
            e = HENRIVisionEncoder(d_model=nb * 8, k_blocks=nb, device="cpu")
            enc[nb] = int(e.encode_grid(grid).numel())
        except Exception as ex:  # noqa: BLE001
            enc[nb] = f"{type(ex).__name__}: {str(ex)[:100]}"
    chk["B_encoder_ref_widths"] = {str(k): v for k, v in enc.items()}

    # --- the decisive comparison ------------------------------------------
    def cand(nb):
        return widths.get(nb, {}).get("out_numel")

    chk["B_full_scale_ref"] = enc.get(8192)
    chk["B_full_scale_cand"] = cand(8192)
    chk["B_widths_match_at_full_scale"] = (isinstance(enc.get(8192), int)
                                           and cand(8192) == enc.get(8192))
    chk["B_reduced_scale_ref"] = enc.get(64)
    chk["B_reduced_scale_cand"] = cand(64)
    chk["B_widths_match_at_reduced_scale"] = (isinstance(enc.get(64), int)
                                              and cand(64) == enc.get(64))

    # NON-VACUITY: every measurement must exist or the verdict is FAIL.
    if cand(8192) is None or not isinstance(enc.get(8192), int):
        out["errors"].append(
            "B NON-VACUITY: the full-scale widths were NOT both measured, so no "
            "verdict exists. A previous probe reported PASS in exactly this state.")
    if widths and not all(v.get("matches_prediction") for v in widths.values()
                          if "out_numel" in v):
        out["errors"].append(
            f"B: the N*8 width rule is FALSIFIED by {chk['B_field_to_wave_widths']}")

    # --- does the veto RUN and is it BIDIRECTIONAL at full width? ----------
    if chk["B_widths_match_at_full_scale"]:
        from sagnac_mcts_planner import SagnacMCTSPlanner
        pl = SagnacMCTSPlanner(d_model=8192 * 8, k_blocks=8192, tau_veto=0.35,
                               device="cpu")
        W = int(enc[8192])
        u = torch.zeros(W); u[0] = 1.0
        r_ok = pl.dual_channel_sagnac_veto(u, u, u, epsilon_hard=pl.tau_veto)
        r_no = pl.dual_channel_sagnac_veto(-u, u, u, epsilon_hard=pl.tau_veto)
        chk["B_full_width_veto_runs"] = True
        chk["B_full_width_pass_hard"] = bool(r_ok[2])
        chk["B_full_width_veto_hard"] = bool(r_no[2])
        chk["B_full_width_both_values"] = (not bool(r_ok[2])) and bool(r_no[2])
        chk["B_full_width_pass_delta"] = round(float(r_ok[0]), 6)
        chk["B_full_width_veto_delta"] = round(float(r_no[0]), 6)
        if not chk["B_full_width_both_values"]:
            out["errors"].append(
                "B: at FULL width the veto did not reach both values -> the gate is "
                "still not bidirectional where it matters")
    chk["B_notes"] = ("At full scale the reference is 65536-wide and an 8192-block "
                      "SU3 field is also 65536-wide, so no bridge is needed.")
except Exception as e:  # noqa: BLE001
    out["errors"].append(f"B fatal: {type(e).__name__}: {e}")
    out["traceback_tail"] = traceback.format_exc()[-900:]

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "action5_su3_width_rule.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("A5=" + out["verdict"] + " errors=" + str(len(out["errors"])))
print("A_sites=" + str(sites))
for si in site_info:
    print(f"  site {si['site_line']}: planner={si['guards_planner_veto']} "
          f"sidecar={si['guards_sidecar_evaluate_veto']} "
          f"gate_status={si['has_gate_status_after']} msg={si['records_message']}")
print("A_missing_gate_status=" + str(need))
print("B_basis=" + str(chk.get("B_basis_source")) + " shape=" + str(chk.get("B_basis_shape")))
print("B_widths=" + json.dumps(chk.get("B_field_to_wave_widths")))
print("B_enc=" + json.dumps(chk.get("B_encoder_ref_widths")))
print("B_full: ref=" + str(chk.get("B_full_scale_ref"))
      + " cand=" + str(chk.get("B_full_scale_cand"))
      + " MATCH=" + str(chk.get("B_widths_match_at_full_scale")))
print("B_reduced: ref=" + str(chk.get("B_reduced_scale_ref"))
      + " cand=" + str(chk.get("B_reduced_scale_cand"))
      + " match=" + str(chk.get("B_widths_match_at_reduced_scale")))
print("B_both_values=" + str(chk.get("B_full_width_both_values"))
      + " pass_d=" + str(chk.get("B_full_width_pass_delta"))
      + " veto_d=" + str(chk.get("B_full_width_veto_delta")))
for e in out["errors"]:
    print("ERR: " + e[:240])
print("RECEIPT=" + str(receipt))
