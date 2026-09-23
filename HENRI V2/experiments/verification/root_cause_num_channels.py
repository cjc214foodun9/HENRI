#!/usr/bin/env python3
"""ROOT CAUSE of the Sagnac width mismatch, and the CORRECT repair (measured).

THE CANDIDATE CAUSE, read from source
    opine_object_mcts.py:7   "temporal macro-options over su(3)^8192"
    opine_object_mcts.py:18  def __init__(self, num_channels: int = 8192,
                                              option_horizon: int = 4):
    opine_object_mcts.py:23  def construct_macro_option(self, generator_sequence,
                                                        device=None)
                      docstring: "... into one [num_channels, 3, 3] unitary
                                   macro-operator."

    So the macro field's channel count is an INSTANCE DEFAULT of 8192 and does not
    track the run's `num_blocks`. Chain to the observed failure:
        _u_macro = _opine.construct_macro_option(_gens, device=DEVICE)   # no num_channels
        _psi_macro = _trans.field_to_wave(_u_macro.unsqueeze(0)).squeeze(0)
        field_to_wave: [B, N, 3, 3] -> [B, N*8]
    With N = 8192 this yields 65536, while a num_blocks=64 run's references are 512.
    MEASURED production call: ((65536,), (512,), (512,))  <-- exactly this.

    AND: 8192 is ALSO `SCALE["num_blocks"]` at GPU scale. So on CUDA the hardcoded
    default coincidentally equals the run's block count and the mismatch DISAPPEARS.
    That is why d_model=65536 looked like "the configuration where it goes away" --
    the real variable is BLOCK COUNT, and the default merely happens to agree there.

WHAT THIS SCRIPT MEASURES
    R1  the runner's construction of the OPINE engine (does it pass num_channels?)
    R2  construct_macro_option's body (does it use self.num_channels?)
    R3  the width chain for N = 8192 (default) vs N = 64 (run-matched)
    R4  whether an N=64 field produces a 512-wide wave that MATCHES the run's refs
    R5  with refs and candidate matched at reduced scale, does the veto RUN and reach
        BOTH hard_vetoed values?  (This is the completion of Action 3's bidirectionality
        requirement, achieved by fixing the resolution instead of relaxing a gate.)
"""
from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
RUNNER = WT / "production_arc_run.py"
OPINE = WT / "opine_object_mcts.py"

out: dict = {"checks": {}, "errors": [], "notes": []}
chk = out["checks"]

# ------------------------------------------------------------------ R1
rl = RUNNER.read_text(encoding="utf-8", errors="ignore").splitlines()
ctor = [(i, l.strip()[:160]) for i, l in enumerate(rl, 1)
        if re.search(r"OPINEObjectMCTS\(", l)]
chk["R1_opine_ctor_sites"] = ctor
ctx = {}
for i, _ in ctor:
    ctx[str(i)] = [f"{k}: {rl[k-1]}" for k in range(max(1, i - 3),
                                                     min(len(rl), i + 6) + 1)]
chk["R1_ctor_context"] = ctx
# does the runner pass num_channels anywhere?
chk["R1_passes_num_channels"] = any("num_channels" in t for _, t in ctor)
chk["R1_passes_num_blocks"] = any("num_blocks" in t for _, t in ctor)

# ------------------------------------------------------------------ R2
ol = OPINE.read_text(encoding="utf-8", errors="ignore").splitlines()
body, cap = [], False
for i, l in enumerate(ol, 1):
    if l.strip().startswith("def construct_macro_option"):
        cap = True
    elif cap and l.strip().startswith("def ") and i > 1:
        break
    if cap:
        body.append(f"{i}: {l}")
    if len(body) > 46:
        break
chk["R2_construct_body"] = body
joined = "\n".join(body)
chk["R2_uses_self_num_channels"] = "self.num_channels" in joined
chk["R2_init_line"] = next((f"{i}: {l.strip()}" for i, l in enumerate(ol, 1)
                            if "num_channels" in l and "def __init__" in l), None)

# ------------------------------------------------------------------ R3/R4/R5
try:
    sys.path.insert(0, str(WT))
    import numpy as np
    import torch
    from opine_object_mcts import OPINEObjectMCTS
    from universal_data_transducer import SU3FieldWaveTransducer
    from henri_vision_encoder import HENRIVisionEncoder
    import math

    s3 = 1.0 / math.sqrt(3.0)
    lm = [[[0, 1, 0], [1, 0, 0], [0, 0, 0]],
          [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
          [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
          [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
          [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
          [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
          [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
          [[s3, 0, 0], [0, s3, 0], [0, 0, -2 * s3]]]
    basis = torch.tensor(lm, dtype=torch.complex64)
    trans = SU3FieldWaveTransducer(basis)

    def gens(n: int, seed: int = 9):
        g = torch.Generator().manual_seed(seed)
        out_ = []
        for _ in range(4):
            a = torch.randn(3, 3, generator=g, dtype=torch.complex64)
            a = a - a.conj().transpose(-2, -1)
            out_.append(torch.matrix_exp(0.05 * a))
        return out_

    widths = {}
    for label, nch in (("default_8192", None), ("run_matched_64", 64)):
        engine = OPINEObjectMCTS() if nch is None else OPINEObjectMCTS(num_channels=nch)
        u = engine.construct_macro_option(gens(4), device="cpu")
        chk[f"R3_{label}_field_shape"] = list(u.shape)
        w = trans.field_to_wave(u.unsqueeze(0)).squeeze(0)
        widths[label] = int(w.numel())
        chk[f"R3_{label}_wave_numel"] = int(w.numel())
        chk[f"R3_{label}_effective_num_channels"] = int(getattr(
            engine, "num_channels", -1))
    chk["R3_widths"] = widths

    # R4: what does a reduced-scale run's reference look like?
    enc = HENRIVisionEncoder(d_model=64 * 8, k_blocks=64, device="cpu")
    ref = enc.encode_grid(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
    chk["R4_ref_width_num_blocks_64"] = int(ref.numel())
    chk["R4_default_matches_ref"] = widths.get("default_8192") == int(ref.numel())
    chk["R4_run_matched_matches_ref"] = widths.get("run_matched_64") == int(ref.numel())
    if chk["R4_run_matched_matches_ref"]:
        out["notes"].append(
            "CONFIRMED: a num_channels=64 macro field produces a 512-wide wave that "
            "MATCHES a num_blocks=64 run's references, so the veto would run at "
            "reduced scale WITHOUT the diagnostic bridge.")
    if chk["R4_default_matches_ref"]:
        out["notes"].append("the default 8192 ALSO matches here; re-check the run scale")

    # R5: with matched widths, is hard_vetoed bidirectional at reduced scale?
    if chk["R4_run_matched_matches_ref"]:
        from sagnac_mcts_planner import SagnacMCTSPlanner
        pl = SagnacMCTSPlanner(d_model=64 * 8, k_blocks=64, tau_veto=0.35, device="cpu")
        W = int(ref.numel())
        u_pass = torch.zeros(W); u_pass[0] = 1.0
        r_ok = pl.dual_channel_sagnac_veto(u_pass, u_pass, u_pass,
                                           epsilon_hard=pl.tau_veto)
        r_no = pl.dual_channel_sagnac_veto(-u_pass, u_pass, u_pass,
                                           epsilon_hard=pl.tau_veto)
        chk["R5_runs_at_reduced_scale_matched"] = True
        chk["R5_pass_hard"] = bool(r_ok[2])
        chk["R5_veto_hard"] = bool(r_no[2])
        chk["R5_both_values"] = (not bool(r_ok[2])) and bool(r_no[2])
        chk["R5_pass_delta"] = round(float(r_ok[0]), 6)
        chk["R5_veto_delta"] = round(float(r_no[0]), 6)
        if not chk["R5_both_values"]:
            out["errors"].append("R5: matched reduced-scale widths did not yield both "
                                 "hard_vetoed values")
    else:
        out["errors"].append(
            "R4: a num_channels=64 field did NOT match the reduced run's reference "
            "width, so the plumbing hypothesis is not confirmed by this probe")
except Exception as e:  # noqa: BLE001
    out["errors"].append(f"R fatal: {type(e).__name__}: {e}")
    out["traceback_tail"] = traceback.format_exc()[-900:]

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "root_cause_num_channels.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("ROOTCAUSE=" + out["verdict"] + " errors=" + str(len(out["errors"])))
print("R1_ctor=" + json.dumps(ctor))
print("R1_passes_num_channels=" + str(chk.get("R1_passes_num_channels")))
print("R2_init=" + str(chk.get("R2_init_line")))
print("R2_uses_self=" + str(chk.get("R2_uses_self_num_channels")))
print("R3_widths=" + json.dumps(chk.get("R3_widths")))
print("R4_ref=" + str(chk.get("R4_ref_width_num_blocks_64"))
      + " default_match=" + str(chk.get("R4_default_matches_ref"))
      + " matched_match=" + str(chk.get("R4_run_matched_matches_ref")))
print("R5_both=" + str(chk.get("R5_both_values"))
      + " pass_d=" + str(chk.get("R5_pass_delta"))
      + " veto_d=" + str(chk.get("R5_veto_delta")))
for n in out.get("notes", []):
    print("NOTE: " + n[:200])
for e in out["errors"]:
    print("ERR: " + e[:220])
print("RECEIPT=" + str(receipt))
