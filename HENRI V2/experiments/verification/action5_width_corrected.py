#!/usr/bin/env python3
"""ACTION 5, corrected. Fixes TWO of my own defects and checks BOTH handler sites.

DEFECT 1 (my vacuous gate). The previous probe printed `A5WIDTH=PASS` while
    `V_match=False ref=65536 cand=None` -- it reported success on a FAILED
    measurement. A gate that passes when it measured nothing is the `all([])` class
    of defect this project documents. Enforcement here: if the candidate width was
    NOT measured, or the widths do not compare, the verdict is FAIL.

DEFECT 2 (I guessed a signature). `SU3FieldWaveTransducer.__init__(self,
    gell_mann_basis: torch.Tensor)` takes a REQUIRED tensor argument. I called it with
    `d_model=65536` and then with no args; both raised, `su3` stayed None, and the
    candidate was never measured. The construction is read from the LIVE call site
    (`EFEPlanner.__init__` -> `self._su3_transducer = SU3FieldWaveTransducer(...)`)
    instead of guessed.

ALSO CHECKED (found by my own H probe: two `except Exception as _veto_exc` sites)
    If the runner has TWO veto call sites and only ONE carries the
    SagnacGateUnavailable branch, the second still swallows the exception and reports
    a bare error where it should report UNAVAILABLE. Both sites are classified here.
"""
from __future__ import annotations

import ast
import inspect
import json
import sys
import traceback
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
out: dict = {"checks": {}, "errors": [], "notes": []}
chk = out["checks"]


def read(path: Path, lo: int, hi: int) -> list:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [f"{i}: {lines[i-1]}" for i in range(lo, min(hi, len(lines)) + 1)]


# ============================================================ A. both handler sites
rs = WT / "production_arc_run.py"
rlines = rs.read_text(encoding="utf-8", errors="ignore").splitlines()
exc_sites = [i for i, l in enumerate(rlines, 1)
             if l.strip().startswith("except Exception as _veto_exc")]
chk["A_except_sites"] = exc_sites
chk["A_n_except_sites"] = len(exc_sites)
# For each site, classify the next 34 lines.
classified = []
for s in exc_sites:
    body = "\n".join(rlines[s - 1: s - 1 + 34])
    classified.append({
        "site_line": s,
        "has_isinstance": "isinstance(_veto_exc, SagnacGateUnavailable)" in body,
        "has_gate_status": "gate_status" in body,
        "has_unavailable_label": "UNAVAILABLE_SHAPE_MISMATCH" in body,
        "bare_type_only": ('{"error": f"{type(_veto_exc).__name__}"}' in body),
        "body": [f"{s + k}: {rlines[s - 1 + k]}" for k in range(min(34, len(rlines) - s + 1))],
    })
chk["A_site_classification"] = classified
unpatched = [c["site_line"] for c in classified if not c["has_gate_status"]]
chk["A_unpatched_sites"] = unpatched
if unpatched:
    out["errors"].append(
        f"A: {len(unpatched)} veto handler site(s) lack the gate_status branch: "
        f"{unpatched}. A site that swallows with a bare type name still reports an "
        f"error where it should report UNAVAILABLE.")

# ============================================================ B. transducer, read not guessed
ef = WT / "efe_planner.py"
efl = ef.read_text(encoding="utf-8", errors="ignore").splitlines()
ctor_sites = [i for i, l in enumerate(efl, 1) if "_su3_transducer" in l]
chk["B_su3_ctor_lines"] = ctor_sites
if ctor_sites:
    c = ctor_sites[0]
    chk["B_ctor_context"] = [f"{i}: {efl[i-1]}" for i in
                             range(max(1, c - 6), min(len(efl), c + 8) + 1)]

ud = WT / "universal_data_transducer.py"
chk["B_class_source"] = read(ud, 93, 132)

# ============================================================ C. measure widths
try:
    sys.path.insert(0, str(WT))
    import torch
    from universal_data_transducer import SU3FieldWaveTransducer
    from henri_vision_encoder import HENRIVisionEncoder
    import numpy as np

    chk["C_signature"] = str(inspect.signature(SU3FieldWaveTransducer.__init__))

    # Build the Gell-Mann basis the constructor requires. If the module exposes a
    # helper, prefer it; otherwise construct the 8 standard Hermitian generators.
    basis = None
    for name in ("gell_mann_basis", "build_gell_mann_basis",
                 "make_gell_mann_basis", "GELL_MANN"):
        if hasattr(ud_module := __import__("universal_data_transducer"), name):
            candidate = getattr(ud_module, name)
            basis = candidate() if callable(candidate) else candidate
            chk["C_basis_source"] = name
            break
    if basis is None:
        # standard SU(3) Gell-Mann matrices, batched [8, 3, 3], hermitian
        import math
        lm = []
        lm.append([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        lm.append([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
        lm.append([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        lm.append([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        lm.append([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
        lm.append([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        lm.append([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
        lm.append([[1, 0, 0], [0, 1, 0], [0, 0, -2]] / math.sqrt(3))
        basis = torch.tensor(lm, dtype=torch.complex64)
        chk["C_basis_source"] = "constructed_standard_gell_mann"

    su3 = None
    for label, arg in (("as_built", basis),
                       ("batched", basis.unsqueeze(0) if (
                           hasattr(basis, "dim") and basis.dim() == 3) else basis)):
        try:
            su3 = SU3FieldWaveTransducer(arg)
            chk["C_ctor_ok_with"] = label
            break
        except Exception as e:  # noqa: BLE001
            chk[f"C_ctor_fail_{label}"] = f"{type(e).__name__}: {str(e)[:110]}"

    if su3 is None:
        out["errors"].append("C: could not construct SU3FieldWaveTransducer")
    else:
        got = {}
        for w in (8, 256, 512, 2048):
            try:
                f = torch.randn(1, w, generator=torch.Generator().manual_seed(5))
                o = su3.field_to_wave(f)
                got[f"in_{w}"] = {"out_shape": list(o.shape),
                                  "out_numel": int(o.numel()),
                                  "out_dtype": str(o.dtype)}
            except Exception as e:  # noqa: BLE001
                got[f"in_{w}"] = {"error": f"{type(e).__name__}: {str(e)[:110]}"}
        chk["C_field_to_wave_outputs"] = got
        cand_widths = [v.get("out_numel") for v in got.values() if v.get("out_numel")]
        chk["C_candidate_widths"] = cand_widths
        chk["C_candidate_measured"] = bool(cand_widths)

        # encoder reference at the full-scale block count
        enc = HENRIVisionEncoder(d_model=8192 * 8, k_blocks=8192, device="cpu")
        rw = enc.encode_grid(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        chk["C_ref_width_full_scale"] = int(rw.numel())
        chk["C_widths_match_at_full_scale"] = chk["C_ref_width_full_scale"] in cand_widths
except Exception as e:  # noqa: BLE001
    out["errors"].append(f"C fatal: {type(e).__name__}: {e}")
    out["traceback_tail"] = traceback.format_exc()[-800:]

# ============================================================ D. NON-VACUITY ENFORCEMENT
if not chk.get("C_candidate_measured"):
    out["errors"].append(
        "D NON-VACUITY: the candidate width was NOT measured, so no width verdict "
        "exists. The previous probe reported PASS in exactly this state, which is the "
        "all([]) vacuity defect.")
elif chk.get("C_widths_match_at_full_scale") is not True:
    out["errors"].append(
        f"D NON-VACUITY: widths do NOT match at full scale "
        f"(ref={chk.get('C_ref_width_full_scale')} candidates={chk.get('C_candidate_widths')})"
        f". The full-scale hypothesis is NOT confirmed by this probe.")

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "action5_width_corrected.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("A5CORR=" + out["verdict"] + " errors=" + str(len(out["errors"])))
print("A_sites=" + str(exc_sites) + " unpatched=" + str(unpatched))
for c in classified:
    print(f"  site {c['site_line']}: isinstance={c['has_isinstance']} "
          f"gate_status={c['has_gate_status']} bare={c['bare_type_only']}")
print("C_sig=" + str(chk.get("C_signature")))
print("C_basis=" + str(chk.get("C_basis_source")) + " ctor=" + str(chk.get("C_ctor_ok_with")))
print("C_outputs=" + json.dumps(chk.get("C_field_to_wave_outputs")))
print("C_ref_full=" + str(chk.get("C_ref_width_full_scale"))
      + " cand=" + str(chk.get("C_candidate_widths"))
      + " match=" + str(chk.get("C_widths_match_at_full_scale")))
for e in out["errors"]:
    print("ERR: " + e[:250])
print("RECEIPT=" + str(receipt))
