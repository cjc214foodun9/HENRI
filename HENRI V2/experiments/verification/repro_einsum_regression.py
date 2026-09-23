#!/usr/bin/env python3
"""Regenerate the einsum failure IN ISOLATION and print its traceback.

THE REGRESSION (named by name_postfix_error.py)
    64x  RuntimeError: einsum(): subscript n has size 8192 for operand 1 which does
         not broadcast with previously seen size 64
    It appeared AFTER binding the macro-field resolution to SCALE["num_blocks"], i.e.
    after the field became 64-wide at CPU scale. So the width mismatch in
    dual_channel_sagnac_veto is genuinely gone, and a SECOND component still carries
    8192 in the macro-option path.

WHY IN ISOLATION
    Re-running the whole gauntlet and grepping logs gives the message but not the
    file:line. Replaying the exact production chain at 64 channels raises the SAME
    exception with a full traceback, at CPU cost, once.

CHAIN REPRODUCED (from production_arc_run.py ~2246-2250, verified by direct read)
    _gens = [action_outcome_store.lie_element(a % num_actions, gm_basis)[0] for a in ...]
    _u_macro = _opine.construct_macro_option(_gens, device=DEVICE)

SCOPE NOTE
    SCALE["num_blocks"] == 8192 at GPU scale, so at production scale the second site
    is also 8192 and agrees. Everything measured here is the REDUCED-SCALE path. The
    fix is a CPU-path correctness fix, not a production behaviour change.
"""
from __future__ import annotations

import inspect
import json
import math
import re
import sys
import traceback
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
out: dict = {"checks": {}, "errors": [], "traceback": None}
chk = out["checks"]

try:
    sys.path.insert(0, str(WT))
    import torch

    # ---- build the Gell-Mann basis the chain needs
    try:
        from chromodynamic_grounding import GELL_MANN_BASIS
        basis = GELL_MANN_BASIS
        chk["basis_source"] = "chromodynamic_grounding.GELL_MANN_BASIS"
    except Exception as e:  # noqa: BLE001
        chk["basis_error"] = f"{type(e).__name__}: {e}"
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
        chk["basis_source"] = "constructed_standard_gell_mann"
    chk["basis_shape"] = list(basis.shape)

    from henri_external_outcome_refactor_module import ActionOutcomeGeneratorStore
    from opine_object_mcts import OPINEObjectMCTS

    # ---- does lie_element use self.num_channels, or a literal?
    try:
        src = inspect.getsource(ActionOutcomeGeneratorStore.lie_element)
        chk["lie_element_uses_self_num_channels"] = "self.num_channels" in src
        chk["lie_element_literals_8192"] = bool(re.search(r"\b8192\b", src))
        chk["lie_element_source"] = src.splitlines()[:26]
    except Exception as e:  # noqa: BLE001
        chk["lie_element_introspect_error"] = f"{type(e).__name__}: {e}"

    # ---- replay the chain at 64 channels (the fixed reduced-scale resolution)
    for nch in (64, 8192):
        rec: dict = {"num_channels": nch}
        try:
            store = ActionOutcomeGeneratorStore(num_actions=8, num_channels=nch,
                                                lr=0.1)
            rec["store_ctor_ok"] = True
            gens = [store.lie_element(a % 8, basis)[0] for a in range(4)]
            rec["gens_shapes"] = [list(g.shape) for g in gens]
            opine = OPINEObjectMCTS(num_channels=nch, option_horizon=4)
            u = opine.construct_macro_option(gens, device="cpu")
            rec["macro_field_shape"] = list(u.shape)
            rec["chain_ok"] = True
        except Exception as e:  # noqa: BLE001
            rec["chain_ok"] = False
            rec["error_type"] = type(e).__name__
            rec["error_msg"] = str(e)[:300]
            tb = traceback.format_exc()
            rec["traceback_tail"] = tb[-1400:]
            if out["traceback"] is None:
                out["traceback"] = tb[-1600:]
        chk[f"chain_{nch}"] = rec
except Exception as e:  # noqa: BLE001
    out["errors"].append(f"fatal: {type(e).__name__}: {e}")
    out["traceback"] = traceback.format_exc()[-1200:]

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "einsum_regression_repro.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("REPRO=" + out["verdict"])
print("basis=" + str(chk.get("basis_source")) + " shape=" + str(chk.get("basis_shape")))
print("lie_element_uses_self_num_channels=" + str(chk.get("lie_element_uses_self_num_channels")))
print("lie_element_has_8192_literal=" + str(chk.get("lie_element_literals_8192")))
for nch in (64, 8192):
    r = chk.get(f"chain_{nch}", {})
    print(f"  chain num_channels={nch}: ok={r.get('chain_ok')} "
          f"gens={r.get('gens_shapes')} field={r.get('macro_field_shape')}")
    if not r.get("chain_ok"):
        print(f"      {r.get('error_type')}: {r.get('error_msg')}")
if chk.get("lie_element_source"):
    print("  lie_element source (head):")
    for l in chk["lie_element_source"][:14]:
        print("    " + l[:120])
if out.get("traceback"):
    print("=" * 70)
    print("TRACEBACK (names the failing file:line):")
    print(out["traceback"])
print("RECEIPT=" + str(receipt))
