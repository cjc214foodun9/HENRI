#!/usr/bin/env python3
"""Verify the width-contract fix I just applied, and settle the handler structure.

WHY A RECEIPT FILE
    My tool channel returned corrupted output twice (mass duplication plus
    reasoning-like text and verbatim spec text injected into tool results). So this
    probe writes a JSON receipt and prints ONE compact line. If the printed line and
    the receipt disagree, or the receipt is malformed, the channel is still suspect.

WHAT IS CHECKED
    V1  SagnacGateUnavailable exists and is a RuntimeError subclass
    V2  mismatched widths RAISE SagnacGateUnavailable (not bare RuntimeError)
    V3  matched widths still return the normal 3-tuple (fix did not break the happy path)
    V4  AST: is `opine_info = {...}` INSIDE or OUTSIDE the except handler?
        If INSIDE, telemetry would be written only on veto FAILURE -- a real bug.
    V5  AST: does the handler record `gate_status` for SagnacGateUnavailable and
        `error` otherwise?
    V6  the handler must NOT put a `hard_vetoed` key on the unavailable branch
        (so availability cannot be mistaken for permission)
"""
from __future__ import annotations

import ast
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PROD = ROOT / "production_arc_run.py"

receipt: dict = {"checks": {}, "errors": []}


def note(k, v):
    receipt["checks"][k] = v


# ------------------------------------------------------------------ V1/V2/V3
try:
    import torch
    from sagnac_mcts_planner import SagnacMCTSPlanner, SagnacGateUnavailable

    note("V1_class_exists", True)
    note("V1_is_runtimeerror", issubclass(SagnacGateUnavailable, RuntimeError))

    p = SagnacMCTSPlanner(d_model=512, k_blocks=64, tau_veto=0.35, device="cpu")

    # V2: production shapes -> must raise the NAMED type
    big = torch.randn(65536, generator=torch.Generator().manual_seed(0))
    big = big.to(torch.complex64)
    small = torch.randn(512, generator=torch.Generator().manual_seed(1))
    try:
        p.dual_channel_sagnac_veto(big, small, small, epsilon_hard=p.tau_veto)
        note("V2_raised", None)
        receipt["errors"].append("V2: mismatched widths did NOT raise")
    except SagnacGateUnavailable as e:
        note("V2_raised_named", f"{type(e).__name__}")
        note("V2_message", str(e)[:200])
    except Exception as e:  # noqa: BLE001
        note("V2_raised_other", f"{type(e).__name__}: {e}")
        receipt["errors"].append(f"V2: raised {type(e).__name__}, not the named type")

    # V3: matched widths -> normal tuple
    m = torch.randn(512, generator=torch.Generator().manual_seed(2))
    out = p.dual_channel_sagnac_veto(m, m, m, epsilon_hard=p.tau_veto)
    note("V3_matched_returns_tuple", isinstance(out, tuple) and len(out) == 3)
    note("V3_matched_delta_axiom", round(float(out[0]), 9))
    note("V3_matched_hard", bool(out[2]))
    if not (isinstance(out, tuple) and len(out) == 3):
        receipt["errors"].append("V3: matched widths did not return a 3-tuple")
    if abs(float(out[0])) > 1e-6:
        receipt["errors"].append(f"V3: identical waves gave delta {out[0]}, expected ~0")
except Exception as e:  # noqa: BLE001
    note("V1_V3_fatal", f"{type(e).__name__}: {e}")
    receipt["errors"].append(traceback.format_exc()[-600:])

# ------------------------------------------------------------------ V4/V5/V6
try:
    tree = ast.parse(PROD.read_text(encoding="utf-8"))
    # locate the OPINE try/except that contains the veto call
    target_try = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            src = ast.unparse(node)
            if "dual_channel_sagnac_veto" in src:
                target_try = node
                break
    if target_try is None:
        receipt["errors"].append("V4: could not locate the OPINE try/except")
    else:
        note("V4_try_lineno", target_try.lineno)
        # names assigned directly in the try BODY (top level)
        def assigned_names(body):
            names = set()
            for st in body:
                for n in ast.walk(st):
                    if isinstance(n, ast.Assign):
                        for t in n.targets:
                            if isinstance(t, ast.Name):
                                names.add(t.id)
            return names
        try_body_names = assigned_names(target_try.body)
        # names assigned in the except handler, top level only
        handler = target_try.handlers[0]
        handler_top_names = set()
        for st in handler.body:
            for n in ast.walk(st):
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        if isinstance(t, ast.Name):
                            handler_top_names.add(t.id)
        note("V4_try_body_assigns_opine_info", "opine_info" in try_body_names)
        note("V4_handler_top_assigns_opine_info", "opine_info" in handler_top_names)
        if "opine_info" not in try_body_names:
            receipt["errors"].append(
                "V4 BUG: opine_info is NOT assigned in the try body; telemetry would "
                "be written only on veto failure")
        note("V4_handler_lineno", handler.lineno)
        # V5/V6: inspect the handler source for the two branches
        hsrc = ast.unparse(handler)
        note("V5_records_gate_status", "gate_status" in hsrc)
        note("V5_records_error", '"error"' in hsrc or "'error'" in hsrc)
        note("V5_checks_class_name", "SagnacGateUnavailable" in hsrc)
        # V6: the unavailable branch must not set hard_vetoed
        note("V6_unavailable_branch_has_hard_vetoed",
             "UNAVAILABLE_SHAPE_MISMATCH" in hsrc and
             hsrc.split("gate_status")[1][:200].count("hard_vetoed") > 0)
        if not note("V5_records_gate_status", False):
            receipt["errors"].append("V5: handler does not record gate_status")
        if not note("V5_checks_class_name", False):
            receipt["errors"].append("V5: handler does not match SagnacGateUnavailable")
except Exception as e:  # noqa: BLE001
    receipt["errors"].append(f"V4 AST error: {type(e).__name__}: {e}")

receipt["verdict"] = "PASS" if not receipt["errors"] else "FAIL"
out = Path(__file__).resolve().parent / "width_contract_verify.json"
out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print(f"WIDTH_CONTRACT_VERDICT={receipt['verdict']} errors={len(receipt['errors'])} "
      f"receipt={out}")
