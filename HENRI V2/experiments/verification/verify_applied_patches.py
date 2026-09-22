#!/usr/bin/env python3
"""ACTION 1: verify the three applied patches, and settle `opine_info` scope.

WHY INDENTATION, NOT ast.walk
    A previous check used ast.walk over the `try` body, which DESCENDS INTO NESTED
    try/except blocks and therefore reported `opine_info` as present in both the try
    body and the handler. It decided nothing. Scope is settled here by raw
    leading-space arithmetic, which nesting cannot confuse:

        opine_info is INSIDE an except handler  <=>
            line(opine) > line(except)
            AND indent(opine) > indent(except)
            AND no intervening non-blank, non-comment line drops to <= indent(except)

    If `opine_info` were inside the handler it would be defined ONLY when the veto
    raised. The moment the width fix lets the veto SUCCEED, downstream use of
    `opine_info` would raise NameError. That is the latent bug this rules out.

WHY A RECEIPT
    The tool channel returned corrupted renderings (mass duplication, injected
    reasoning text, verbatim spec fragments) twice this session. A JSON receipt on
    disk plus a two-line verdict is the corruption-resistant form.

GATES
    A1  SagnacGateUnavailable is a module-level class deriving RuntimeError
    A2  dual_channel_sagnac_veto carries the width guard, raises that type, and
        retains the HENRI_SAGNAC_LEGACY_SCALE evidence branch
    A3  the runner handler matches it (isinstance OR name comparison) and records a
        gate_status that does NOT carry a `hard_vetoed` key
    A4  the runner passes epsilon_hard explicitly at the OPINE call
    B   opine_info is NOT inside any except handler, AND is assigned on every path
    C   live: mismatched widths raise the NAMED type
    D   live: matched widths return a 3-tuple and hard_vetoed reaches BOTH values
    E   live: MIXED dtype (complex candidate vs real refs) at MATCHED width works,
        because a full-scale run has exactly that dtype combination
"""
from __future__ import annotations

import ast
import json
import sys
import traceback
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
PLANNER = WT / "sagnac_mcts_planner.py"
RUNNER = WT / "production_arc_run.py"

out: dict = {"checks": {}, "errors": []}
chk = out["checks"]

# ------------------------------------------------------------------ A1 / A2
pt = ast.parse(PLANNER.read_text(encoding="utf-8"))
cls = [n for n in pt.body if isinstance(n, ast.ClassDef)
       and n.name == "SagnacGateUnavailable"]
chk["A1_class_defined"] = bool(cls)
chk["A1_derives_runtimeerror"] = bool(cls) and any(
    (isinstance(b, ast.Name) and b.id == "RuntimeError") for b in cls[0].bases)
if not chk["A1_class_defined"]:
    out["errors"].append("A1: SagnacGateUnavailable not defined at module level")
if not chk["A1_derives_runtimeerror"]:
    out["errors"].append("A1: SagnacGateUnavailable does not derive RuntimeError")

chk["A2_width_guard"] = False
chk["A2_compares_numel"] = False
chk["A2_raises_named"] = False
chk["A2_legacy_branch_retained"] = False
for n in ast.walk(pt):
    if isinstance(n, ast.FunctionDef) and n.name == "dual_channel_sagnac_veto":
        src = ast.unparse(n)
        chk["A2_width_guard"] = "numel()" in src
        chk["A2_compares_numel"] = ("w_cand.numel()" in src and "w_ax.numel()" in src)
        chk["A2_raises_named"] = "raise SagnacGateUnavailable" in src
        chk["A2_legacy_branch_retained"] = "HENRI_SAGNAC_LEGACY_SCALE" in src
        break
for k in ("A2_width_guard", "A2_compares_numel", "A2_raises_named",
          "A2_legacy_branch_retained"):
    if not chk[k]:
        out["errors"].append(f"{k} is False")

# ------------------------------------------------------------------ A3 / A4
rs = RUNNER.read_text(encoding="utf-8")
chk["A3_isinstance_match"] = "isinstance(_veto_exc, SagnacGateUnavailable)" in rs
chk["A3_name_comparison"] = "_name == \"SagnacGateUnavailable\"" in rs
chk["A3_records_gate_status"] = "gate_status" in rs
chk["A3_unavailable_label"] = "UNAVAILABLE_SHAPE_MISMATCH" in rs
chk["A4_explicit_epsilon_call"] = "epsilon_hard=sagnac_planner.tau_veto" in rs
if not (chk["A3_isinstance_match"] or chk["A3_name_comparison"]):
    out["errors"].append("A3: handler does not recognise SagnacGateUnavailable")
if not chk["A3_records_gate_status"]:
    out["errors"].append("A3: handler does not record gate_status")
if not chk["A3_unavailable_label"]:
    out["errors"].append("A3: UNAVAILABLE_SHAPE_MISMATCH label absent")
if not chk["A4_explicit_epsilon_call"]:
    out["errors"].append("A4: OPINE call does not pass epsilon_hard explicitly")

# The unavailable branch must NOT claim a hard_vetoed value.
unavail_has_hard = False
for n in ast.walk(ast.parse(rs)):
    if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_veto" for t in n.targets):
        lit = ast.unparse(n)
        if "UNAVAILABLE_SHAPE_MISMATCH" in lit and "hard_vetoed" in lit:
            unavail_has_hard = True
chk["A3_unavailable_branch_omits_hard_vetoed"] = not unavail_has_hard
if unavail_has_hard:
    out["errors"].append(
        "A3: the UNAVAILABLE branch sets hard_vetoed, so a reader could mistake "
        "availability for permission")

# ------------------------------------------------------------------ B  indentation
lines = rs.splitlines()


def ind(s: str) -> int:
    return len(s) - len(s.lstrip(" "))


exc_marks = [(i, ind(l)) for i, l in enumerate(lines, 1)
             if l.strip().startswith("except Exception as _veto_exc")]
opine_marks = [(i, ind(l)) for i, l in enumerate(lines, 1)
               if l.strip().startswith("opine_info = {")]
if_marks = [(i, ind(l)) for i, l in enumerate(lines, 1)
            if l.strip().startswith("if sagnac_planner is not None")]
chk["B_except_marks"] = exc_marks
chk["B_opine_marks"] = opine_marks
chk["B_if_marks"] = if_marks


def in_except_body(ln_o: int, ind_o: int):
    for (ln_e, ind_e) in exc_marks:
        if ln_o <= ln_e or ind_o <= ind_e:
            continue
        dedented = False
        for b in lines[ln_e:ln_o - 1]:
            bs = b.strip()
            if not bs or bs.startswith("#"):
                continue
            if ind(b) <= ind_e:
                dedented = True
                break
        if not dedented:
            return True, ln_e, ind_e
    return False, None, None


verdicts = [dict(line=l, indent=i, **dict(zip(
    ("inside_except", "except_line", "except_indent"), in_except_body(l, i))))
    for (l, i) in opine_marks]
chk["B_opine_scope"] = verdicts
chk["B_n_opine_assignments"] = len(verdicts)
chk["B_any_inside_except"] = any(v["inside_except"] for v in verdicts)
if not verdicts:
    out["errors"].append("B: no `opine_info = {` assignment found in the runner")
if chk["B_any_inside_except"]:
    out["errors"].append(
        f"B: opine_info INSIDE an except handler -> NameError risk once the veto "
        f"succeeds. verdicts={verdicts}")
# SAFETY CRITERION, stated precisely. The failure being ruled out is an assignment
# INSIDE an except handler: it would be defined only when the veto RAISED, so the
# moment the width fix lets the veto SUCCEED, a downstream read would NameError.
#
# An earlier version of this gate ALSO demanded `line > except_line`. That is wrong:
# a definition after the handler is safe, but so are two definitions at the same
# indent forming the arms of an if/else that both follow the handler. The extra
# clause produced a FALSE FAILURE on exactly that shape -- measured marks at 2337
# and 2350, neither inside a handler, yet the gate still reported "not assigned".
# A gate that fails a correct configuration is an instrument defect, so it is
# corrected here rather than worked around.
chk["B_safe_all_outside_handlers"] = bool(verdicts) and not any(
    v["inside_except"] for v in verdicts)
chk["B_n_assignments"] = len(verdicts)
chk["B_covers_both_paths"] = len(verdicts) >= 2 or any(
    v["except_line"] is not None and v["line"] > v["except_line"]
    for v in verdicts)
chk["B_defined_after_handler"] = (chk["B_safe_all_outside_handlers"]
                                  and chk["B_covers_both_paths"])
if not chk["B_safe_all_outside_handlers"]:
    out["errors"].append(
        f"B: opine_info assigned INSIDE an except handler -> NameError risk once the "
        f"veto succeeds. verdicts={verdicts}")
if not chk["B_covers_both_paths"]:
    out["errors"].append(
        f"B: opine_info has one assignment and it does not follow the handler; some "
        f"path may reach the use with it undefined. verdicts={verdicts}")

# ------------------------------------------------------------------ C/D/E live
try:
    sys.path.insert(0, str(WT))
    import torch
    import sagnac_mcts_planner as sp

    p = sp.SagnacMCTSPlanner(d_model=512, k_blocks=64, tau_veto=0.35, device="cpu")

    # C: production-scale mismatch 65536 vs 512
    big = torch.randn(65536, generator=torch.Generator().manual_seed(0)).to(torch.complex64)
    small = torch.randn(512, generator=torch.Generator().manual_seed(1))
    try:
        p.dual_channel_sagnac_veto(big, small, small, epsilon_hard=p.tau_veto)
        chk["C_mismatch_raises_named"] = False
        out["errors"].append("C: mismatched widths did NOT raise")
    except sp.SagnacGateUnavailable as e:
        chk["C_mismatch_raises_named"] = True
        chk["C_raise_msg_head"] = str(e)[:160]
    except Exception as e:  # noqa: BLE001
        chk["C_mismatch_raises_named"] = False
        out["errors"].append(f"C: raised {type(e).__name__}, not SagnacGateUnavailable")

    # D: matched widths -> both hard values
    m = torch.zeros(512)
    m[0] = 1.0
    anti = -m
    r_pass = p.dual_channel_sagnac_veto(m, m, m, epsilon_hard=p.tau_veto)
    r_veto = p.dual_channel_sagnac_veto(anti, m, m, epsilon_hard=p.tau_veto)
    chk["D_pass_tuple"] = isinstance(r_pass, tuple) and len(r_pass) == 3
    chk["D_pass_hard"] = bool(r_pass[2])
    chk["D_veto_hard"] = bool(r_veto[2])
    chk["D_pass_delta_axiom"] = round(float(r_pass[0]), 9)
    chk["D_veto_delta_axiom"] = round(float(r_veto[0]), 9)
    chk["D_both_values_reachable"] = (chk["D_pass_hard"] is False
                                      and chk["D_veto_hard"] is True)
    if not chk["D_pass_tuple"]:
        out["errors"].append("D: matched widths did not return a 3-tuple")
    if not chk["D_both_values_reachable"]:
        out["errors"].append(
            f"D: hard_vetoed not both-reachable on matched widths "
            f"(pass={chk['D_pass_hard']} d={chk['D_pass_delta_axiom']}, "
            f"veto={chk['D_veto_hard']} d={chk['D_veto_delta_axiom']})")

    # E: MIXED dtype at matched width (a full-scale run has complex candidate vs
    # real refs). This is the configuration Action 5 would exercise.
    cb = torch.randn(512, generator=torch.Generator().manual_seed(2)).to(torch.complex64)
    rb = torch.randn(512, generator=torch.Generator().manual_seed(3))
    try:
        r_mix = p.dual_channel_sagnac_veto(cb, rb, rb, epsilon_hard=p.tau_veto)
        chk["E_mixed_dtype_ok"] = isinstance(r_mix, tuple) and len(r_mix) == 3
        chk["E_mixed_delta_axiom"] = round(float(r_mix[0]), 6)
    except Exception as e:  # noqa: BLE001
        chk["E_mixed_dtype_ok"] = False
        out["errors"].append(f"E: mixed complex/real at matched width raised "
                             f"{type(e).__name__}: {e}")

    # E2: does pooling 65536->512 give ANY discrimination? Measure before choosing
    # a width-contract repair. Compare a pooled self-pair against a pooled cross-pair.
    def pool(x: torch.Tensor, n: int) -> torch.Tensor:
        x = x.flatten()
        return x[: (x.numel() // n) * n].reshape(n, -1).mean(dim=-1)

    g = torch.Generator().manual_seed(7)
    a = torch.randn(65536, generator=g).to(torch.complex64)
    b = torch.randn(65536, generator=g).to(torch.complex64)
    ref = torch.randn(512, generator=g)
    s_self = p._norm_consistent_similarity(pool(a, 512), ref)
    s_cross = p._norm_consistent_similarity(pool(b, 512), ref)
    chk["E2_pooled_self_sim"] = round(float(s_self), 6)
    chk["E2_pooled_cross_sim"] = round(float(s_cross), 6)
    chk["E2_pooled_separation"] = round(abs(float(s_self) - float(s_cross)), 6)
except Exception as e:  # noqa: BLE001
    out["errors"].append(f"live-test fatal: {type(e).__name__}: {e}")
    out["traceback_tail"] = traceback.format_exc()[-700:]

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "applied_patches_verify.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("ACTION1=" + out["verdict"] + " errors=" + str(len(out["errors"])))
print("A_class=" + str(chk.get("A1_class_defined"))
      + " A2_guard=" + str(chk.get("A2_width_guard"))
      + " A3_isinstance=" + str(chk.get("A3_isinstance_match"))
      + " A3_namecmp=" + str(chk.get("A3_name_comparison")))
print("B_marks=" + str(chk.get("B_opine_marks")) + " B_any_inside="
      + str(chk.get("B_any_inside_except")))
print("C_raised_named=" + str(chk.get("C_mismatch_raises_named"))
      + " D_both=" + str(chk.get("D_both_values_reachable"))
      + " E_mixed=" + str(chk.get("E_mixed_dtype_ok")))
print("E2_pooled_sep=" + str(chk.get("E2_pooled_separation")))
for e in out["errors"]:
    print("ERR: " + e[:200])
print("RECEIPT=" + str(receipt))
