#!/usr/bin/env python3
"""Settle ONE question with indentation arithmetic: where is `opine_info` assigned?

WHY
    Two handlings of the OPINE block are both consistent with everything observed so far:
      (a) opine_info sits AFTER the try/except, inside `if sagnac_planner is not None:`
          -> defined on BOTH success and failure. Correct.
      (b) opine_info sits INSIDE the except handler
          -> defined ONLY when the veto raises. Then the moment the width mismatch is
             fixed and the veto SUCCEEDS, the downstream use of opine_info raises
             NameError. That is a latent bug that only appears after the fix.

    The gauntlet cannot distinguish them: in every run so far the veto RAISED (main
    checkout had no fixes; the worktree run hit the 65536-vs-512 width mismatch), so the
    except path executed either way.

    My previous AST check could not decide it, and I found why: it used ast.walk over
    the try BODY, which descends into NESTED try/except handlers, so it reported the
    name as present in both regions. Subtree walking is the wrong instrument.

    This uses raw leading-space counts, which are unambiguous and cheap to verify.

OUTPUT IS DELIBERATELY TINY (a few bytes per line) because my tool channel has been
returning corrupted, massively-duplicated renderings.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROD = Path(__file__).resolve().parents[2] / "production_arc_run.py"
lines = PROD.read_text(encoding="utf-8", errors="ignore").splitlines()


def indent_of(s: str) -> int:
    return len(s) - len(s.lstrip(" "))


marks = {"if_sagnac": None, "try": None, "except": None,
         "opine_info": [], "veto_error_assign": None, "print_opine": None}
for i, line in enumerate(lines, 1):
    s = line.strip()
    ind = indent_of(line)
    if s.startswith("if sagnac_planner is not None:") and marks["if_sagnac"] is None:
        marks["if_sagnac"] = (i, ind)
    elif s == "try:" and marks["try"] is None and i > (marks["if_sagnac"] or (0, 0))[0]:
        marks["try"] = (i, ind)
    elif s.startswith("except Exception as _veto_exc:") and marks["except"] is None:
        marks["except"] = (i, ind)
    elif s.startswith("opine_info = {"):
        marks["opine_info"].append((i, ind))
    elif s.startswith('_veto = {"error"'):
        marks["veto_error_assign"] = (i, ind)
    elif "engaged={opine_info" in s or "opine_info['engaged']" in s:
        if marks["print_opine"] is None:
            marks["print_opine"] = (i, ind)

res: dict = {"marks": marks}
verdict = "UNKNOWN"
if marks["opine_info"] and marks["except"] and marks["if_sagnac"]:
    ex_ind = marks["except"][1]
    if_ind = marks["if_sagnac"][1]
    placements = []
    for ln, ind in marks["opine_info"]:
        if ind > ex_ind:
            placements.append("INSIDE_EXCEPT")
        elif ind == if_ind + 4:
            placements.append("AFTER_TRY_INSIDE_IF")
        else:
            placements.append(f"OTHER(ind={ind},if_ind={if_ind},ex_ind={ex_ind})")
    res["placements"] = placements
    res["n_opine_info_assignments"] = len(marks["opine_info"])
    if "INSIDE_EXCEPT" in placements and len(placements) == 1:
        verdict = "BUG_ONLY_IN_EXCEPT"
    elif any(p == "AFTER_TRY_INSIDE_IF" for p in placements):
        verdict = "SAFE_OUTSIDE_TRY"
    elif len(placements) >= 2:
        verdict = "BOTH_PATHS"
res["verdict"] = verdict

out = Path(__file__).resolve().parent / "opine_placement.json"
out.write_text(json.dumps(res, indent=2), encoding="utf-8")
print("OPINE_PLACEMENT=" + verdict)
print("n_assign=" + str(len(marks["opine_info"])))
print("if_ind=" + str((marks["if_sagnac"] or (0, -1))[1]) +
      " try_ind=" + str((marks["try"] or (0, -1))[1]) +
      " exc_ind=" + str((marks["except"] or (0, -1))[1]))
print("opine_indents=" + ",".join(str(i) for _, i in marks["opine_info"]))
print("opine_lines=" + ",".join(str(l) for l, _ in marks["opine_info"]))
print("veto_error_ind=" + str((marks["veto_error_assign"] or (0, -1))[1]))
print("print_ind=" + str((marks["print_opine"] or (0, -1))[1]))
