#!/usr/bin/env python3
"""Close ACTION 5's diagnosis: find the SECOND coupled resolution mismatch.

WHAT IS ALREADY ESTABLISHED (all measured, own receipts)
    1. The Sagnac veto raised on every live call at reduced scale because the macro
       field's resolution did not track the run's block count:
           num_channels 8192 -> field [8192,3,3] -> field_to_wave -> 65536-wide wave
           references in a num_blocks=64 run                           512-wide
       MEASURED: 60/60 payloads gate_status=UNAVAILABLE_SHAPE_MISMATCH.
    2. With num_channels=64 the wave is 512-wide, MATCHES, and the veto runs and
       reaches BOTH hard_vetoed values with NO bridge.
    3. BUT enabling that alone DISPLACES the failure. 64/64 steps then raise:
           RuntimeError: einsum(): subscript n has size 8192 for operand 1 which does
           not broadcast with previously seen size 64
    So a SECOND component still carries 8192 at the same boundary.

WHAT THIS SCRIPT DOES
    Locate the second operand's source by reading the einsum's ENCLOSING FUNCTION and
    finding which parameter carries N, then tracing that parameter back to its
    producer in the runner. Reported with file:line, not inferred.

    It also re-verifies that the default path is UNCHANGED (the flag is off unless
    explicitly set), because the project rule is that a default path is preserved
    unless an experiment explicitly changes it.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
RUNNER = WT / "production_arc_run.py"
OPINE = WT / "opine_object_mcts.py"
STORE = WT / "henri_external_outcome_refactor_module.py"

out: dict = {"checks": {}, "errors": [], "notes": []}
chk = out["checks"]


def lines_of(p: Path) -> list:
    return p.read_text(encoding="utf-8", errors="ignore").splitlines()


# ============================================================ 1. enclosing function
ol = lines_of(OPINE)
target = None
for i, l in enumerate(ol, 1):
    if 'einsum("nij,njk->nik"' in l or "einsum('nij,njk->nik'" in l:
        target = i
        break
chk["second_einsum_line"] = target
if target:
    # walk back to the enclosing def
    def_line = None
    for j in range(target, 0, -1):
        if re.match(r"\s*def\s+\w+", ol[j - 1]):
            def_line = j
            break
    chk["enclosing_def_line"] = def_line
    if def_line:
        body = []
        indent = len(ol[def_line - 1]) - len(ol[def_line - 1].lstrip())
        for k in range(def_line - 1, min(len(ol), def_line + 40)):
            cur = ol[k]
            if k + 1 > def_line and cur.strip() and (
                    len(cur) - len(cur.lstrip())) <= indent:
                break
            body.append(f"{k+1}: {cur}")
        chk["enclosing_def_body"] = body
        sig = ol[def_line - 1].strip()
        chk["enclosing_signature"] = sig
        params = re.findall(r"(\w+)\s*:\s*torch\.Tensor", sig) or re.findall(
            r"\b(\w+)\b(?=\s*[:,)])", sig)
        chk["enclosing_params"] = params
        # which param is the FIRST operand of the failing einsum?
        m = re.search(r'einsum\(\s*["\']nij,njk->nik["\']\s*,\s*([\w\.]+)\s*,\s*([\w\.]+)',
                      ol[target - 1])
        if not m:
            # operand may be on a following line
            window = "\n".join(ol[target - 1: target + 3])
            m = re.search(r'einsum\(\s*["\']nij,njk->nik["\']\s*,\s*([\w\.]+)\s*,\s*([\w\.]+)',
                          window)
        chk["einsum_operands"] = [m.group(1), m.group(2)] if m else None
        chk["einsum_window"] = [f"{target + k}: {ol[target - 1 + k]}"
                                for k in range(0, min(4, len(ol) - target + 1))]

# ============================================================ 2. where does operand 2 come from
# In OPINE the second operand is usually a caller-supplied field. Find the PUBLIC
# methods whose parameter names match, then find their call sites in the runner.
methods = []
for n in ast.walk(ast.parse(OPINE.read_text(encoding="utf-8"))):
    if isinstance(n, ast.FunctionDef):
        try:
            src = ast.unparse(n)
        except Exception:
            continue
        if 'einsum("nij,njk->nik"' in src or "nij,njk->nik" in src:
            methods.append({"method": n.name, "line": n.lineno})
chk["ethods_containing_einsum"] = methods

rl = lines_of(RUNNER)
calls = []
for m in methods:
    for i, l in enumerate(rl, 1):
        if m["method"] in l and "_opine" in l:
            ctx = [f"{k}: {rl[k-1]}" for k in range(max(1, i - 2),
                                                     min(len(rl), i + 4) + 1)]
            calls.append({"method": m["method"], "call_line": i, "context": ctx})
chk["runner_call_sites"] = calls

# ============================================================ 3. other 8192 producers
producers = []
for p in sorted(WT.rglob("*.py")):
    if "_archive" in p.parts or ".git" in p.parts:
        continue
    if p.name in {x.name for x in WT.glob("experiments/*")}:
        pass
    txt = p.read_text(encoding="utf-8", errors="ignore")
    if re.search(r"(?<![_\w])8192(?![_\w])", txt):
        for i, l in enumerate(txt.splitlines(), 1):
            s = l.strip()
            if re.search(r"(?<![_\w])8192(?![_\w])", s) and not s.startswith("#"):
                if re.search(r"num_channels\s*=\s*8192|= 8192\b|, 8192\b|8192,", s):
                    producers.append({"file": p.name, "line": i, "text": s[:120]})
chk["literal_8192_producers"] = producers
chk["n_8192_producers"] = len(producers)

# ============================================================ 4. default path preserved
rtext = RUNNER.read_text(encoding="utf-8")
chk["flag_name"] = "HENRI_MACRO_NUM_CHANNELS"
chk["flag_present"] = "HENRI_MACRO_NUM_CHANNELS" in rtext
chk["default_arm_present"] = bool(re.search(r"else\s+8192\)", rtext))
chk["scale_arm_present"] = bool(re.search(
    r'_num_channels\s*=\s*\(int\(SCALE\["num_blocks"\]\)', rtext))
chk["default_is_8192"] = ("HENRI_MACRO_NUM_CHANNELS\", \"0\") == \"1\"" in rtext)
if not (chk["flag_present"] and chk["default_arm_present"] and chk["scale_arm_present"]):
    out["errors"].append("the flag-gated form is incomplete: default or SCALE arm missing")
if not chk["default_is_8192"]:
    out["errors"].append("the flag does not default to OFF; the default path may have "
                         "changed")

# ============================================================ verdict
out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "second_resolution_mismatch.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("SECOND_8192=" + out["verdict"] + " errors=" + str(len(out["errors"])))
print("einsum_line=" + str(chk.get("second_einsum_line"))
      + " enclosing_def=" + str(chk.get("enclosing_def_line")))
print("signature=" + str(chk.get("enclosing_signature")))
print("einsum_operands=" + str(chk.get("einsum_operands")))
print("methods_with_einsum=" + json.dumps(chk.get("ethods_containing_einsum")))
for c in chk.get("runner_call_sites", [])[:3]:
    print("  call %s @%s" % (c["method"], c["call_line"]))
    for l in c["context"][:5]:
        print("     " + l[:120])
print("n_8192_literal_producers=" + str(chk.get("n_8192_producers")))
for pr in chk.get("literal_8192_producers", [])[:10]:
    print("   %s:%s  %s" % (pr["file"], pr["line"], pr["text"]))
print("flag: present=%s default_8192=%s scale_arm=%s"
      % (chk.get("flag_present"), chk.get("default_is_8192"), chk.get("scale_arm_present")))
if chk.get("enclosing_def_body"):
    print("--- enclosing body ---")
    for l in chk["enclosing_def_body"][:18]:
        print("   " + l[:130])
for e in out["errors"]:
    print("ERR: " + e[:200])
print("RECEIPT=" + str(receipt))
