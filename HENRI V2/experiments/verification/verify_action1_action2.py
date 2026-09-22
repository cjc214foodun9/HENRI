#!/usr/bin/env python3
"""ACTION 1 completion + ACTION 2 verification. Corrects two of MY OWN defects.

DEFECT 1 (my scan): the class-locator loop skipped any path containing
`.worktrees`, which is EVERY file under the worktree root, so it reported
`class_defs=[]` and proved nothing. The filter is corrected: skip only `_archive`
and `.git`.

DEFECT 2 (my gate): my ACTION1 gate demanded `line > except_line` for opine_info
and produced a FALSE FAILURE on a definition that follows the handler and covers
both arms. Corrected criterion: the ONLY unsafe shape is an assignment INSIDE an
except handler, which restores the NameError risk. That is what is tested now.

GATES
    V1  binding: SagnacGateUnavailable IS now bound in production_arc_run.py
    V2  `os` is bound in BOTH files that call os.environ in the veto path
    V3  opine_info scope by indentation -> not inside any except handler
    V4  SU(3) transducer located, and field_to_wave's WIDTH RULE read from source
    V5  bridge OFF (default) -> mismatched widths still raise the named type
    V6  bridge ON -> computes, returns a 3-tuple, and records the bridge
    V7  bridge OFF leaves the function's contract unchanged (3-tuple, no bridge)
    V8  full-scale width arithmetic: at num_blocks=8192 do the three widths match?
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
PLANNER = WT / "sagnac_mcts_planner.py"
RUNNER = WT / "production_arc_run.py"

out: dict = {"checks": {}, "errors": []}
chk = out["checks"]


def py_files():
    for p in sorted(WT.rglob("*.py")):
        # CORRECTED FILTER: only _archive and .git are excluded. The previous version
        # also excluded anything containing ".worktrees", which is the ROOT ITSELF.
        if "_archive" in p.parts or ".git" in p.parts:
            continue
        yield p


# ------------------------------------------------------------------ V1 / V2
rt = ast.parse(RUNNER.read_text(encoding="utf-8"))
bound = set()
for n in ast.walk(rt):
    if isinstance(n, ast.ImportFrom) and n.module == "sagnac_mcts_planner":
        bound.update(a.name for a in n.names)
    if isinstance(n, ast.Import):
        bound.update(a.name.split(".")[0] for a in n.names)
    if isinstance(n, ast.ImportFrom) and n.module == "os":
        bound.add("os")
chk["V1_runnder_planner_names"] = sorted(bound)
chk["V1_gate_bound"] = "SagnacGateUnavailable" in bound
chk["V2_os_bound_in_runner"] = "os" in bound or (RUNNER.read_text(
    encoding="utf-8").count("import os") > 0)
if not chk["V1_gate_bound"]:
    out["errors"].append(
        "V1: SagnacGateUnavailable still NOT bound in production_arc_run.py -> the "
        "handler raises NameError the moment the veto raises")
if not chk["V2_os_bound_in_runner"]:
    out["errors"].append("V2: `os` not bound in production_arc_run.py")

psrc = PLANNER.read_text(encoding="utf-8")
chk["V2_os_bound_in_planner"] = bool(re.search(r"^import os\b", psrc, re.M))
if not chk["V2_os_bound_in_planner"]:
    out["errors"].append("V2: `os` not bound in sagnac_mcts_planner.py")

# ------------------------------------------------------------------ V3
lines = RUNNER.read_text(encoding="utf-8").splitlines()


def indent(s: str) -> int:
    return len(s) - len(s.lstrip(" "))


exc = [(i, indent(l)) for i, l in enumerate(lines, 1)
       if l.strip().startswith("except Exception as _veto_exc")]
opine = [(i, indent(l)) for i, l in enumerate(lines, 1)
         if l.strip().startswith("opine_info") and "=" in l
         and l.split("=")[0].strip() == "opine_info"]
chk["V3_except_marks"] = exc
chk["V3_opine_marks"] = opine


def inside_except(ln_o, ind_o):
    for (ln_e, ind_e) in exc:
        if ln_o <= ln_e or ind_o <= ind_e:
            continue
        dedented = False
        for b in lines[ln_e:ln_o - 1]:
            bs = b.strip()
            if not bs or bs.startswith("#"):
                continue
            if indent(b) <= ind_e:
                dedented = True
                break
        if not dedented:
            return True
    return False


scopes = [{"line": l, "indent": i, "inside_except": inside_except(l, i)}
          for (l, i) in opine]
chk["V3_opine_scope"] = scopes
chk["V3_n_assignments"] = len(scopes)
chk["V3_unsafe_inside_except"] = any(s["inside_except"] for s in scopes)
if not scopes:
    out["errors"].append("V3: no opine_info assignment found")
if chk["V3_unsafe_inside_except"]:
    out["errors"].append(
        f"V3: opine_info assigned INSIDE an except handler -> NameError risk once the "
        f"veto succeeds. scope={scopes}")

# ------------------------------------------------------------------ V4
class_defs, fw_defs = [], []
for p in py_files():
    try:
        txt = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if "class SU3FieldWaveTransducer" in txt:
        for n in ast.walk(ast.parse(txt)):
            if isinstance(n, ast.ClassDef) and n.name == "SU3FieldWaveTransducer":
                class_defs.append({"file": p.name, "lineno": n.lineno,
                                   "end": getattr(n, "end_lineno", None)})
    if "def field_to_wave" in txt:
        for n in ast.walk(ast.parse(txt)):
            if isinstance(n, ast.FunctionDef) and n.name == "field_to_wave":
                fw_defs.append({"file": p.name, "lineno": n.lineno})
chk["V4_class_defs"] = class_defs
chk["V4_fw_defs"] = fw_defs

rule = {}
for cd in class_defs:
    p = WT / cd["file"]
    src_lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    body, cap = [], False
    for ln in src_lines[cd["lineno"] - 1: (cd["end"] or cd["lineno"] + 500)]:
        s = ln.strip()
        if s.startswith("def field_to_wave"):
            cap = True
        elif cap and (s.startswith("def ") or s.startswith("class ")):
            break
        if cap:
            body.append(ln)
    j = "\n".join(body)
    rule[cd["file"]] = {
        "uses_d_model": "d_model" in j,
        "uses_num_blocks": "num_blocks" in j,
        "uses_field_shape": bool(re.search(r"field\.(numel|shape|reshape|size)", j)),
        "literal_65536": "65536" in j,
        "emits_2d_blocks": bool(re.search(r"\[\s*num_blocks\s*,\s*8\s*\]", j)),
        "source_excerpt": body[:28],
    }
    break
chk["V4_width_rule"] = rule
if not class_defs:
    out["errors"].append("V4: SU3FieldWaveTransducer not located with the corrected scan")
else:
    r = next(iter(rule.values()))
    if r["uses_d_model"] and not r["uses_field_shape"]:
        chk["V4_verdict"] = "FOLLOWS_d_model -> widths track, mismatch from ELSEWHERE"
    elif r["uses_field_shape"] and not r["uses_d_model"]:
        chk["V4_verdict"] = "FOLLOWS_INPUT_FIELD -> width set by the incoming field"
    elif r["literal_65536"] and not r["uses_d_model"]:
        chk["V4_verdict"] = "FIXED_65536 -> reduced-scale-only mismatch"
    else:
        chk["V4_verdict"] = f"INDETERMINATE flags={r}"

# ------------------------------------------------------------------ V5/V6/V7 live
try:
    sys.path.insert(0, str(WT))
    import torch
    import sagnac_mcts_planner as sp

    p = sp.SagnacMCTSPlanner(d_model=512, k_blocks=64, tau_veto=0.35, device="cpu")
    big = torch.randn(65536, generator=torch.Generator().manual_seed(0)).to(torch.complex64)
    small = torch.randn(512, generator=torch.Generator().manual_seed(1))

    # V5: bridge OFF -> named raise
    os.environ.pop("HENRI_SAGNAC_WIDTH_BRIDGE", None)
    try:
        p.dual_channel_sagnac_veto(big, small, small, epsilon_hard=p.tau_veto)
        chk["V5_raises_when_off"] = False
        out["errors"].append("V5: bridge OFF did not raise")
    except sp.SagnacGateUnavailable:
        chk["V5_raises_when_off"] = True
    except Exception as e:  # noqa: BLE001
        chk["V5_raises_when_off"] = False
        out["errors"].append(f"V5: raised {type(e).__name__}, not the named type")

    # V6: bridge ON -> computes, 3-tuple, records bridge
    os.environ["HENRI_SAGNAC_WIDTH_BRIDGE"] = "1"
    try:
        res = p.dual_channel_sagnac_veto(big, small, small, epsilon_hard=p.tau_veto)
        chk["V6_tuple_len"] = len(res) if isinstance(res, tuple) else None
        chk["V6_delta_axiom"] = round(float(res[0]), 6)
        chk["V6_hard"] = bool(res[2])
        chk["V6_bridge_recorded"] = getattr(p, "last_sagnac_bridge", None)
        if chk["V6_tuple_len"] != 3:
            out["errors"].append(f"V6: bridge ON returned {chk['V6_tuple_len']} values, "
                                 f"expected 3")
        if not chk["V6_bridge_recorded"]:
            out["errors"].append("V6: bridge applied but not recorded on the instance")
    except Exception as e:  # noqa: BLE001
        chk["V6_error"] = f"{type(e).__name__}: {e}"
        out["errors"].append(f"V6: bridge ON raised {type(e).__name__}: {e}")
    finally:
        os.environ.pop("HENRI_SAGNAC_WIDTH_BRIDGE", None)

    # V7: bridge OFF, matched widths -> unchanged 3-tuple, no bridge
    m = torch.zeros(512)
    m[0] = 1.0
    r = p.dual_channel_sagnac_veto(m, m, m, epsilon_hard=p.tau_veto)
    chk["V7_tuple_len"] = len(r) if isinstance(r, tuple) else None
    chk["V7_bridge_none"] = getattr(p, "last_sagnac_bridge", "MISSING") is None
    if chk["V7_tuple_len"] != 3:
        out["errors"].append("V7: matched-width contract changed")
    if not chk["V7_bridge_none"]:
        out["errors"].append("V7: bridge recorded on a matched-width call")

    # V8: does the bridged comparison DISCRIMINATE? Two independent pools vs a shared
    # pool. If separation is ~0 the bridged gate cannot decide and Action 3 cannot
    # prove anything through the bridge -- reported, not assumed.
    g = torch.Generator().manual_seed(11)
    a = torch.randn(65536, generator=g).to(torch.complex64)
    b = torch.randn(65536, generator=g).to(torch.complex64)
    ref = torch.randn(512, generator=g)
    os.environ["HENRI_SAGNAC_WIDTH_BRIDGE"] = "1"
    d_same = p.dual_channel_sagnac_veto(a, ref, ref, epsilon_hard=p.tau_veto)[0]
    d_othr = p.dual_channel_sagnac_veto(b, ref, ref, epsilon_hard=p.tau_veto)[0]
    os.environ.pop("HENRI_SAGNAC_WIDTH_BRIDGE", None)
    chk["V8_bridged_delta_a"] = round(float(d_same), 6)
    chk["V8_bridged_delta_b"] = round(float(d_othr), 6)
    chk["V8_bridged_separation"] = round(abs(float(d_same) - float(d_othr)), 6)

    # V9: at TRUE matched width with a real discriminator, both hard values reachable
    # (already measured; re-confirmed here so the receipt is self-contained)
    unit = torch.zeros(512); unit[0] = 1.0
    rp_ = p.dual_channel_sagnac_veto(unit, unit, unit, epsilon_hard=p.tau_veto)
    rv_ = p.dual_channel_sagnac_veto(-unit, unit, unit, epsilon_hard=p.tau_veto)
    chk["V9_pass_hard"] = bool(rp_[2])
    chk["V9_veto_hard"] = bool(rv_[2])
    chk["V9_both_reachable"] = (not bool(rp_[2])) and bool(rv_[2])
    if not chk["V9_both_reachable"]:
        out["errors"].append("V9: hard_vetoed not both-reachable at matched width")
except Exception as e:  # noqa: BLE001
    out["errors"].append(f"live fatal: {type(e).__name__}: {e}")
    out["traceback_tail"] = traceback.format_exc()[-600:]

# ------------------------------------------------------------------ runtime bound
py312 = Path.home() / "AppData/Local/Temp/arc312_env/Scripts/python.exe"
if py312.exists():
    code = ("import sys,warnings;warnings.filterwarnings('ignore');"
            "sys.path.insert(0,r'%s');import production_arc_run as p;"
            "print('RBOUND=' + str(hasattr(p,'SagnacGateUnavailable')))" % str(WT))
    cp = subprocess.run([str(py312), "-c", code], capture_output=True, text=True,
                        timeout=600, cwd=str(WT))
    ln = [l for l in cp.stdout.splitlines() if l.startswith("RBOUND=")]
    chk["V1_runtime_bound"] = (ln[-1].split("=")[1] == "True") if ln else None
    if chk["V1_runtime_bound"] is False:
        out["errors"].append("V1: runtime import confirms SagnacGateUnavailable unbound")
    if not ln:
        chk["V1_stderr_tail"] = cp.stderr[-300:]

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "action1_action2_verify.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("VERDICT=" + out["verdict"] + " errors=" + str(len(out["errors"])))
print("V1_gate_bound=" + str(chk.get("V1_gate_bound"))
      + " runtime=" + str(chk.get("V1_runtime_bound"))
      + " os_planner=" + str(chk.get("V2_os_bound_in_planner")))
print("V3_marks=" + str(chk.get("V3_opine_marks"))
      + " unsafe=" + str(chk.get("V3_unsafe_inside_except")))
print("V4_class=" + str([(d["file"], d["lineno"]) for d in class_defs])
      + " verdict=" + str(chk.get("V4_verdict")))
print("V5_off_raises=" + str(chk.get("V5_raises_when_off"))
      + " V6_on_tuple=" + str(chk.get("V6_tuple_len"))
      + " V6_bridge=" + str(chk.get("V6_bridge_recorded")))
print("V7_tuple=" + str(chk.get("V7_tuple_len"))
      + " V9_both=" + str(chk.get("V9_both_reachable")))
print("V8_sep=" + str(chk.get("V8_bridged_separation")))
for e in out["errors"]:
    print("ERR: " + e[:240])
print("RECEIPT=" + str(receipt))
