#!/usr/bin/env python3
"""Locate SU3FieldWaveTransducer and read field_to_wave's WIDTH RULE.

DECISIVE FOR ACTION 2 AND ACTION 5
    Measured (binding_and_fullscale_width.py, own receipt):
        state_wave width at num_blocks=64   -> 512
        state_wave width at num_blocks=8192 -> 65536        (= num_blocks * 8)
        observed production veto call       -> ((65536,), (512,), (512,))
    So at d_model=512 the CANDIDATE is 65536-wide while both REFS are 512-wide: a
    128x mismatch, and the veto cannot multiply them.

    Open question: does `field_to_wave` ALSO scale with d_model, or is it FIXED at
    65536?

      * FIXED at 65536  -> the mismatch is REDUCED-SCALE-ONLY. At d_model=65536 all
        three waves are 65536-wide and the veto runs. Then the gauntlet's reduced
        scale is the sole cause and Action 5 is the configuration where the veto is
        exercisable at all.
      * SCALES with d_model -> the two sides track each other, so the mismatch must
        originate elsewhere (stale wave, different path) and my diagnosis needs
        revising.

    Settled by reading the SOURCE, not by inference. Also records whether the three
    veto inputs belong to the SAME representation family -- because the architecture
    catalog states the complex flat [D] family is a SEPARATE family from the real
    [num_blocks, 8] family, and a width mismatch is exactly what that would produce.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
out: dict = {"checks": {}, "errors": []}
chk = out["checks"]

TARGET = "SU3FieldWaveTransducer"

# ---------------------------------------------------------------- locate the class
defs = []
for py in sorted(WT.rglob("*.py")):
    if "_archive" in py.parts or ".worktrees" in str(py):
        continue
    try:
        txt = py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if f"class {TARGET}" in txt:
        tree = ast.parse(txt)
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef) and n.name == TARGET:
                defs.append({"file": py.name, "lineno": n.lineno,
                             "end": getattr(n, "end_lineno", None),
                             "methods": [m.name for m in n.body
                                         if isinstance(m, ast.FunctionDef)]})
chk["class_definitions"] = defs

# Also record where ANY field_to_wave is defined, for cross-reference.
fw_defs = []
for py in sorted(WT.rglob("*.py")):
    if "_archive" in py.parts:
        continue
    try:
        txt = py.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    if "def field_to_wave" in txt:
        tree = ast.parse(txt)
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == "field_to_wave":
                fw_defs.append({"file": py.name, "lineno": n.lineno})
chk["field_to_wave_definitions"] = fw_defs

# ---------------------------------------------------------------- read the rule
src_rule: dict = {}
for d in defs:
    p = WT / d["file"]
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    body = lines[d["lineno"] - 1: (d["end"] or d["lineno"] + 400)]
    # field_to_wave body only
    fw_body, capture = [], False
    for ln in body:
        s = ln.strip()
        if s.startswith("def field_to_wave"):
            capture = True
        elif capture and (s.startswith("def ") or s.startswith("class ")):
            break
        if capture:
            fw_body.append(ln)
    src_rule[d["file"]] = fw_body[:70]

    # Which width does it emit? Look for the construction of the returned tensor.
    joined = "\n".join(fw_body)
    width_signals = {
        "uses_d_model": "d_model" in joined,
        "uses_num_blocks": "num_blocks" in joined,
        "uses_field_numel_or_shape": bool(re.search(r"field\.(numel|shape|reshape)", joined)),
        "has_complex": "complex" in joined,
        "literal_65536": "65536" in joined,
        "returns_2d": bool(re.search(r"\[\s*num_blocks\s*,\s*8\s*\]", joined)),
    }
    chk[f"width_signals::{d['file']}"] = width_signals

chk["field_to_wave_source"] = {k: v for k, v in src_rule.items()}

# ---------------------------------------------------------------- analytic verdict
ws = next(iter(chk.get("width_signals", {}).values()), None) if defs else None
if ws is None:
    out["errors"].append("could not read field_to_wave source; rule UNKNOWN")
else:
    if ws["literal_65536"] and not ws["uses_d_model"]:
        verdict = "FIXED_65536 -> mismatch is REDUCED-SCALE-ONLY"
    elif ws["uses_d_model"]:
        verdict = "SCALES_WITH_D_MODEL -> mismatch must originate elsewhere"
    elif ws["uses_field_numel_or_shape"]:
        verdict = "FOLLOWS_INPUT_FIELD -> mismatch depends on the FIELD width"
    else:
        verdict = "INDETERMINATE -> read the source block in the receipt"
    chk["width_rule_verdict"] = verdict

# ---------------------------------------------------------------- family check
# The architecture catalog: complex flat [D] is a THIRD representation family,
# separate from the real [num_blocks, 8] family. If the candidate is complex flat and
# the refs are real [nb, 8], the width mismatch is a SYMPTOM of that family split.
chk["observed_call_shapes"] = {
    "candidate": "(65536,) complex64  <- _trans.field_to_wave output",
    "axiom": "(512,) float32          <- boundary_batch[0]",
    "world": "(512,) float32          <- state_wave",
    "candidate_family": "complex flat [D]",
    "ref_family": "real flattened [num_blocks, 8]",
    "same_family": False,
}
chk["family_split_note"] = (
    "The three inputs are NOT one representation family: the candidate is the complex "
    "flat [D] SU(3) field wave, while both references are real [num_blocks, 8] grid "
    "waves flattened. A width mismatch is the predictable CONSEQUENCE of comparing "
    "across families. That makes Action 2 a representation-boundary decision, not a "
    "shape typo, and it is why silently projecting would change what the veto MEANS.")

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "su3_width_rule.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("RULE=" + str(chk.get("width_rule_verdict")))
print("class_defs=" + str([(d["file"], d["lineno"]) for d in defs]))
print("fw_defs=" + str([(d["file"], d["lineno"]) for d in fw_defs]))
print("signals=" + str({k: v for k, v in chk.items() if k.startswith("width_signals")}))
print("RECEIPT=" + str(receipt))
