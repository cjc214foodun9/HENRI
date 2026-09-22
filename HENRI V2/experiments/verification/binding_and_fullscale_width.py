#!/usr/bin/env python3
"""P0: is `SagnacGateUnavailable` actually BOUND in production_arc_run.py?

WHY THIS IS URGENT
    My isinstance(...) patch to the veto handler SUCCEEDED (ACTION1 receipt:
    A3_isinstance=True, A3_namecmp=False). Its companion module-level IMPORT patch
    FAILED with "Found 2 matches for old_string", because
    `from sagnac_mcts_planner import SagnacMCTSPlanner` appears TWICE (L51, L776).
    If the name is used in the handler and never imported, the handler raises
    NameError the moment the veto raises -- converting a recorded UNAVAILABLE
    status into a crash. Same defect class as the missing `import os` earlier.

METHOD
    AST for the binding question (deterministic, dependency-free): collect every
    alias bound by ANY ImportFrom whose module is `sagnac_mcts_planner`, at module
    scope and nested, then list every bare Name usage. Then a runtime `hasattr`
    confirmation in the py3.12 arc env.

ALSO MEASURED (Action 5 go/no-go)
    The width question at FULL scale. The veto raises at d_model=512 because
    `_psi_macro` is 65536-wide (SU(3) transducer output) while the refs are
    d_model-wide. If `state_wave` and `boundary_batch[0]` grow with d_model, the
    mismatch vanishes at 65536 and the defect is REDUCED-SCALE-ONLY. Measured here
    by instantiating the real tokenizer at num_blocks 64 and 8192.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

WT = Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\semantic-backbone\HENRI V2")
RUNNER = WT / "production_arc_run.py"
PLANNER = WT / "sagnac_mcts_planner.py"

out: dict = {"checks": {}, "errors": []}
chk = out["checks"]

tree = ast.parse(RUNNER.read_text(encoding="utf-8"))

# ---------------------------------------------------------- P1/P2 binding + usage
bound: dict = {"module_level": [], "nested": []}
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom) and node.module == "sagnac_mcts_planner":
        names = [a.name for a in node.names]
        rec = {"line": node.lineno, "names": names}
        if node.col_offset == 0:
            bound["module_level"].append(rec)
        else:
            bound["nested"].append(rec)

uses = [{"line": n.lineno, "col": n.col_offset}
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id == "SagnacGateUnavailable"]

all_bound = set()
for r in bound["module_level"] + bound["nested"]:
    all_bound.update(r["names"])

chk["P1_importfrom_sites"] = bound
chk["P1_names_bound"] = sorted(all_bound)
chk["P1_gate_unavailable_imported"] = "SagnacGateUnavailable" in all_bound
chk["P2_usage_sites"] = uses
chk["P2_usage_count"] = len(uses)

if uses and not chk["P1_gate_unavailable_imported"]:
    out["errors"].append(
        f"P0 BINDING DEFECT: SagnacGateUnavailable used at {[u['line'] for u in uses]} "
        f"but bound by NO import. The handler will raise NameError the moment the "
        f"veto raises. Bound names are {sorted(all_bound)}.")
elif uses:
    chk["P0_binding_ok"] = True
else:
    out["errors"].append("P2: SagnacGateUnavailable is never used in the runner; "
                         "the isinstance branch is dead code")

# ---------------------------------------------------------- P3 runtime confirmation
py312 = Path.home() / "AppData/Local/Temp/arc312_env/Scripts/python.exe"
if py312.exists():
    code = (
        "import sys,warnings;warnings.filterwarnings('ignore');sys.path.insert(0,r'%s');"
        "import production_arc_run as p;"
        "print('RUNTIME_BOUND=' + str(hasattr(p,'SagnacGateUnavailable')))"
    ) % str(WT)
    cp = subprocess.run([str(py312), "-c", code], capture_output=True, text=True,
                        timeout=600, cwd=str(WT))
    line = [l for l in cp.stdout.splitlines() if l.startswith("RUNTIME_BOUND=")]
    chk["P3_runtime_bound"] = (line[-1].split("=")[1] == "True") if line else None
    if chk["P3_runtime_bound"] is False:
        out["errors"].append("P3: runtime confirms SagnacGateUnavailable is NOT bound")
    if not line:
        chk["P3_stderr_tail"] = cp.stderr[-300:]
else:
    chk["P3_runtime_bound"] = "env-absent"

# ---------------------------------------------------------- P4 full-scale widths
try:
    sys.path.insert(0, str(WT))
    import numpy as np
    from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer

    grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    widths = {}
    for nb in (64, 8192):
        tk = O_VSA_IngressTokenizer(num_blocks=nb, vocab_size=256)
        w = tk.encode_spatial_grid(grid)
        widths[f"num_blocks_{nb}"] = {
            "state_wave_numel": int(w.numel()),
            "shape": list(w.shape),
            "dtype": str(w.dtype),
        }
    chk["P4_state_wave_widths"] = widths
    sm = widths["num_blocks_64"]["state_wave_numel"]
    lg = widths["num_blocks_8192"]["state_wave_numel"]
    chk["P4_follows_num_blocks"] = (sm != lg)
    # The transducer output width does not depend on num_blocks (it is fixed at
    # 65536 per the measured production call: ((65536,), (512,), (512,))).
    chk["P4_predicted_full_scale"] = (
        "MATCH_AT_65536" if lg == 65536 else f"MISMATCH_(state={lg})")
except Exception as e:  # noqa: BLE001
    chk["P4_error"] = f"{type(e).__name__}: {e}"
    out["errors"].append(f"P4 width probe failed: {type(e).__name__}: {e}")

# ---------------------------------------------------------- SU3 transducer location
scan = []
src = (WT / "efe_planner.py").read_text(encoding="utf-8", errors="ignore")
chk["efe_planner_total_lines"] = len(src.splitlines())
for i, l in enumerate(src.splitlines(), 1):
    s = l.strip()
    if s.startswith("class SU3FieldWaveTransducer") or s.startswith("def field_to_wave") \
            or s.startswith("class WaveStateTransducer"):
        scan.append((i, s[:90]))
chk["su3_scan_in_efe_planner"] = scan

for other in ("synthesize_sagnac_guarded_macro_options.py",):
    p = WT / other
    if p.exists():
        s2 = p.read_text(encoding="utf-8", errors="ignore")
        chk[f"{other}_total_lines"] = len(s2.splitlines())
        hits = [(i, l.strip()[:90]) for i, l in enumerate(s2.splitlines(), 1)
                if l.strip().startswith(("class SU3FieldWaveTransducer",
                                         "def field_to_wave"))]
        chk[f"{other}_scan"] = hits

out["verdict"] = "PASS" if not out["errors"] else "FAIL"
receipt = WT / "experiments" / "verification" / "binding_and_fullscale_width.json"
receipt.write_text(json.dumps(out, indent=1), encoding="utf-8")

print("P0=" + out["verdict"] + " errors=" + str(len(out["errors"])))
print("bound_names=" + str(chk.get("P1_names_bound")))
print("gate_imported=" + str(chk.get("P1_gate_unavailable_imported"))
      + " usage_lines=" + str([u["line"] for u in uses]))
print("runtime_bound=" + str(chk.get("P3_runtime_bound")))
print("state_w512=" + str(chk.get("P4_state_wave_widths", {}).get(
    "num_blocks_64", {}).get("state_wave_numel"))
      + " state_w8192=" + str(chk.get("P4_state_wave_widths", {}).get(
          "num_blocks_8192", {}).get("state_wave_numel")))
print("full_scale_pred=" + str(chk.get("P4_predicted_full_scale")))
print("efe_lines=" + str(chk.get("efe_planner_total_lines")) + " su3_scan="
      + str(chk.get("su3_scan_in_efe_planner")))
for e in out["errors"]:
    print("ERR: " + e[:260])
print("RECEIPT=" + str(receipt))
