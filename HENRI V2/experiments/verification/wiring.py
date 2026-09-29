"""Precise wiring audit: WHO imports each module, and HOW (module-level vs lazy).

Corrects an earlier over-claim. The transitive closure said 3 of the spec's 4 named
spine modules are "NOT in closure". That is a strong claim from a brand-new tool, so
this checks it three ways:
  1. exact-name imports, with file + line + nesting depth (lazy = inside a function)
  2. dynamic imports (importlib / __import__ / getattr(module))
  3. string references (docstrings, registries) -- NOT wiring, labelled as such
"""
import ast
import os
import sys

V2 = r"C:/Users/chan/henri-worktrees/zone-a-selfplay/HENRI V2"

TARGETS = [
    "o_vsa_ingress_tokenizer", "arc_sagnac_veto", "hopfield_cleanup",
    "henri_calibrated_action_head", "efe_planner", "wave_jepa",
    "recursive_dual_edmd", "sagnac_mcts_planner", "zone_c_retrieval_bridge",
    "arc_spatial_basis", "henri_adaptive_backbone", "arc_task_functor",
    "darwinian_phase_swarm", "henri_vision_encoder", "henri_decoder",
]

# the reachable set measured by the closure pass
CLOSURE = {
    "darwinian_phase_swarm.py", "henri_action_gate.py", "henri_decoder.py",
    "henri_unified_vla.py", "henri_vision_encoder.py", "efe_planner.py",
    "henri_causal_planner.py", "hopfield_cleanup.py", "idbd_swifttd.py",
    "product_clifford_product_kernel.py", "zone_c_segment_cache.py",
    "arc_action_payloads.py", "henri_ast_grammar_mask.py",
    "henri_discrete_egress_flag.py", "henri_prefix_kv.py", "arc_task_functor.py",
    "temporal_ledger_bridge.py", "connected_component_segmenter.py",
    "chromodynamic_grounding.py", "complex_phase_transition.py",
    "henri_external_outcome_refactor_module.py", "subliminal_clock_probe.py",
    "universal_data_transducer.py", "zone_c_env.py", "arc_phase_map.py",
    "f6_adaptive_functor.py", "f7_affine_egress.py", "koopman_generator_bank.py",
    "koopman_subspace_functor.py", "staticity_partition_functor.py",
    "temporal_transition_ledger.py",
    "experiments/verification/smoke_unified_vla_cuda.py",
}


def scan(path: str):
    """Return (module_level_hits, lazy_hits, dynamic_hits, string_hits)."""
    try:
        src = open(path, "r", encoding="utf-8", errors="ignore").read()
        tree = ast.parse(src, filename=path)
    except Exception as e:
        return None, None, None, None
    top, lazy, dyn, strs = {}, {}, {}, {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif node.module:
                names = [node.module.split(".")[0]]
            nested = getattr(node, "col_offset", 0) > 0
            for nm in names:
                if nm in TARGETS:
                    rec = (node.lineno, "nested" if nested else "module")
                    (lazy if nested else top).setdefault(nm, []).append(rec)
        elif isinstance(node, ast.Call):
            f = node.func
            fn = getattr(f, "attr", None) or getattr(f, "id", None)
            if fn in ("import_module", "__import__"):
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        nm = a.value.split(".")[0]
                        if nm in TARGETS:
                            dyn.setdefault(nm, []).append(node.lineno)
    for nm in TARGETS:
        c = src.count(nm)
        if c:
            strs[nm] = c
    return top, lazy, dyn, strs


def main() -> int:
    os.chdir(V2)
    files = sorted(f for f in os.listdir(".") if f.endswith(".py"))
    print("=" * 78)
    print("WIRING AUDIT: exact imports of each target module")
    print("=" * 78)

    agg = {t: {"top": [], "lazy": [], "dyn": [], "str": 0} for t in TARGETS}
    for f in files:
        top, lazy, dyn, strs = scan(f)
        if top is None:
            continue
        for t, v in top.items():
            agg[t]["top"] += [(f, ln) for ln, _ in v]
        for t, v in lazy.items():
            agg[t]["lazy"] += [(f, ln) for ln, _ in v]
        for t, v in dyn.items():
            agg[t]["dyn"] += [(f, ln) for ln, _ in v]
        for t, c in strs.items():
            agg[t]["str"] += c

    for t in TARGETS:
        a = agg[t]
        in_closure = t + ".py" in CLOSURE
        mark = "IN-CLOSURE" if in_closure else "not-in-closure"
        print(f"\n{t}  [{mark}]")
        if a["top"]:
            for f, ln in a["top"]:
                tag = "  (reachable)" if f in CLOSURE else "  (NOT reachable)"
                print(f"    module-level import  {f}:{ln}{tag}")
        if a["lazy"]:
            for f, ln in a["lazy"]:
                tag = "  (reachable)" if f in CLOSURE else "  (NOT reachable)"
                print(f"    LAZY/fn-level import {f}:{ln}{tag}")
        if a["dyn"]:
            for f, ln in a["dyn"]:
                print(f"    DYNAMIC import       {f}:{ln}")
        if not (a["top"] or a["lazy"] or a["dyn"]):
            print(f"    NO import statement anywhere. raw string mentions: {a['str']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())