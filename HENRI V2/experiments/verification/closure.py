"""Transitive import closure of the live spine -- what is ACTUALLY connected?

Answers the user's real question ("100s of disconnected modules") with a measured
reachability set instead of a count. Static AST parse; no imports executed.
"""
import ast
import os
import sys
from collections import deque

V2 = r"C:/Users/chan/henri-worktrees/zone-a-selfplay/HENRI V2"
ENTRY = "experiments/verification/smoke_unified_vla_cuda.py"


def local_mods(path: str) -> set:
    """Module names imported by `path` that resolve to a local .py file."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except Exception:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.level:                      # relative import -> local
                if n.module:
                    out.add(n.module.split(".")[0])
            elif n.module:
                out.add(n.module.split(".")[0])
    return out


def resolve(name: str):
    p = os.path.join(V2, name + ".py")
    if os.path.isfile(p):
        return p
    pkg = os.path.join(V2, name, "__init__.py")
    if os.path.isfile(pkg):
        return pkg
    return None


def main() -> int:
    ep = os.path.join(V2, ENTRY)
    if not os.path.isfile(ep):
        print("ENTRY MISSING:", ep)
        return 1

    seen, order, unresolved = set(), [], set()
    q = deque([(ENTRY, 0)])
    depth = {}
    while q:
        rel, d = q.popleft()
        key = rel.replace("\\", "/")
        if key in seen:
            continue
        seen.add(key)
        depth[key] = d
        order.append(key)
        full = os.path.join(V2, rel)
        for m in sorted(local_mods(full)):
            r = resolve(m)
            if r:
                q.append((os.path.relpath(r, V2), d + 1))
            else:
                unresolved.add(m)

    print("=" * 78)
    print("TRANSITIVE IMPORT CLOSURE OF THE LIVE SPINE")
    print("  entry:", ENTRY)
    print("=" * 78)
    print(f"\nLOCAL MODULES REACHABLE: {len(seen)}")
    for rel in order:
        print(f"   d{depth[rel]}  {rel}")

    third = sorted(m for m in unresolved if m not in
                   {os.path.splitext(os.path.basename(x))[0] for x in seen})
    print(f"\nTHIRD-PARTY / EXTERNAL ({len(third)}): {', '.join(third)}")

    # The spec's four named spine modules: reachable, or not?
    print("\n" + "=" * 78)
    print("SPEC SPINE MODULES -- REACHABLE FROM THE ENTRYPOINT?")
    print("=" * 78)
    for m in ("o_vsa_ingress_tokenizer", "arc_sagnac_veto", "hopfield_cleanup",
              "henri_calibrated_action_head", "efe_planner", "wave_jepa",
              "recursive_dual_edmd", "arc_task_functor", "henri_vision_encoder",
              "darwinian_phase_swarm", "henri_decoder", "henri_action_gate"):
        hits = [k for k in seen if os.path.basename(k)[:-3] == m]
        if hits:
            print(f"  REACHABLE   {m:36s} via {hits[0]}  (depth {depth[hits[0]]})")
        else:
            # is it imported by anything reachable, even if not resolved?
            print(f"  NOT in closure  {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())