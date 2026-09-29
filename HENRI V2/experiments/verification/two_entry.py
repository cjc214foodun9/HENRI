"""Which entrypoint does the spec's spine actually belong to?

The spec names `smoke_unified_vla_cuda.py` but describes a spine
(o_vsa -> arc_sagnac_veto -> hopfield -> calibrated_action_head) that the closure
pass says is NOT reachable from it. wiring.py shows those modules ARE imported by
`production_arc_run.py`. Test the two-entrypoint hypothesis directly.

Also classifies the 19 orphan modules: dead experiments vs real capability.
"""
import ast
import os
import sys
from collections import deque

V2 = r"C:/Users/chan/henri-worktrees/zone-a-selfplay/HENRI V2"


def local_imports(path):
    try:
        t = ast.parse(open(path, "r", encoding="utf-8", errors="ignore").read())
    except Exception:
        return set()
    out = set()
    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom) and n.module:
            out.add(n.module.split(".")[0])
    return out


def resolve(name):
    for cand in (name + ".py", os.path.join(name, "__init__.py")):
        if os.path.isfile(os.path.join(V2, cand)):
            return cand
    return None


def closure(entry):
    seen, q = set(), deque([entry])
    while q:
        rel = q.popleft()
        if rel in seen or not os.path.isfile(os.path.join(V2, rel)):
            continue
        seen.add(rel)
        for m in local_imports(os.path.join(V2, rel)):
            r = resolve(m)
            if r:
                q.append(r)
    return seen


def main():
    os.chdir(V2)
    print("=" * 78)
    print("TWO-ENTRYPOINT TEST")
    print("=" * 78)
    ents = ["production_arc_run.py", "experiments/verification/smoke_unified_vla_cuda.py",
            "unified_henri_vla_engine.py", "henri_vla_engine.py"]
    clos = {}
    for e in ents:
        p = os.path.join(V2, e)
        if not os.path.isfile(p):
            print(f"  {e:52s} MISSING")
            continue
        c = closure(e)
        clos[e] = c
        print(f"  {e:52s} {len(c):4d} modules  {os.path.getsize(p):7d} b")

    print("\n" + "=" * 78)
    print("SPEC SPINE MODULES: which entrypoint reaches them?")
    print("=" * 78)
    spine = ["o_vsa_ingress_tokenizer", "arc_sagnac_veto", "hopfield_cleanup",
             "henri_calibrated_action_head", "efe_planner", "sagnac_mcts_planner",
             "zone_c_retrieval_bridge", "wave_jepa", "recursive_dual_edmd",
             "arc_task_functor", "arc_spatial_basis"]
    hdr = "  " + f"{'module':34s}" + "".join(f"{e.split('/')[-1][:14]:16s}" for e in clos)
    print(hdr)
    for m in spine:
        row = f"  {m:34s}"
        for e in clos:
            row += f"{('YES' if m + '.py' in clos[e] else '-'):16s}"
        print(row)

    print("\n" + "=" * 78)
    print("THE ORPHANS: which modules nothing imports")
    print("=" * 78)
    allp = sorted(f for f in os.listdir(".") if f.endswith(".py"))
    imported = set()
    for f in allp:
        imported |= local_imports(os.path.join(f))
    reached_any = set()
    for e in clos:
        reached_any |= clos[e]
    orphans = []
    for f in allp:
        m = f[:-3]
        if m == "__init__":
            continue
        if m in imported:
            continue
        orphans.append((f, os.path.getsize(f), f in reached_any))
    print(f"  total .py: {len(allp)}   orphan (imported by nobody): {len(orphans)}")
    for f, sz, reach in orphans:
        tag = "REACHED" if reach else "DEAD   "
        print(f"    {tag}  {sz:8d} b  {f}")

    # what do orphans mention -> are they capability or experiment?
    print("\n" + "=" * 78)
    print("ORPHAN CONTENT CLASSIFICATION")
    print("=" * 78)
    kw = ("benchmark", "runner", "scorecard", "dashboard", "harness", "probe",
          "gate", "engine", "backbone", "database", "seeding", "ingestion",
          "bridge", "alignment", "inference", "graph")
    for f, sz, reach in orphans:
        hits = [k for k in kw if k in f]
        cls = ",".join(hits) if hits else "other"
        print(f"    {f:56s} {cls}")
    return 0


if __name__ == "__main__":
    sys.exit(main())