#!/usr/bin/env python3
"""henri_graph_render.py — deterministic holonic DAG audit + render.

Closes the BLOCKED computer-vision dimension "holonic state-graph debugging".
Two layers, deliberately separated:

  DETERMINISTIC (authoritative): parse a declarative edge list, then detect
    unreachable nodes, missing nodes, DAG cycles, and duplicate edges. This is
    the answer. A picture is not evidence.

  VISUAL (filter only): render the same graph to PNG so vision_analyze can spot
    layout-level anomalies in one pass. Vision never overrides the analysis.

Input: JSON {"nodes": [{"id","holon","kind"}], "edges": [{"from","to","kind"}]}
Nodes referenced by an edge but absent from "nodes" are reported, not invented.

CLI:
  python henri_graph_render.py --default --out reports/figures/05_holon_dag.png
  python henri_graph_render.py --graph path.json --out x.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

H = Path(r"C:\Users\chan\AppData\Local\hermes")


def default_graph() -> dict:
    """The HENRI holonic development loop, as declared in the bundle skills.
    Declarative only — this encodes the published workflow, not inferred code."""
    nodes = [
        {"id": "cron_collector", "holon": "deterministic", "kind": "collector"},
        {"id": "artifact", "holon": "deterministic", "kind": "artifact"},
        {"id": "telemetry_report", "holon": "deterministic", "kind": "report"},
        {"id": "coe_gate", "holon": "audit", "kind": "gate"},
        {"id": "ledger", "holon": "audit", "kind": "ledger"},
        {"id": "approval", "holon": "human", "kind": "gate"},
        {"id": "implement", "holon": "agent", "kind": "worker"},
        {"id": "ci", "holon": "remote", "kind": "verify"},
        {"id": "decision", "holon": "human", "kind": "gate"},
        {"id": "research", "holon": "research", "kind": "worker"},
        {"id": "vault", "holon": "research", "kind": "store"},
    ]
    edges = [
        {"from": "cron_collector", "to": "artifact", "kind": "data"},
        {"from": "artifact", "to": "telemetry_report", "kind": "data"},
        {"from": "telemetry_report", "to": "coe_gate", "kind": "evidence"},
        {"from": "coe_gate", "to": "ledger", "kind": "seal"},
        {"from": "ledger", "to": "approval", "kind": "govern"},
        {"from": "approval", "to": "implement", "kind": "authorize"},
        {"from": "implement", "to": "ci", "kind": "dispatch"},
        {"from": "ci", "to": "decision", "kind": "outcome"},
        {"from": "decision", "to": "research", "kind": "replan"},
        {"from": "research", "to": "vault", "kind": "write"},
        {"from": "vault", "to": "approval", "kind": "propose"},
    ]
    # The loop is intentional: research feeds implementation, and outcomes feed
    # the next research pass. Declaring it prevents a false FAIL verdict.
    return {"nodes": nodes, "edges": edges, "cycle_is_intended": True,
            "cycle_rationale": "Iterative development loop: decision -> research "
                               "-> vault -> approval is the intended re-plan path."}


def analyze(g: dict) -> dict:
    nodes = [n["id"] for n in g.get("nodes", []) if "id" in n]
    nset = set(nodes)
    dupes = sorted({n for n in nodes if nodes.count(n) > 1})
    edges = [(e.get("from"), e.get("to")) for e in g.get("edges", [])]

    missing = sorted({x for e in edges for x in e if x not in nset})
    self_loop = sorted({a for a, b in edges if a == b})
    dup_edges = sorted({e for e in edges if edges.count(e) > 1})

    adj: dict[str, list[str]] = {n: [] for n in nset}
    indeg = {n: 0 for n in nset}
    for a, b in edges:
        if a in nset and b in nset:
            adj[a].append(b)
            indeg[b] += 1

    # Kahn topological sort -> cycle detection is deterministic
    indeg2 = dict(indeg)
    q = sorted([n for n, d in indeg2.items() if d == 0])
    order = []
    while q:
        n = q.pop(0)
        order.append(n)
        for m in adj[n]:
            indeg2[m] -= 1
            if indeg2[m] == 0:
                q.append(m)
                q.sort()

    cyclic = sorted(n for n in nset if n not in set(order))
    reachable = set(q0 := [n for n in nset if indeg.get(n, 0) == 0])
    frontier = list(reachable)
    while frontier:
        n = frontier.pop()
        for m in adj.get(n, []):
            if m not in reachable:
                reachable.add(m)
                frontier.append(m)
    orphans = sorted(nset - reachable)
    sinks = sorted(n for n in nset if not adj.get(n))

    # A development LOOP is iterative by design. A cycle is only a defect when
    # the graph does not declare it. Conflating the two produces a false alarm.
    intended = bool(g.get("cycle_is_intended"))
    cycle_status = ("INTENDED" if (cyclic and intended) else
                    "UNEXPECTED" if cyclic else "NONE")

    hard_fail = bool(missing or orphans or dupes or self_loop
                     or (cyclic and not intended))
    return {"n_nodes": len(nset), "n_edges": len(edges),
            "duplicate_node_ids": dupes, "missing_nodes": missing,
            "self_loops": self_loop, "duplicate_edges": dup_edges,
            "is_dag": not cyclic, "cycle_nodes": cyclic,
            "cycle_status": cycle_status,
            "topological_order": order, "orphans": orphans, "sinks": sinks,
            "verdict": "FAIL" if hard_fail else "PASS"}


def render(g: dict, res: dict, out: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("RENDER BLOCKED: matplotlib absent "
              "(use viz-venv: %LOCALAPPDATA%\\hermes\\viz-venv)")
        return False

    out.parent.mkdir(parents=True, exist_ok=True)
    # layered layout from the topological order
    layers: dict[str, int] = {}
    for i, n in enumerate(res["topological_order"]):
        layers[n] = i
    for n in res["cycle_nodes"] or []:
        layers.setdefault(n, len(layers))
    cols: dict[int, list[str]] = {}
    for n, L in layers.items():
        cols.setdefault(L, []).append(n)

    pos = {}
    for L, ns in cols.items():
        for k, n in enumerate(sorted(ns)):
            pos[n] = (L, k)

    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=110)
    cmap = {"deterministic": "#2b6cb0", "audit": "#b7791f", "human": "#276749",
            "agent": "#6b46c1", "remote": "#c53030", "research": "#2c7a7b"}
    kindmap = {n["id"]: n.get("holon", "agent") for n in g["nodes"]}
    for a, b in [(e["from"], e["to"]) for e in g["edges"]]:
        if a in pos and b in pos:
            ax.annotate("", xy=pos[b], xytext=pos[a],
                        arrowprops=dict(arrowstyle="->", lw=1.4, color="#4a5568",
                                        shrinkA=12, shrinkB=12))
    for n, (x, y) in pos.items():
        bad = n in res["cycle_nodes"] or n in res["orphans"]
        ax.scatter([x], [y], s=1500,
                   c="#e53e3e" if bad else cmap.get(kindmap.get(n, "agent"), "#718096"),
                   edgecolors="black", linewidths=1.2, zorder=3)
        ax.text(x, y, n, ha="center", va="center", fontsize=7.2,
                color="white", zorder=4, wrap=True, weight="bold")
    ax.set_title(f"HENRI holonic development DAG — {res['verdict']} "
                 f"({res['n_nodes']} nodes, {res['n_edges']} edges, "
                 f"dag={res['is_dag']})", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"rendered {out}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--default", action="store_true")
    ap.add_argument("--graph")
    ap.add_argument("--out", default=str(H / "reports" / "figures" / "05_holon_dag.png"))
    ap.add_argument("--json")
    a = ap.parse_args()

    if a.graph:
        g = json.loads(Path(a.graph).read_text(encoding="utf-8"))
    elif a.default:
        g = default_graph()
    else:
        ap.error("pass --default or --graph <file>")

    res = analyze(g)
    print(json.dumps(res, indent=2)[:2400])
    for label, key in (("CYCLE NODES", "cycle_nodes"), ("ORPHANS", "orphans"),
                       ("MISSING NODES", "missing_nodes"),
                       ("DUPLICATE EDGES", "duplicate_edges")):
        v = res[key]
        print(f"  {label:16s} {'(none)' if not v else v}")
    render(g, res, Path(a.out))
    if a.json:
        Path(a.json).write_text(json.dumps({"graph": g, "analysis": res},
                                           indent=2), encoding="utf-8")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
