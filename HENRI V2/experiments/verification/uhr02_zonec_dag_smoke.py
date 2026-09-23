"""UHR-02 / Zone C causal engram DAG — functional smoke with the mandatory controls.

Runs deterministically on CPU. Exit 0 = every control behaved. This doubles as the
evidence artifact for the blueprint section 3 implementation.

CONTROLS THAT MUST BE ABLE TO FAIL
  1. SOLIPSISM_VETO   : ext_delta == 0 forges NOTHING (the anti-solipsism gate)
  2. NO_TEMPORAL_PRIORITY : a successor at or before the source time forges nothing
  3. CONJUNCTION_FAILED : a candidate residual above tau forges nothing
  4. DEAD-INPUT negative control : a never-reinforced hypothesis MUST be pruned
     by the Landauer clock, and a reinforced one MUST survive it
  5. DOMAIN_VIOLATION : a complex operand RAISES (the recorded cross-family defect)
  6. ATTRIBUTION_VIOLATION : scoring a candidate against an unrecorded transition RAISES
"""
from __future__ import annotations

import math
import sys

import torch

sys.path.insert(0, r"C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2")
from uhr02_exteroceptive_gate import ad_of, delta, predict_next  # noqa: E402
from zone_c_causal_engram_dag import (  # noqa: E402
    SOLIPSISM_VETO,
    ZoneCCausalEngramDAG,
    retention_energy,
)

K = 2048
_s3 = math.sqrt(3.0)
BASIS = torch.tensor([
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]], [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], [[1 / _s3, 0, 0], [0, 1 / _s3, 0], [0, 0, -2 / _s3]],
], dtype=torch.complex64)


def roles(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(K, 8, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def gens(seed: int, sc: float = 0.25, k: int = 4) -> list:
    g = torch.Generator().manual_seed(seed)
    th = torch.randn(8, generator=g) * sc
    return [1j * torch.einsum("a,aij->ij", th.to(BASIS.dtype), BASIS) for _ in range(k)]


def nxt(state: torch.Tensor, gs: list, noise: float = 0.0, seed: int = 11) -> torch.Tensor:
    y = predict_next(state, ad_of(gs, BASIS))
    if noise:
        y = y + noise * torch.randn(y.shape, generator=torch.Generator().manual_seed(seed))
    return y / y.norm(dim=-1, keepdim=True)


def main() -> int:
    print("Zone C causal engram DAG — functional smoke")
    print("K =", K, "| tau = 0.35 | tau_retain = 64 ticks | prune_floor = 0.05")
    print()

    R0 = roles(1)
    gs = gens(4242)
    obs = nxt(R0, gs, noise=0.02, seed=11)
    ok = {}

    dag = ZoneCCausalEngramDAG()
    dag.add_node("s0", R0, contract="grid_boundary")
    print("nodes after add:", dag.count_populated())

    # 1. SOLIPSISM_VETO -- checked first, so it is not masked by timing.
    #    NO advance() here on purpose: the solipsism gate must fire even at the
    #    source's own tick.
    o = dag.forge_edge("s0", obs, action=0, ext_delta=0.0, truth_generators=gs,
                       gell_mann_basis=BASIS)
    ok["1_SOLIPSISM_VETO"] = (o.forged is False and o.reason == SOLIPSISM_VETO)
    print("  [1] ext_delta=0     -> forged=%s reason=%s store=%d" % (o.forged, o.reason, dag.store_size()))

    # 2. NO_TEMPORAL_PRIORITY (successor at the source's own tick)
    o = dag.forge_edge("s0", obs, action=0, ext_delta=0.5, truth_generators=gs,
                       gell_mann_basis=BASIS, t_dst=0)
    ok["2_NO_TEMPORAL_PRIORITY"] = (o.forged is False and o.reason == "NO_TEMPORAL_PRIORITY")
    print("  [2] t_dst<=t_src    -> forged=%s reason=%s" % (o.forged, o.reason))

    # the observation arrives at the NEXT tick: advance the clock, then forge
    dag.advance(1)

    # 3. CONJUNCTION_FAILED (an unrelated option predicts badly -> above tau)
    dag2 = ZoneCCausalEngramDAG()
    dag2.add_node("s0", R0, contract="grid_boundary")
    dag2.advance(1)
    gs_bad = gens(9999)
    obs_bad = nxt(R0, gs, noise=0.02, seed=11)          # env did the TRUE transition
    o = dag2.forge_edge("s0", obs_bad, action=1, ext_delta=0.5, truth_generators=gs_bad,
                        gell_mann_basis=BASIS)
    resid = delta(predict_next(R0, ad_of(gs_bad, BASIS)), obs_bad)
    ok["3_CONJUNCTION_FAILED"] = (o.forged is False and o.reason == "CONJUNCTION_FAILED")
    print("  [3] wrong option    -> forged=%s reason=%s residual=%.6f (tau=0.35)"
          % (o.forged, o.reason, resid))

    # 4. GOOD EDGE forges
    o = dag.forge_edge("s0", obs, action=2, ext_delta=0.5, truth_generators=gs,
                       gell_mann_basis=BASIS, dst="s1")
    ok["4_EDGE_FORGED"] = bool(o.forged and dag.store_size() == 1)
    print("  [4] true option     -> forged=%s dst=%s delta_pred=%.6f store=%d"
          % (o.forged, (o.edge.dst if o.edge else None),
             (o.edge.delta_pred if o.edge else float("nan")), dag.store_size()))

    # 5. DEAD-INPUT NEGATIVE CONTROL: an unreinforced hypothesis MUST be pruned
    #    by the Landauer clock; a reinforced one MUST survive the SAME clock.
    #
    #    TWO fixture defects fixed here (both mine, found by running it):
    #    (a) the nodes were forged at the source's own tick, so BOTH forges
    #        returned NO_TEMPORAL_PRIORITY and `e1.edge` was None -> the control
    #        silently tested nothing. The clock is now advanced between add and
    #        forge.
    #    (b) 48 idle ticks cannot reach the 0.05 floor: 2^(-48/64) = 0.594.
    #        The pruning crossover for floor f is idle = tau_retain * log2(1/f),
    #        i.e. 276 ticks at tau_retain=64, f=0.05. The loop now runs 320.
    dag3 = ZoneCCausalEngramDAG()
    dag3.add_node("a", roles(2), contract="c")
    dag3.add_node("b", roles(3), contract="c")
    dag3.advance(1)                                     # observation arrives next tick
    gs_b = gens(5150)
    e1 = dag3.forge_edge("a", nxt(roles(2), gs, 0.02, 11), action=0, ext_delta=0.4,
                         truth_generators=gs, gell_mann_basis=BASIS, dst="a_next")
    e2 = dag3.forge_edge("b", nxt(roles(3), gs_b, 0.02, 11), action=0, ext_delta=0.4,
                         truth_generators=gs_b, gell_mann_basis=BASIS, dst="b_next")
    both = bool(e1.forged and e2.forged)
    assert both, "fixture: both edges must forge before the pruning control is meaningful"
    k1, k2 = e1.edge.key, e2.edge.key

    all_pruned: list = []
    for _ in range(5):
        all_pruned += dag3.advance(ticks=64)            # 320 idle ticks total
        dag3.reinforce(k1)                              # keep e1 alive across the clock
    keys_after = {e.key for e in dag3.edges()}
    ok["5_DEAD_INPUT_PRUNED"] = (k2 not in keys_after) and any(p.key == k2 for p in all_pruned)
    ok["5_REINFORCED_SURVIVES"] = k1 in keys_after
    ok["5_PRUNE_LOGGED"] = bool(all_pruned) and all(
        p.reason == "LANDAUER_APOPTOSIS" for p in all_pruned)
    print("  [5] forged=%s pruned=%d  reinforced-survives=%s  dead-input-pruned=%s  reasons=%s"
          % (both, len(all_pruned), k1 in keys_after, k2 not in keys_after,
             sorted({p.reason for p in all_pruned})))
    print("      retention_energy(idle=0)=%.6f  idle=64 -> %.6f  idle=276 -> %.6f  idle=320 -> %.6f"
          % (retention_energy(0.0), retention_energy(64.0),
             retention_energy(276.0), retention_energy(320.0)))

    # 6. DOMAIN_VIOLATION on a complex operand
    try:
        dag3.add_node("bad", torch.randn(K, 8, dtype=torch.complex64), contract="c")
        ok["6_DOMAIN_VIOLATION"] = False
    except ValueError as e:
        ok["6_DOMAIN_VIOLATION"] = "DOMAIN_VIOLATION" in str(e)
        print("  [6] complex operand -> RAISED (%s...)" % str(e)[:52])

    # 7. ATTRIBUTION_VIOLATION: score against an unrecorded transition
    dag4 = ZoneCCausalEngramDAG()
    dag4.add_node("lonely", roles(7), contract="c")
    try:
        dag4.gate_candidate("lonely", gs, BASIS)
        ok["7_ATTRIBUTION_VIOLATION"] = False
    except ValueError as e:
        ok["7_ATTRIBUTION_VIOLATION"] = "ATTRIBUTION_VIOLATION" in str(e)
        print("  [7] no recorded edge -> RAISED (%s...)" % str(e)[:52])

    # 8. READ PATH on the good DAG: the gated residual must separate true from other
    g_other = gens(9999)
    r_true = dag.gate_candidate("s0", gs, BASIS)
    r_othr = dag.gate_candidate("s0", g_other, BASIS)
    sep = abs(r_true["delta_pred"] - r_othr["delta_pred"])
    ok["8_READ_PATH_SEPARATES"] = sep > 0.05
    print("  [8] gate true=%.6f other=%.6f sep=%.6f vetoed=%s/%s"
          % (r_true["delta_pred"], r_othr["delta_pred"], sep,
             r_true["hard_vetoed"], r_othr["hard_vetoed"]))

    print()
    print("VERDICT")
    for k in sorted(ok):
        print("  %-26s %s" % (k, ok[k]))
    allok = all(ok.values())
    print()
    print("RESULT=%s" % ("ALL_CONTROLS_PASS" if allok else "CONTROL_FAILED"))
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
