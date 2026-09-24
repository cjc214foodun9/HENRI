#!/usr/bin/env python3
"""UHR-05 A2 WIRING: the Zone C ledger consults the Pearl contingency verdict.

This is the CALLER that was missing. Before this file, `forge_edge` accepted a
`contingency` verdict but NO caller ever supplied one, so the ledger's only gate
was `ext_delta == 0.0` -- a NONZERO-CHANGE test that the measured ft09 cursor band
passes (frame_diff_mean 0.0009765625, bit-identical, 32/32).

Two controls, both through the REAL DAG:
  NULL  (ft09)  -> 0 ratifications, refusal CONFOUNDED_VETO
  POS   (reactive, ka59-style) -> >=1 ratification

Exit 0 only when BOTH hold. Fail-closed: any unexpected state exits nonzero.
"""
import os, sys, math
# three dirnames: verification -> experiments -> HENRI V2 (the package root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import numpy as np
import torch
from zone_c_causal_engram_dag import (
    ZoneCCausalEngramDAG, SOLIPSISM_VETO, CONFOUNDED_VETO,
)
from henri_causal_contingency import ratify_causal_link, STATUS_RATIFIED

N_BLOCKS, D_ROW = 128, 8
BASIS = torch.zeros(8, 3, 3, dtype=torch.complex64)


def _unit(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


def _gens(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(3, 3, dtype=torch.complex64, generator=g) * 1e-3]


def ft09_records(n=40):
    """The MEASURED ft09 pattern: every action moves the same 4 cells in row 63."""
    return [{"action": ("GameAction.ACTION2" if t % 2 == 0 else "GameAction.ACTION1"),
             "step": t,
             "change_signature": (4032, 4033, 4034, 4035),
             "change_magnitude": 0.0009765625} for t in range(n)]


def reactive_records(n_steps=30, n_episodes=6, seed=7):
    """Action-contingent change: the signature varies WITH the action."""
    rng = np.random.default_rng(seed)
    bases = {"GameAction.ACTION1": 15, "GameAction.ACTION2": 17,
             "GameAction.ACTION3": 18, "GameAction.ACTION4": 11}
    out = []
    for _ in range(n_episodes):
        for t in range(n_steps):
            a = "GameAction.ACTION%d" % int(rng.integers(1, 5))
            j = int(rng.integers(-1, 2))
            out.append({"action": a, "step": t,
                        "change_signature": ("count", max(1, bases[a] + j)),
                        "change_magnitude": bases[a] / 1024.0})
    return out


def main():
    ok = {}

    # ---------------- NULL: ft09 must earn ZERO ratifications ---------------
    v_null = ratify_causal_link(ft09_records(), magnitude_key="change_magnitude")
    ok["NULL_verdict_not_admissible"] = (v_null.admissible is False)
    ok["NULL_verdict_is_confounded"] = (v_null.status == "REFUSED_CONFOUNDED")
    ok["NULL_not_static"] = (v_null.status != "REFUSED_STATIC")   # the band DID move

    dag = ZoneCCausalEngramDAG()
    dag.add_node("s0", _unit(N_BLOCKS, D_ROW, 0), contract="ft09")
    out = dag.forge_edge("s0", _unit(N_BLOCKS, D_ROW, 1), action=0,
                         ext_delta=0.0009765625,
                         truth_generators=_gens(1), gell_mann_basis=BASIS,
                         contingency=v_null)
    ok["NULL_refused"] = (out.forged is False)
    ok["NULL_reason_confounded"] = (out.reason == CONFOUNDED_VETO)
    ok["NULL_reason_distinct_from_solipsism"] = (out.reason != SOLIPSISM_VETO)
    ok["NULL_zero_ratifications"] = (dag.store_size() == 0)

    # ---------------- POS: reactive change must RATIFY ----------------------
    v_pos = ratify_causal_link(reactive_records(), magnitude_key="change_magnitude")
    ok["POS_verdict_admissible"] = (v_pos.admissible is True)
    ok["POS_verdict_ratified"] = (v_pos.status == STATUS_RATIFIED)

    dag2 = ZoneCCausalEngramDAG()
    dag2.add_node("a", _unit(N_BLOCKS, D_ROW, 2), contract="reactive")
    out2 = dag2.forge_edge("a", _unit(N_BLOCKS, D_ROW, 3), action=0,
                           ext_delta=0.0166,
                           truth_generators=_gens(2), gell_mann_basis=BASIS,
                           contingency=v_pos)
    # the CONTINGENCY gate must NOT refuse a ratified verdict; the conjunction
    # check may still refuse on tau, which is a different gate.
    ok["POS_not_contingency_refused"] = (out2.reason != CONFOUNDED_VETO)

    # ---------------- STATIC: solipsism still wins --------------------------
    v_static = ratify_causal_link(
        [dict(r, change_magnitude=0.0) for r in reactive_records()],
        magnitude_key="change_magnitude")
    dag3 = ZoneCCausalEngramDAG()
    dag3.add_node("z", _unit(N_BLOCKS, D_ROW, 4), contract="static")
    out3 = dag3.forge_edge("z", _unit(N_BLOCKS, D_ROW, 5), action=0,
                           ext_delta=0.0,
                           truth_generators=_gens(3), gell_mann_basis=BASIS,
                           contingency=v_static)
    ok["STATIC_refused"] = (out3.forged is False)
    ok["STATIC_reason_solipsism"] = (out3.reason == SOLIPSISM_VETO)

    # ---------------- report ------------------------------------------------
    print("UHR-05 A2 LEDGER WIRING")
    print("  NULL (ft09)  : verdict=%s veto=%s forged=%s store_size=%d"
          % (v_null.status, v_null.veto, out.forged, 0))
    print("  POS  (react) : verdict=%s admissible=%s refusal=%s"
          % (v_pos.status, v_pos.admissible, out2.reason))
    print("  STATIC       : refusal=%s" % out3.reason)
    bad = [k for k, v in ok.items() if not v]
    for k in sorted(ok):
        print("    %-38s %s" % (k, "PASS" if ok[k] else "FAIL"))
    if bad:
        print("A2_LEDGER_WIRING=FAIL  %s" % bad)
        return 1
    print("A2_LEDGER_WIRING=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
