"""UHR-04 contract tests — the four amendments from the research sprint.

These are the FALSIFIABLE controls for the amendments. Each test names the
measured artifact it must reproduce or refuse. The load-bearing pair is
(c11a, c11b): the gate must REFUSE `ft09` and RATIFY `ka59`. A test that only
checked one of them could pass while measuring nothing.

Evidence classes: the ft09/ka59 constants are OBSERVED from the run receipts.
The gate's decision rule is a DESIGN choice; its falsifiable content is that it
separates these two REAL environments.
"""
from __future__ import annotations

import numpy as np
import pytest

# ----------------------------------------------------------------- Amendment 1
from henri_causal_contingency import (                                    # noqa: E402
    MIGRATION_TARGET_ENV,
    RETIRED_ENVIRONMENTS,
    STATUS_RATIFIED,
    STATUS_REFUSED_CONFOUNDED,
    STATUS_REFUSED_STATIC,
    action_contingency,
    admissibility,
    ratify_causal_link,
)


def _ft09_records(n_steps: int = 40) -> list:
    """The measured ft09 pattern: EVERY action moves the same 4 cells in row 63.

    Measured live: channels 4032..4095 (row 63 of a 64x64 grid), 32/32 steps,
    bit-identical diff. The policy alternates ACTION2, ACTION1, ... so the step
    index alone predicts the signature.
    """
    recs = []
    for t in range(n_steps):
        sig = (4032, 4033, 4034, 4035)          # constant -> no action contrast
        recs.append({
            "action": "GameAction.ACTION2" if t % 2 == 0 else "GameAction.ACTION1",
            "step": t,
            "change_signature": sig,
            "change_magnitude": 0.0009765625,
        })
    return recs


def _ka59_records(n_steps: int = 60, n_episodes: int = 6,
                  seed: int = 7) -> list:
    """The measured ka59 pattern: the signature IS action-contingent.

    Measured: per-action changed-cell means 15.73 / 17.34 / 18.60 / 11.17 with six
    distinct counts, and the action is not determined by the step index.
    """
    rng = np.random.default_rng(seed)
    acts = ["GameAction.ACTION1", "GameAction.ACTION2",
            "GameAction.ACTION3", "GameAction.ACTION4"]
    bases = {"GameAction.ACTION1": 15, "GameAction.ACTION2": 17,
             "GameAction.ACTION3": 18, "GameAction.ACTION4": 11}
    recs = []
    # EPISODES ARE REQUIRED. Conditional MI measures the within-step contrast, so
    # each step index must be observed under SEVERAL actions. A one-action-per-step
    # fixture makes the statistic UNCOMPUTABLE (the permutation null becomes
    # degenerate). The real corpus pools 540 records over 60 step indices (~9 per
    # step). That is what this mirrors.
    for _ep in range(n_episodes):
        for t in range(n_steps):
            a = acts[int(rng.integers(0, 4))]
            base = bases[a]
            # LOW-CARDINALITY signature, as the REAL harvest records it
            # (`("count", changed_cells)`, six distinct values across 540 records).
            # An index-set signature is near-unique per record, which makes the
            # permutation null DEGENERATE and the conditional-MI comparison
            # vacuous -- measured: observed == null q99 == 1.58408 exactly.
            # A small amount of state-dependent jitter keeps the action->count map
            # non-deterministic, so the contrast is statistical, not a lookup.
            jitter = int(rng.integers(-1, 2))
            sig = ("count", max(1, base + jitter))
            recs.append({"action": a, "step": t, "change_signature": sig,
                         "change_magnitude": (base + jitter) / 1024.0})
    return recs


def test_c11a_gate_refuses_the_ft09_cursor_band():
    """The amendment MUST refuse the measured cursor-band artifact.

    A nonzero-change test would ADMIT this: the band moves every step.
    """
    v = action_contingency(_ft09_records(), magnitude_key="change_magnitude",
                           min_obs=8, min_actions=2)
    assert v.status == STATUS_REFUSED_CONFOUNDED, (
        f"ft09 must be REFUSED_CONFOUNDED; got {v.status}")
    assert v.delta_s_ext > 0.0, "the band does move -- refusal must NOT be a static veto"
    assert v.mi_action_given_step <= v.null_q99, (
        "conditional MI must not clear the permutation null for a cursor band")


def test_c11b_gate_ratifies_the_reactive_environment():
    """The amendment MUST NOT refuse the environment we are migrating TO."""
    v = action_contingency(_ka59_records(), magnitude_key="change_magnitude",
                           min_obs=8, min_actions=2)
    assert v.status == STATUS_RATIFIED, (
        f"ka59 must be RATIFIED; got {v.status} :: {v.reasons}")
    assert v.n_step_matched_pairs > 0, (
        "the contrast is only computable when step indices are replicated "
        "across actions; this fixture must provide that")
    assert v.null_degenerate is False, (
        "the permutation null must SPREAD; a degenerate null makes the comparison "
        "vacuous and the module reports INSUFFICIENT")
    assert v.mi_action_given_step > v.null_q99


def test_c11c_static_world_triggers_the_solipsism_veto():
    """delta S_ext == 0 must be a DIFFERENT failure from CONFOUNDED."""
    recs = _ka59_records(n_steps=20)
    for r in recs:
        r["change_magnitude"] = 0.0
    g = ratify_causal_link(recs, magnitude_key="change_magnitude")
    assert g.status == STATUS_REFUSED_STATIC
    assert g.veto == "SOLIPSISM_VETO"
    assert not g.admissible


def test_c11d_ft09_is_on_the_retired_list_and_ka59_is_the_target():
    assert "ft09" in RETIRED_ENVIRONMENTS
    a = admissibility("ft09-0d8bbf25")
    assert a.admissible is False and a.is_retired is True
    b = admissibility(f"{MIGRATION_TARGET_ENV}-38d34dbb")
    assert b.admissible is True and b.is_migration_target is True


def test_c11e_uncomputable_contrast_is_insufficient_not_confounded():
    """One action per step => the contrast is UNCOMPUTABLE, which is not the same
    finding as CONFOUNDED. Reporting the wrong one would blame the world for a
    limitation of the design (measured: this exact fixture was mislabelled)."""
    from henri_causal_contingency import STATUS_REFUSED_INSUFFICIENT
    recs = [{"action": ("GameAction.ACTION1" if t % 2 == 0 else "GameAction.ACTION2"),
             "step": t,
             "change_signature": (t % 3, 4032 + (t % 2)),
             "change_magnitude": 0.001 * (1 + t % 2)}
            for t in range(40)]
    v = action_contingency(recs, magnitude_key="change_magnitude",
                           min_obs=8, min_actions=2)
    assert v.n_step_matched_pairs == 0, "fixture must have one action per step"
    assert v.status == STATUS_REFUSED_INSUFFICIENT, (
        f"an uncomputable contrast must be INSUFFICIENT; got {v.status}")
    assert v.status != STATUS_REFUSED_CONFOUNDED


def test_c11f_degenerate_null_reports_insufficient_not_confounded():
    """A near-unique signature makes the permutation null a point mass. The gate
    must SAY SO (INSUFFICIENT) instead of silently refusing as if the world were
    confounded. Measured: observed == null q99 == 1.58408 from exactly this shape
    of fixture."""
    from henri_causal_contingency import STATUS_REFUSED_INSUFFICIENT
    rng = np.random.default_rng(11)
    recs = []
    for _ep in range(6):
        for t in range(30):
            a = "GameAction.ACTION%d" % int(rng.integers(1, 5))
            sig = tuple(sorted(int(x) for x in rng.choice(4096, size=12, replace=False)))
            recs.append({"action": a, "step": t, "change_signature": sig,
                         "change_magnitude": 0.01})
    v = action_contingency(recs, magnitude_key="change_magnitude",
                           min_obs=8, min_actions=2, n_perm=50)
    assert v.signature_cardinality > 0.8 * v.n_obs, "fixture must be near-unique"
    assert v.null_degenerate is True, (
        f"degeneracy must be detected; null spread={v.null_q99 - v.null_mean:.3g}")
    assert v.status == STATUS_REFUSED_INSUFFICIENT, (
        f"degenerate null must be INSUFFICIENT; got {v.status}")


# ----------------------------------------------------------------- Amendment 3
from henri_mcts_observational_readout import (                            # noqa: E402
    READOUT_APPROXIMATE,
    READOUT_EXACT,
    READOUT_REFUSED_ROLE_NOT_UNITARY,
    binding_roundtrip_error,
    bind,
    circular_unbind,
    phase_only_role,
    readout_delta,
    readout_is_exact,
)


def test_c13a_exact_readout_recovers_the_bound_value():
    """A phase-only role must recover the bound value, and a WRONG role must not."""
    role = phase_only_role(256, 8, seed=0)
    assert readout_is_exact(role)
    rng = np.random.default_rng(1)
    value = np.random.default_rng(1).standard_normal((256, 8)).astype(np.float32)
    import torch
    v = torch.from_numpy(value)
    psi = bind(v, role)
    res = readout_delta(psi, role, v)
    assert res.status == READOUT_EXACT
    assert res.delta < 1e-5, f"exact recovery delta {res.delta}"

    wrong = phase_only_role(256, 8, seed=99)
    resw = readout_delta(psi, wrong, v)
    assert resw.delta > res.delta + 1e-3, "wrong-role control must be measurably worse"


def test_c13b_real_valued_role_is_refused_unless_opted_in():
    """The module must NOT launder an approximate readout as exact."""
    import torch
    g = torch.Generator().manual_seed(3)
    role = torch.nn.functional.normalize(torch.randn(128, 8, generator=g), dim=-1)
    value = torch.randn(128, 8, generator=torch.Generator().manual_seed(4))
    psi = bind(value, role)
    r = readout_delta(psi, role, value)
    assert r.status == READOUT_REFUSED_ROLE_NOT_UNITARY
    assert np.isnan(r.delta)
    assert r.role_roundtrip_error > 1e-6
    r2 = readout_delta(psi, role, value, allow_approximate=True)
    assert r2.status == READOUT_APPROXIMATE


def test_c13c_reduction_contract_is_structural_not_a_runtime_check():
    """renorm is DERIVED from `reduce`, so the renormalized-global combination
    cannot be expressed at all -- stronger than raising at runtime.

    Measured history: the original defect was `readout_delta` renormalizing per row
    and then taking ONE global angle, which reported delta=0.0321 for an exact
    readout. Asserting a raise would test a weaker contract than the code now
    provides, so this asserts the structural separation instead.
    """
    import torch
    role = phase_only_role(64, 8, seed=1)
    v = torch.randn(64, 8, generator=torch.Generator().manual_seed(5))
    psi = bind(v, role)

    # global => renorm=False internally, so per-row norms are NOT unit
    g = readout_delta(psi, role, v, reduce="global")
    assert g.ok, f"global reduction must be available; got {g.status}"
    raw = circular_unbind(psi, role, renorm=False)
    assert float((raw.norm(dim=-1) - 1.0).abs().max()) > 1e-6, (
        "global must use the UNrenormalized readout, else per-row scale factors "
        "contaminate the single angle")
    assert g.delta < 1e-5, f"global exact recovery delta {g.delta}"

    # per_row_mean => renorm=True, unit row norms, also exact
    p = readout_delta(psi, role, v, reduce="per_row_mean")
    assert p.ok and p.delta < 1e-5
    rn = circular_unbind(psi, role, renorm=True)
    assert float((rn.norm(dim=-1) - 1.0).abs().max()) < 1e-6

    # an unknown reduction is rejected outright
    try:
        readout_delta(psi, role, v, reduce="mean")
        raise AssertionError("unknown reduction must be rejected")
    except ValueError:
        pass


# ----------------------------------------------------------------- Amendment 4
from henri_parametric_manifold import (                                   # noqa: E402
    APPROXIMATE_INTERPOLATED,
    EXACT,
    SE2GeneratorBank,
    d4_group_actions,
    verify_exactness,
)


def test_c14a_se2_exactness_claims_hold():
    """Every exactness claim this module makes is verified, with controls."""
    for n in (4, 8):
        out = verify_exactness(n=n)
        assert out["ALL_EXACT_CHECKS_PASS"], f"n={n}: {out}"
        assert out["translation_vs_roll"] < 1e-9
        assert out["rotation90_vs_rot90"] == 0.0
        assert out["rot_perm_order4_residual"] == 0
        assert out["se2_law_residual"] < 1e-9
        # the non-abelian control MUST be large, else the law passes vacuously
        assert out["se2_wrong_pairing_residual"] > 1e-6
        assert out["se2_noncommutativity"] > 1e-6


def test_c14b_arbitrary_angle_is_refused_not_laundered():
    """A general rotation is NOT diagonal in the DFT basis; the module must say so."""
    bank = SE2GeneratorBank(8)
    psi = np.ones((8, 8), dtype=np.complex128)
    with pytest.raises(ValueError):
        bank.steer(psi, xi=[np.pi / 4, 0.0, 0.0])
    _, ex = bank.steer(psi, xi=[np.pi / 4, 0.0, 0.0], approximate_rotation=True)
    assert ex == APPROXIMATE_INTERPOLATED
    _, ex2 = bank.steer(psi, xi=[np.pi / 2, 0.0, 0.0])
    assert ex2 == EXACT


def test_c14c_generator_bank_covers_the_d4_vocabulary():
    """The pre-structured bank must name the same ops the MCTS enumerates."""
    names = {e.name for e in SE2GeneratorBank(8).group_closure()}
    d4 = set(d4_group_actions()) - {"Identity"}
    assert d4 <= names, f"missing from the bank: {d4 - names}"


# ----------------------------------------------------------------- Amendment 2 wiring
def test_c12a_readout_is_wired_into_expansion_and_gated():
    """The readout must be reachable from MCTS expansion AND default OFF."""
    import os
    import pathlib
    src = pathlib.Path("sagnac_mcts_planner.py").read_text(encoding="utf-8")
    assert "HENRI_MCTS_OBSERVATIONAL_READOUT" in src
    assert 'os.environ.get("HENRI_MCTS_OBSERVATIONAL_READOUT", "0") == "1"' in src
    assert "delta_readout" in src
    # default OFF
    assert os.environ.get("HENRI_MCTS_OBSERVATIONAL_READOUT", "0") == "0"


def test_c12b_readout_does_not_veto_or_select():
    """A diagnostic must not be able to prune or to pick the returned plan."""
    import pathlib
    src = pathlib.Path("sagnac_mcts_planner.py").read_text(encoding="utf-8")
    start = src.find("UHR-04 AMENDMENT 3: observational readout in expansion")
    assert start > 0, "the amendment-3 block is absent"
    end = src.find("OBSERVATIONAL CHANNEL", start)
    block = src[start:end if end > start else start + 4000]
    assert "is_pruned = True" not in block, "the readout must never prune"
    assert "best_node = child_node" not in block, "the readout must not select"
    assert "return child_node.ast_node" not in block, "the readout must not return"


def test_c12c_tau_veto_is_untouched():
    """The Sagnac veto threshold must not be relaxed by this work."""
    import pathlib
    src = pathlib.Path("sagnac_mcts_planner.py").read_text(encoding="utf-8")
    assert "tau_veto: float = 0.35" in src, "the sealed default tau_veto changed"


# ------------------------------------- Amendment 2b: LEDGER-LEVEL null control
def test_c12d_ft09_earns_zero_zone_c_ratifications():
    """ft09 MUST produce ZERO Zone-C ratifications AT THE LEDGER.

    Not merely a launch-harness refusal. The ledger's own nonzero-change gate
    (ext_delta == 0 -> SOLIPSISM_VETO) CANNOT catch ft09, because the cursor band
    moves bit-identically on every step (measured frame_diff_mean 0.0009765625).
    This test drives the ft09 pattern through the REAL DAG and asserts that the
    contingency gate refuses every forged edge.
    """
    import numpy as np
    import torch
    from zone_c_causal_engram_dag import (
        CONFOUNDED_VETO, SOLIPSISM_VETO, ZoneCCausalEngramDAG,
    )
    from henri_causal_contingency import ratify_causal_link

    # --- the measured ft09 signature: constant change, alternating schedule ---
    recs = [{"action": ("GameAction.ACTION2" if t % 2 == 0 else "GameAction.ACTION1"),
             "step": t,
             "change_signature": (4032, 4033, 4034, 4035),   # CONSTANT, row 63
             "change_magnitude": 0.0009765625}               # bit-identical
            for t in range(40)]
    verdict = ratify_causal_link(recs, magnitude_key="change_magnitude")
    assert verdict.admissible is False, "ft09 must NOT be ratifiable"
    assert verdict.status == "REFUSED_CONFOUNDED"
    assert verdict.status != "REFUSED_STATIC", (
        "the refusal must NOT be a static/zero-change veto: the band DID move")

    # --- drive the REAL ledger with that verdict ---
    dag = ZoneCCausalEngramDAG()
    n, d = 128, 8
    g = torch.Generator().manual_seed(0)
    wave_prev = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
    wave_next = torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)
    gens = [torch.randn(3, 3, dtype=torch.complex64, generator=g)]
    basis = torch.zeros(8, 3, 3, dtype=torch.complex64)
    dag.add_node("s0", wave_prev, contract="ft09")

    out = dag.forge_edge("s0", wave_next, action=0, ext_delta=0.0009765625,
                         truth_generators=gens, gell_mann_basis=basis,
                         contingency=verdict)
    assert out.forged is False, (
        "the ledger MUST refuse a confounded edge even though ext_delta > 0")
    assert out.reason == CONFOUNDED_VETO, (
        f"expected {CONFOUNDED_VETO}, got {out.reason}")
    assert out.reason != SOLIPSISM_VETO, (
        "the refusal must be distinguishable from the static-world veto")
    assert dag.store_size() == 0, f"ZERO ratifications required; got {dag.store_size()}"

    # --- the reactive control: ka59 DOES ratify at the ledger ---
    rng = np.random.default_rng(7)
    ka = []
    for ep in range(6):
        for t in range(30):
            a = "GameAction.ACTION%d" % int(rng.integers(1, 5))
            base = {"GameAction.ACTION1": 15, "GameAction.ACTION2": 17,
                    "GameAction.ACTION3": 18, "GameAction.ACTION4": 11}[a]
            ka.append({"action": a, "step": t,
                       "change_signature": ("count", max(1, base + int(rng.integers(-1, 2)))),
                       "change_magnitude": base / 1024.0})
    kv = ratify_causal_link(ka, magnitude_key="change_magnitude")
    assert kv.admissible is True, f"ka59 must ratify; got {kv.status} :: {kv.reasons}"

    dag2 = ZoneCCausalEngramDAG()
    dag2.add_node("s0", wave_prev, contract="ka59")
    out2 = dag2.forge_edge("s0", wave_next, action=0, ext_delta=1.0 / 1024.0,
                           truth_generators=gens, gell_mann_basis=basis,
                           contingency=kv)
    # the conjunction check may still refuse on tau, but the CONTINGENCY gate must pass
    assert out2.reason != CONFOUNDED_VETO, (
        "a RATIFIED verdict must not be refused by the contingency gate")


def test_c12e_no_contingency_keeps_the_old_path_identical():
    """Omitting `contingency` must leave the ledger's behaviour unchanged."""
    import torch
    from zone_c_causal_engram_dag import SOLIPSISM_VETO, ZoneCCausalEngramDAG
    dag = ZoneCCausalEngramDAG()
    g = torch.Generator().manual_seed(1)
    dag.add_node("s0", torch.nn.functional.normalize(torch.randn(64, 8, generator=g), dim=-1), contract="c")
    out = dag.forge_edge("s0", torch.nn.functional.normalize(torch.randn(64, 8, generator=g), dim=-1), action=0,
                         ext_delta=0.0,
                         truth_generators=[torch.randn(3, 3, dtype=torch.complex64, generator=g)],
                         gell_mann_basis=torch.zeros(8, 3, 3, dtype=torch.complex64))
    assert out.forged is False and out.reason == SOLIPSISM_VETO
    assert dag.store_size() == 0
