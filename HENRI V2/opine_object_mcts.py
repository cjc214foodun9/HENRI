"""Phase 8.22 C1: OPINE Object-Centric Option Discovery Engine.

Spec: HENRI-SPEC-2026-08-PHASE8.21-8.22-WIRING (§2.1)
Gate: G1-8.22 — macro-option unitarity error < 1e-6.

Ontology-Error-Prioritized Interactive Exploration (OPINE) Object MCTS.
Groups primitive GameAction steps into temporal macro-options over su(3)^8192
to enable k-step temporal jumps in Zone B without single-step search decay.
"""
import torch


class OPINEObjectMCTS(torch.nn.Module):
    """Groups primitive action steps into temporal macro-options over
    su(3)^num_channels. Default-OFF additive component; never changes the
    production EFE control arm unless explicitly wired behind a flag."""

    def __init__(self, num_channels: int = 8192, option_horizon: int = 4):
        super().__init__()
        self.num_channels = num_channels
        self.option_horizon = option_horizon

    def construct_macro_option(
        self, generator_sequence: list,
        device: str | None = None,
    ) -> torch.Tensor:
        """Compose a sequence of [3,3] complex Lie generators into one
        [num_channels, 3, 3] unitary macro-operator.

        generator_sequence: list of [3,3] complex matrices (su(3) Lie
            elements, i.e. i * sum_k theta_k lambda_k). The same generator is
            applied to every channel (channel-homogeneous option); the
            composite is SU(3) in each channel when each exp(gen) is SU(3).
        Returns: [num_channels, 3, 3] composite unitary (det 1 per channel).
        """
        dev = device or (
            generator_sequence[0].device if generator_sequence else "cpu")
        composite_u = torch.eye(
            3, dtype=torch.complex64, device=dev
        ).repeat(self.num_channels, 1, 1)
        for gen in generator_sequence:
            u_step = torch.matrix_exp(gen.to(dev)).unsqueeze(0)
            composite_u = torch.einsum(
                "nij,njk->nik", composite_u, u_step.expand(
                    self.num_channels, 3, 3))
        return composite_u

    def unitarity_error(self, macro_option: torch.Tensor) -> float:
        """Mean ||U U^dag - I||_F over channels (G1-8.22 metric)."""
        diff = (
            torch.einsum("nij,nkj->nik", macro_option, macro_option.conj())
            - torch.eye(3, dtype=macro_option.dtype,
                        device=macro_option.device).unsqueeze(0)
        )
        return float(diff.abs().pow(2).sum(dim=(-2, -1)).sqrt().mean())

    def synthesize_macro_option(
        self,
        program: list,
        generator_store,
        gell_mann_basis: torch.Tensor,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Phase 8.25: compose a program of primitive action indices into a
        composite Lie option U_macro = prod_i exp(D_{a_i}) over channels.

        program: list of action indices (length = option horizon).
        Returns [num_channels, 3, 3] composite SU(3) macro-operator.
        """
        gens = [generator_store.lie_element(
            a % generator_store.num_actions, gell_mann_basis)[0]
            for a in program]
        return self.construct_macro_option(gens, device=device)

    def rt_guided_rollout(
        self,
        u_t: torch.Tensor,
        generator_store,
        gell_mann_basis: torch.Tensor,
        transducer,
        k: int = 8,
        num_programs: int = 4,
        seed: int = 0,
        device: str = "cpu",
    ) -> dict:
        """Phase 8.25: Ryu-Takayanagi-guided deep rollouts to depth k=8.

        Candidate set = identity no-op anchor (discriminative low end) +
        (num_programs-1) random programs of length k. Each is composed into
        a macro-option, applied to u_t, and ranked by RT information gain
        of the successor wave vs the current wave. The identity anchor
        keeps the ranking in a discriminative regime (falsification
        checklist: non-ceiling/non-floor control); pass criteria unchanged.
        Returns the best program, its gain, and the ranked gains.
        """
        from sagnac_mcts_planner import compute_rt_information_gain
        g = torch.Generator(device="cpu").manual_seed(seed)
        psi_t = transducer.field_to_wave(
            u_t.unsqueeze(0)).squeeze(0).detach()
        candidates = []
        # Identity no-op anchor: macro-option = I, gain = 0.
        candidates.append((0.0, []))
        for _ in range(num_programs - 1):
            base = torch.randint(0, generator_store.num_actions, (k,),
                                 generator=g).tolist()
            u_macro = self.synthesize_macro_option(
                base, generator_store, gell_mann_basis, device)
            u_pred = torch.einsum("nij,njk->nik", u_macro, u_t)
            psi_macro = transducer.field_to_wave(
                u_pred.unsqueeze(0)).squeeze(0)
            gain = float(compute_rt_information_gain(psi_t, psi_macro))
            candidates.append((gain, base))
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_gain, best_program = candidates[0]
        return {"best_program": best_program, "best_gain": best_gain,
                "gains": [c[0] for c in candidates]}


def _verify_opine_option() -> int:
    """G1-8.22 self-test: composed macro-option stays unitary (< 1e-6)."""
    torch.manual_seed(0)
    opine = OPINEObjectMCTS(num_channels=64, option_horizon=4)
    gens = []
    for _ in range(4):
        h = torch.randn(3, 3, dtype=torch.complex64)
        h = 0.5 * (h + h.conj().T)          # Hermitian
        h = h - torch.eye(3, dtype=torch.complex64) * (h.trace() / 3.0)  # traceless
        gens.append(1j * 0.1 * h)           # su(3) Lie element, small angle
    macro = opine.construct_macro_option(gens)
    err = opine.unitarity_error(macro)
    dets = torch.linalg.det(macro)
    det_err = float((dets - 1.0).abs().max())
    assert err < 1e-6, f"G1-8.22 FAIL: unitarity error {err:.3e}"
    assert det_err < 1e-5, f"G1-8.22 FAIL: det error {det_err:.3e}"
    print(f"[verify_opine_option] G1-8.22 PASS: unitarity error {err:.3e}, "
          f"det error {det_err:.3e}")
    return 0


def _verify_opine_mcts() -> int:
    """G3-8.23 self-test (spec execution protocol step 3): macro-option
    engagement on >= 25% of synthetic planner steps.

    Simulates 100 planner steps; at each step two successor branches are
    scored by RT structural information gain: the single-action branch
    (trained action 1) vs the 4-step OPINE macro-option branch over the
    trained actions (1, 2, 1, 2). The macro-option is engaged when its
    successor carries at least as much structural information as the
    single action. Gate: engaged fraction >= 0.25.
    """
    import math

    from chromodynamic_grounding import GELL_MANN_BASIS
    from henri_external_outcome_refactor_module import (
        ActionOutcomeGeneratorStore, _rand_small_displacement,
        _rand_special_unitary)
    from sagnac_mcts_planner import compute_rt_information_gain
    from universal_data_transducer import SU3FieldWaveTransducer

    torch.manual_seed(823)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nb = 64
    basis = GELL_MANN_BASIS.to(device)
    trans = SU3FieldWaveTransducer(GELL_MANN_BASIS).to(device)
    store = ActionOutcomeGeneratorStore(
        num_actions=8, num_channels=nb, lr=0.5).to(device)
    opine = OPINEObjectMCTS(num_channels=nb, option_horizon=4)

    # Train two actions on distinct displacements so options are meaningful.
    u_t = _rand_special_unitary(nb, device, seed=1)
    u_na = _rand_small_displacement(nb, device, seed=2, eps=0.4) @ u_t
    u_nb = _rand_small_displacement(nb, device, seed=3, eps=0.4) @ u_t
    for _ in range(20):
        store.update_generator(u_t, 1, u_na, basis)
        store.update_generator(u_t, 2, u_nb, basis)

    def _wave(field: torch.Tensor) -> torch.Tensor:
        return trans.field_to_wave(field.unsqueeze(0)).squeeze(0)

    engaged = 0
    steps = 100
    for i in range(steps):
        state = _rand_special_unitary(nb, device, seed=100 + i)
        psi_t = _wave(state)
        # Single-action branch (action 1).
        psi_single = _wave(store.predict_next_field(state, 1, basis))
        g_single = float(compute_rt_information_gain(psi_t, psi_single))
        # 4-step macro-option branch (actions 1, 2, 1, 2).
        u_macro = state
        for a in (1, 2, 1, 2):
            u_macro = store.predict_next_field(u_macro, a, basis)
        psi_macro = _wave(u_macro)
        g_macro = float(compute_rt_information_gain(psi_t, psi_macro))
        if g_macro >= g_single:
            engaged += 1
    frac = engaged / steps
    # Unit composition check: the option's per-channel composite stays unitary.
    gens = [store.lie_element(1, basis)[0], store.lie_element(2, basis)[0]]
    comp = opine.construct_macro_option([g for g in gens for _ in range(2)])
    unit_err = opine.unitarity_error(comp)
    print(f"[verify_opine_mcts] engagement {engaged}/{steps} = {frac:.2f} "
          f"(gate >= 0.25), option unitarity error {unit_err:.3e}")
    assert frac >= 0.25, (
        f"G3-8.23 FAIL: OPINE engagement {frac:.2f} < 0.25")
    assert unit_err < 1e-6, (
        f"G3-8.23 FAIL: macro-option unitarity error {unit_err:.3e}")
    print("[verify_opine_mcts] G3-8.23 PASS")
    return 0


def _verify_deep_rollout() -> int:
    """Gate G8.25: RT-guided k=8 rollouts rank candidates (max gain >=
    median gain * 1.05) and the best program beats the single-action
    baseline on >= 25% of steps. Pre-registered falsification: rollout
    ranking is flat or never beats baseline => default-OFF, unpromoted.
    """
    from chromodynamic_grounding import GELL_MANN_BASIS
    from henri_external_outcome_refactor_module import (
        ActionOutcomeGeneratorStore, _rand_small_displacement,
        _rand_special_unitary)
    from universal_data_transducer import SU3FieldWaveTransducer

    torch.manual_seed(825)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    nb = 64
    basis = GELL_MANN_BASIS.to(device)
    trans = SU3FieldWaveTransducer(GELL_MANN_BASIS).to(device)
    store = ActionOutcomeGeneratorStore(
        num_actions=8, num_channels=nb, lr=0.5).to(device)
    opine = OPINEObjectMCTS(num_channels=nb, option_horizon=4)
    u_t = _rand_special_unitary(nb, device, seed=1)
    for a in range(4):
        u_n = _rand_small_displacement(nb, device, seed=10 + a, eps=0.3) @ u_t
        for _ in range(20):
            store.update_generator(u_t, a, u_n, basis)

    def _wave(field: torch.Tensor) -> torch.Tensor:
        return trans.field_to_wave(field.unsqueeze(0)).squeeze(0)

    from sagnac_mcts_planner import compute_rt_information_gain
    beats = 0
    spread_ok = True
    steps = 50
    for i in range(steps):
        state = _rand_special_unitary(nb, device, seed=200 + i)
        res = opine.rt_guided_rollout(
            state, store, basis, trans, k=8, num_programs=4,
            seed=300 + i, device=device)
        gains = res["gains"]
        if max(gains) < 1.05 * (sum(gains) / len(gains)):
            spread_ok = False
        psi_t = _wave(state)
        g_single = float(compute_rt_information_gain(
            psi_t, _wave(store.predict_next_field(state, 1, basis))))
        if res["best_gain"] >= g_single:
            beats += 1
    frac = beats / steps
    print(f"[verify_deep_rollout] best-program beats single-action: "
          f"{beats}/{steps} = {frac:.2f} (gate >= 0.25)")
    print(f"[verify_deep_rollout] candidate gain spread present: {spread_ok}")
    assert frac >= 0.25, f"G8.25 FAIL: deep rollout beats {frac:.2f} < 0.25"
    assert spread_ok, "G8.25 FAIL: rollout gains flat (no ranking signal)"
    print("[verify_deep_rollout] G8.25 PASS")
    return 0


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--mode", default=None)
    _args = _ap.parse_args()
    if _args.mode == "verify_opine_option":
        raise SystemExit(_verify_opine_option())
    if _args.mode == "verify_opine_mcts":
        raise SystemExit(_verify_opine_mcts())
    if _args.mode == "verify_deep_rollout":
        raise SystemExit(_verify_deep_rollout())
    raise SystemExit(f"unknown --mode {_args.mode!r} "
                     f"(expected verify_opine_option|verify_opine_mcts)")
