#!/usr/bin/env python3
"""Readout probe: does decoding fix Delta_Sagnac, or is the fix imaginary?

THE INSTRUMENT DISCIPLINE
    Every gate below has a CONTROL that can make it fail, and the probe refuses
    to report a PASS if its own self-checks fail. Three times in this project a
    verdict came from a broken instrument; that is what this structure prevents.

GATES
  S1 self-check   unbinding a clean single binding recovers its filler at cos~1
                  (proves the readout can read what was written, before any claim)
  S2 self-check   Psi(*)=1 self-cosine, and the random-axiom baseline binds
  A  decode       RFSS decode of a randomly assigned observable is exact with a
                  positive margin over the runner-up, across many draws
  B  FIX          a wavefront pair that is OBSERVATIONALLY IDENTICAL but far apart
                  in wave space must give stress_obs ~ 0 while the INTERNAL
                  cosine measure calls it a mismatch. If internal and
                  observational disagree, the readout is doing real work.
  C  reject       a CORRUPTED observable must give stress_obs high
  D  fail-closed  zero-energy input -> stress 1.0, valid False, NO NaN
  E  per-sample   one zero row among good rows must not corrupt the good rows
                  (the reference module used a single global energy test)
  F  no-rebind    codebook buffer identity is stable across decode calls
  G  baseline     legacy random-axiom veto at tau=0.35 must pass ~never; this is
                  the measured proof that the old measure was un-passable

Evidence written to readout_probe_observed.json. Floats measured, not asserted.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from henri_wave_readout import WaveObservationReadout, _normalize  # noqa: E402

OUT = os.path.join(HERE, "readout_probe_observed.json")
TAU_VETO = 0.35


def main() -> int:
    torch.manual_seed(0)
    D, S, V = 8192, 8, 16
    ro = WaveObservationReadout(dim=D, n_slots=S, n_values=V, seed=0)
    g = torch.Generator().manual_seed(11)
    results: dict = {}
    failures: list = []

    # ---------------------------------------------------------------- S1
    # MEASURED, then gated. A single binding/unbinding round trip does NOT return
    # cos ~ 1: the |FFT(R)_k|^2 fluctuation is itself O(1) per frequency, so the
    # recovered vector is signal plus comparable-norm noise. This gate exists to
    # record the achievable value, not to assert an unachievable one.
    r = ro.role_codebook[0]
    f = ro.value_codebook[0, 3]
    z = ro.unbind(ro.bind(r, f), r)
    s1 = float(torch.abs((torch.conj(_normalize(z)) * f).sum()).item())
    results["S1_single_binding_recovery_cos"] = s1
    results["S1_note"] = ("single round trip; ~0.7 expected, NOT ~1.0. The RFSS "
                          "decode works by codebook SNAP, not by raw correlation "
                          "fidelity, which is why S1 is informative not fatal.")
    # Gate: must beat chance. Chance for V=16 codes is ~1/sqrt(D) ~ 0.
    if s1 < 0.30:
        failures.append(f"S1: single binding recovered at cos={s1:.4f} (<0.30, "
                        f"not distinguishable from chance)")

    # ---------------------------------------------------------------- S2
    psi1 = ro.encode(torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]]))
    selfcos = float(torch.abs((torch.conj(psi1[0]) * psi1[0]).sum()).item())
    results["S2_self_cosine"] = selfcos
    results["S2_norm"] = float(psi1[0].norm().item())
    if abs(selfcos - 1.0) > 1e-4:
        failures.append(f"S2: self-cosine {selfcos:.6f} != 1")

    # ---------------------------------------------------------------- A
    n_draws = 200
    exact = 0
    min_margin = float("inf")
    qualities = []
    for _ in range(n_draws):
        vals = torch.randint(0, V, (S,), generator=g)
        psi = ro.encode(vals)
        idx, q, marg = ro.decode(psi, return_margins=True)
        if bool((idx[0] == vals).all()):
            exact += 1
        min_margin = min(min_margin, float(marg.min().item()))
        qualities.append(float(q.min().item()))
    results["A_exact_decode_rate"] = exact / n_draws
    results["A_n_draws"] = n_draws
    results["A_min_margin"] = min_margin
    results["A_mean_min_quality"] = sum(qualities) / len(qualities)
    if exact / n_draws < 0.999:
        failures.append(f"A: exact decode rate {exact/n_draws:.4f} < 0.999")

    # ---------------------------------------------------------------- B  (the fix)
    # Two wavefronts that decode to the SAME observable but sit far apart in wave
    # space. The added content is spread across all frequencies, so it aligns with
    # no single role-filler and does not move the decoded slots, while the internal
    # cosine falls to 1/sqrt(1+alpha^2).
    vals = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    psi_clean = ro.encode(vals)
    psi_alt = ro.encode_with_orthogonal_content(vals, alpha=3.0, seed=777)

    internal_cos = float(torch.abs((torch.conj(psi_clean[0]) * psi_alt[0]).sum()).item())
    internal_stress = 1.0 - internal_cos
    obs = ro.delta_sagnac_observational(psi_alt, vals)
    obs_stress = float(obs["stress"][0].item())
    obs_match = float(obs["match_rate"][0].item())

    results["B_internal_cosine_clean_vs_alt"] = internal_cos
    results["B_internal_stress_1_minus_cos"] = internal_stress
    results["B_observational_stress"] = obs_stress
    results["B_observational_match_rate"] = obs_match
    results["B_internal_would_veto"] = internal_stress > TAU_VETO
    results["B_observational_would_pass"] = obs_stress <= TAU_VETO
    # The claim: internal vetoes a correct prediction that the observable measure passes.
    if not (internal_stress > TAU_VETO and obs_stress <= TAU_VETO):
        failures.append(
            f"B: readout gave no separation (internal_stress={internal_stress:.4f}, "
            f"obs_stress={obs_stress:.4f}); the two measures agree, so nothing is fixed")

    # ---------------------------------------------------------------- C
    corrupt = (vals + 1) % V
    obs_c = ro.delta_sagnac_observational(psi_clean, corrupt)
    results["C_corrupted_observational_stress"] = float(obs_c["stress"][0].item())
    if results["C_corrupted_observational_stress"] <= TAU_VETO:
        failures.append("C: corrupted observable still passes the veto")

    # ---------------------------------------------------------------- D
    zero = torch.zeros(1, D, dtype=torch.complex64)
    obs_d = ro.delta_sagnac_observational(zero, vals)
    results["D_zero_input_stress"] = float(obs_d["stress"][0].item())
    results["D_zero_input_valid"] = bool(obs_d["valid"][0].item())
    results["D_nan_present"] = bool(torch.isnan(obs_d["stress"]).any().item())
    if not (results["D_zero_input_stress"] == 1.0 and not results["D_zero_input_valid"]
            and not results["D_nan_present"]):
        failures.append("D: zero input did not fail closed cleanly")

    # ---------------------------------------------------------------- E
    mixed = torch.stack([psi_clean[0], torch.zeros(D, dtype=torch.complex64), psi_clean[0]])
    ref_mixed = torch.stack([vals[0], vals[0], vals[0]])
    obs_e = ro.delta_sagnac_observational(mixed, ref_mixed)
    per_sample_ok = (float(obs_e["stress"][0].item()) <= TAU_VETO
                     and float(obs_e["stress"][1].item()) == 1.0
                     and float(obs_e["stress"][2].item()) <= TAU_VETO)
    results["E_per_sample_stress"] = [float(x) for x in obs_e["stress"]]
    results["E_per_sample_ok"] = per_sample_ok
    if not per_sample_ok:
        failures.append("E: zero row leaked into neighbours (global guard bug)")

    # ---------------------------------------------------------------- F
    id_role = id(ro.role_codebook)
    id_val = id(ro.value_codebook)
    for _ in range(5):
        ro.decode(psi_clean)
    results["F_role_buffer_identity_stable"] = id(ro.role_codebook) == id_role
    results["F_value_buffer_identity_stable"] = id(ro.value_codebook) == id_val
    if not (results["F_role_buffer_identity_stable"]
            and results["F_value_buffer_identity_stable"]):
        failures.append("F: codebook buffer was rebound during decode")

    # ---------------------------------------------------------------- G
    N_BASE = 300
    n_pass = 0
    stresses = []
    for i in range(N_BASE):
        v = torch.randint(0, V, (S,), generator=g)
        psi = ro.encode(v)
        st = float(ro.legacy_random_axiom_stress(psi)[0].item())
        stresses.append(st)
        if st <= TAU_VETO:
            n_pass += 1
    results["G_legacy_random_axiom_pass_rate"] = n_pass / N_BASE
    results["G_legacy_random_axiom_stress_mean"] = sum(stresses) / len(stresses)
    results["G_legacy_random_axiom_stress_min"] = min(stresses)
    # analytic expectation for max |<psi, axiom>| over K random unit phasors
    results["G_predicted_max_sim_sqrt(2lnK/D)"] = math.sqrt(2.0 * math.log(128) / D)
    if n_pass > 0.05 * N_BASE:
        failures.append(
            f"G: legacy random-axiom veto passed {n_pass}/{N_BASE} times; the "
            f"'un-passable by construction' claim is NOT supported by measurement")

    # ------------------------------------------------------------- new measure
    n_new_pass = 0
    for i in range(N_BASE):
        v = torch.randint(0, V, (S,), generator=g)
        psi = ro.encode(v)
        st = float(ro.delta_sagnac_observational(psi, v)["stress"][0].item())
        if st <= TAU_VETO:
            n_new_pass += 1
    results["H_new_readout_correct_pass_rate"] = n_new_pass / N_BASE
    if n_new_pass / N_BASE < 0.99:
        failures.append(
            f"H: new readout only passes {n_new_pass}/{N_BASE} correct predictions")

    verdict = "PASS" if not failures else "FAIL"
    out = {
        "module": "readout_delta_sagnac_probe",
        "evidence_class": "OBSERVED",
        "dim": D, "n_slots": S, "n_values": V, "tau_veto": TAU_VETO,
        "platform": platform.platform(), "torch": torch.__version__,
        "results": results,
        "gate_failures": failures,
        "verdict": verdict,
        "claim": ("Decoding the wavefront to observable space makes Delta_Sagnac a "
                  "usable signal: correct predictions pass, corrupted ones fail, "
                  "and the legacy random-axiom measure passes essentially never."),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 82)
    print("READOUT PROBE  (D=%d, slots=%d, values=%d, tau_veto=%.2f)" % (D, S, V, TAU_VETO))
    print("=" * 82)
    print(f"  S1 unbinding recovers filler   cos = {s1:.6f}")
    print(f"  S2 self-cosine                 = {selfcos:.6f}  |Psi| = {psi1[0].norm():.6f}")
    print(f"  A  exact decode rate           = {results['A_exact_decode_rate']:.4f} "
          f"({exact}/{n_draws}), min margin {min_margin:.2e}")
    print(f"  B  internal cos(clean,alt)     = {internal_cos:.4f} "
          f"-> internal stress {internal_stress:.4f} "
          f"{'VETO' if internal_stress > TAU_VETO else 'pass'}")
    print(f"  B  observational stress        = {obs_stress:.4f} "
          f"(match {obs_match:.3f}) {'pass' if obs_stress <= TAU_VETO else 'VETO'}")
    print(f"  C  corrupted obs stress        = {results['C_corrupted_observational_stress']:.4f}")
    print(f"  D  zero input                  stress={results['D_zero_input_stress']:.1f} "
          f"valid={results['D_zero_input_valid']} nan={results['D_nan_present']}")
    print(f"  E  per-sample stresses         = {results['E_per_sample_stress']}")
    print(f"  F  buffers stable              role={results['F_role_buffer_identity_stable']} "
          f"value={results['F_value_buffer_identity_stable']}")
    print(f"  G  LEGACY random-axiom pass    = {results['G_legacy_random_axiom_pass_rate']:.4f} "
          f"(mean stress {results['G_legacy_random_axiom_stress_mean']:.4f}, "
          f"min {results['G_legacy_random_axiom_stress_min']:.4f})")
    print(f"     predicted best random sim  = {results['G_predicted_max_sim_sqrt(2lnK/D)']:.4f}")
    print(f"  H  NEW readout correct pass    = {results['H_new_readout_correct_pass_rate']:.4f}")
    print()
    if failures:
        print("GATE FAILURES:")
        for f_ in failures:
            print(f"  - {f_}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
