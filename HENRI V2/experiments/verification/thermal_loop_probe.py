#!/usr/bin/env python3
"""Probe: does the Pillar 3 Sagnac-veto -> Langevin coupling hold in isolation?

This file is the artifact cited by section 7 of
`docs/directive-2026-critical-execution-record.md` ("A separate 8-claim probe
holds"). It lives in the repo so that citation is reproducible:

    python experiments/verification/thermal_loop_probe.py ; echo $?

Exit 0 = all claims hold. Exit 1 = at least one claim failed (the failing claim
numbers are printed). This is a DIAGNOSTIC, not a capability claim: nothing here
measures a task metric, and the module has 0 production importers.

CLAIMS (each is able to fail):
  1. flag OFF            -> theta returned UNCHANGED, event DISABLED, no energy
  2. VETO_UNAVAILABLE    -> quiescent; a failed measurement is NOT a veto
  3. coherent candidate  -> quiescent, zero energy injected
  4. hard veto           -> fires, energy > 0, steps = max_steps
  5. DISCRIMINATING      -> kT STRICTLY increasing in delta_axiom AND strictly
     greater than kT_base. If kT were flat in delta the loop would be a constant
     noise injector carrying no information from the veto; claim 5 is what
     separates a coupling from theatre.
  6. seed reproducibility -> same seed gives bit-identical output (atol=0)
  7. bounded             -> kT <= kT_max for extreme deltas
  8. unit norm preserved -> the creeped vector keeps ||theta||

METRIC NOTE (measured, and the reason this probe's first revision FAILED):
  `arc_sagnac_veto._sagnac_similarity` for complex waves is |mean(conj(a)*b)| --
  a MEAN over D components, not a normalised inner product. A ONE-HOT therefore
  self-compares to 1/D (D=64 -> 0.015625 -> delta 0.984 -> FIRES). Measured on
  this fixture set:
      ones vs ones       -> sim 1.0000 -> delta 0.0000 -> quiet
      ones vs rand-phase -> sim 0.0847 -> delta 0.9153 -> FIRES
  So eps_hard = 0.35 is meaningful only for near-unit-modulus waves, and BOTH the
  reference waves and the coherent candidate must be unit-modulus for claim 3 to
  be a valid test of the module rather than of the fixture.
"""
from __future__ import annotations

import os
import sys

# experiments/verification/thermal_loop_probe.py -> HENRI V2
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402

import arc_sagnac_veto as ASV  # noqa: E402
import henri_sagnac_thermal_loop as TL  # noqa: E402

D = 64


def onehot(i: int, d: int = D) -> torch.Tensor:
    v = torch.zeros(d, dtype=torch.complex64)
    v[i] = 1.0 + 0.0j
    return v


def main() -> int:
    fails = []
    print("eps_hard =", ASV.DEFAULT_EPSILON_HARD)
    print("tau_veto =", TL.DEFAULT_TAU_VETO)

    # Unit-modulus fixtures: see METRIC NOTE above.
    axiom = torch.ones(D, dtype=torch.complex64)
    world = torch.ones(D, dtype=torch.complex64)
    coherent = torch.ones(D, dtype=torch.complex64)   # matching phases -> quiet
    _g = torch.Generator().manual_seed(3)
    _th = torch.rand(D, generator=_g) * 2.0 * torch.pi
    vetoed = torch.exp(1j * _th).to(torch.complex64)  # phases disagree -> FIRES
    theta = torch.randn(D) * 0.1

    # ---- 1. flag OFF preserves the production path --------------------------
    # The flag is read at construction, so build the loop with it unset.
    os.environ.pop(TL.FLAG_ENV, None)
    loop = TL.SagnacThermalLoop()
    out, ev = loop.step(vetoed, axiom, world, theta)
    ok = (ev.fired is False) and (ev.status == "DISABLED") and torch.equal(out, theta)
    print("1 flag OFF   :", ok, "| fired=", ev.fired, "status=", ev.status,
          "unchanged=", torch.equal(out, theta))
    if not ok:
        fails.append("1")

    # ---- enable the loop for the rest --------------------------------------
    os.environ[TL.FLAG_ENV] = "1"
    print("   loop_enabled() =", TL.loop_enabled())

    # ---- 2. unavailable veto never fires -----------------------------------
    out, ev = loop.step(None, axiom, world, theta)
    ok = (ev.fired is False) and (ev.status == ASV.VETO_UNAVAILABLE)
    print("2 unavailable:", ok, "| fired=", ev.fired, "status=", ev.status)
    if not ok:
        fails.append("2")

    # ---- 3. coherent candidate injects nothing -----------------------------
    out, ev = loop.step(coherent, axiom, world, theta)
    ok = (ev.fired is False) and (ev.energy_injected == 0.0) and torch.equal(out, theta)
    print("3 coherent   :", ok, "| fired=", ev.fired, "status=", ev.status,
          "energy=", ev.energy_injected)
    if not ok:
        fails.append("3")

    # ---- 4. hard veto fires with bounded creep -----------------------------
    out, ev = loop.step(vetoed, axiom, world, theta)
    ok = (ev.fired is True) and (ev.energy_injected > 0.0) and (ev.steps == loop.cfg.max_steps)
    print("4 hard veto  :", ok, "| fired=", ev.fired, "delta=", round(ev.delta_axiom, 4),
          "kT=", round(ev.kT, 4), "steps=", ev.steps, "energy=", round(ev.energy_injected, 6))
    if not ok:
        fails.append("4")

    # ---- 5. DISCRIMINATING: kT strictly increases in delta -----------------
    cfg = TL.ThermalLoopConfig()
    ds = [0.40, 0.70, 1.00, 1.50]
    ks = [TL.kT_from_delta(d, cfg) for d in ds]
    strict = all(ks[i] < ks[i + 1] for i in range(len(ks) - 1))
    above = ks[0] > cfg.kT_base
    print("5 kT(delta)  :", strict and above, "| deltas", ds, "-> kT", [round(k, 4) for k in ks])
    if not (strict and above):
        fails.append("5")

    # ---- 6. seed reproducibility -------------------------------------------
    a = TL.SagnacThermalLoop(TL.ThermalLoopConfig(seed=7))
    b = TL.SagnacThermalLoop(TL.ThermalLoopConfig(seed=7))
    oa, _ = a.step(vetoed, axiom, world, theta)
    ob, _ = b.step(vetoed, axiom, world, theta)
    ok = torch.allclose(oa, ob, atol=0.0, rtol=0.0)
    print("6 seed repro :", ok, "| max|diff| =", float((oa - ob).abs().max()))
    if not ok:
        fails.append("6")

    # ---- 7. bounded --------------------------------------------------------
    ks_big = [TL.kT_from_delta(d, cfg) for d in (3.0, 10.0, 1e6)]
    ok = all(k <= cfg.kT_max for k in ks_big)
    print("7 bounded    :", ok, "| kT", [round(k, 4) for k in ks_big], "cap", cfg.kT_max)
    if not ok:
        fails.append("7")

    # ---- 8. unit norm preserved -------------------------------------------
    n_before = float(theta.norm())
    n_after = float(out.norm())
    ok = abs(n_after - n_before) < 1e-4
    print("8 norm       :", ok, "| before", round(n_before, 6), "after", round(n_after, 6))
    if not ok:
        fails.append("8")

    print()
    if fails:
        print("FAILED CLAIMS:", ",".join(fails))
        return 1
    print("ALL 8 CLAIMS HOLD")
    return 0


if __name__ == "__main__":
    sys.exit(main())
