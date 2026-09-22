#!/usr/bin/env python3
"""Why did the thermodynamic learner fail to descend? Find the real cause.

MEASURED FACTS (this session, all local CPU, numeric):
  * The analytic gradient is CORRECT -- matches a float64 central difference to
    3.6e-12 relative (experiments/verification/thermo_gradcheck.py).
  * The pre-registered loss nevertheless descended only ~1.9% and generation did
    not beat the zero-coupling null arm.
  * Trajectory sanity showed corr(x(t0)_visible, pattern) = -0.2862: the imposed
    pattern is BURIED IN THERMAL NOISE.

HYPOTHESIS D2 (scale defect, not algorithm defect).
    In the Langevin computer the thermal standard deviation of a unit is
        sigma = sqrt(kT / (2 J2))
    and the deterministic displacement produced by an imposed bias b is
        x* ~ b / (2 J2).
    With the source's J2 = kT = 10, 1 (his units):
        sigma = sqrt(1 / 20) = 0.2236
        b = noise_amp = 2.0  ->  x* ~ 2/20 = 0.10
    So the imprint is ~0.45 sigma: below the noise floor. The noising trajectory
    therefore does not carry the pattern, the coupling gradient is fitted to
    noise-induced correlations, and the system cannot learn the data.

    The control that makes this falsifiable: sweep noise_amp and measure the
    imprint correlation. If D2 is right, correlation must rise monotonically
    toward +1 with noise_amp, and the learned model must start beating the null
    sampler. If the correlation stays near zero no matter the amplitude, D2 is
    falsified and the defect is elsewhere.

This is a diagnostic. It is not a capability claim.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch

from henri_thermo_langevin import (LangevinComputer, LangevinConfig,
                                   make_stripe_patterns)


def imprint_correlation(cfg: LangevinConfig, n: int = 8, seed: int = 0) -> dict:
    """Mean correlation of the visible slice with the imposed pattern, at t0 and tE."""
    patterns = make_stripe_patterns(cfg.n_visible)
    torch.manual_seed(seed)
    m = LangevinComputer(cfg, seed=seed)
    c0, cE = [], []
    for k in range(n):
        pat = patterns[k % patterns.shape[0]]
        traj, _ = m.noise_trajectory(pat, generator=torch.Generator().manual_seed(seed + k))
        pn = pat / pat.norm().clamp_min(1e-9)
        v0 = traj[0, :cfg.n_visible]
        vE = traj[-1, :cfg.n_visible]
        c0.append(float((v0 / v0.norm().clamp_min(1e-9)) @ pn))
        cE.append(float((vE / vE.norm().clamp_min(1e-9)) @ pn))
    return {"corr_t0": sum(c0) / len(c0), "corr_tE": sum(cE) / len(cE),
            "amplitude_rms_t0": float(traj[0, :cfg.n_visible].norm() / (cfg.n_visible ** 0.5))}


def learn_and_score(cfg: LangevinConfig, steps: int = 150, alpha: float = 0.05,
                    seed: int = 0) -> dict:
    """Train, then compare generated-pattern overlap against the zero-coupling null."""
    patterns = make_stripe_patterns(cfg.n_visible)
    torch.manual_seed(seed)
    m = LangevinComputer(cfg, seed=seed)
    res = m.train_on_patterns(patterns, steps=steps, n_traj_per_step=2,
                              alpha=alpha, seed=seed)

    def overlap(model: LangevinComputer, n: int = 24) -> float:
        g = torch.Generator().manual_seed(seed + 777)
        pn = patterns / patterns.norm(dim=-1, keepdim=True)
        vals = []
        for _ in range(n):
            x = model.generate(generator=g)[:cfg.n_visible]
            vals.append(float(((x / x.norm().clamp_min(1e-9)) @ pn.t()).max()))
        return sum(vals) / len(vals)

    trained_ov = overlap(m)
    null_ov = overlap(LangevinComputer(cfg, seed=seed))
    return {
        "loss_first": res["loss_first"], "loss_last": res["loss_last"],
        "relative_drop": ((res["loss_first"] - res["loss_last"]) /
                          abs(res["loss_first"]) if res["loss_first"] else 0.0),
        "overlap_trained": trained_ov, "overlap_null": null_ov,
        "overlap_gain": trained_ov - null_ov,
    }


def main() -> int:
    base = dict(n_units=48, n_visible=16, t_final=0.5)   # K = 100 steps
    rows = []
    print("=== D2 TEST: imprint amplitude vs learnability ===")
    print(f"{'noise_amp':>10} {'corr_t0':>9} {'corr_tE':>9} {'rel_drop':>10} "
          f"{'ov_trained':>11} {'ov_null':>9} {'gain':>8}")
    for amp in (2.0, 6.0, 12.0, 20.0, 30.0, 45.0):
        cfg = LangevinConfig(noise_amp=amp, **base)
        imp = imprint_correlation(cfg)
        lrn = learn_and_score(cfg, steps=150, alpha=0.05, seed=0)
        rows.append({"noise_amp": amp, **imp, **lrn})
        print(f"{amp:>10.1f} {imp['corr_t0']:>+9.4f} {imp['corr_tE']:>+9.4f} "
              f"{lrn['relative_drop']:>10.5f} {lrn['overlap_trained']:>11.4f} "
              f"{lrn['overlap_null']:>9.4f} {lrn['overlap_gain']:>+8.4f}")

    sigma = (1.0 / 20.0) ** 0.5
    print(f"\n  thermal sigma = sqrt(kT/2J2) = {sigma:.4f}; "
          f"imprint x* ~ noise_amp/(2*J2) = noise_amp/20")
    for r in rows:
        print(f"  amp={r['noise_amp']:>5.1f}  x*/sigma = "
              f"{(r['noise_amp'] / 20.0) / sigma:>6.2f}  gained={r['overlap_gain'] > 0.05}")

    best = max(rows, key=lambda r: r["overlap_gain"])
    mono = all(rows[i + 1]["corr_t0"] > rows[i]["corr_t0"]
               for i in range(len(rows) - 1))
    payload = {
        "schema": "henri.thermo-imprint-sweep.v1",
        "hypothesis": "D2_SCALE_DEFECT",
        "thermal_sigma": sigma,
        "correlation_monotone_in_amplitude": mono,
        "rows": rows,
        "best": best,
        "verdict": ("D2_SUPPORTED -- imprint correlation and learned overlap both rise "
                    "with the imposed amplitude, so the earlier failure was a scale "
                    "defect and the learning rule itself works."
                    if mono and best["overlap_gain"] > 0.05 else
                    "D2_NOT_SUPPORTED -- amplitude does not recover learnability; the "
                    "defect is elsewhere and the rule must not be promoted."),
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "thermo_imprint_sweep_observed.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n{payload['verdict']}\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
