#!/usr/bin/env python3
"""Can an OBSERVABLE metric discriminate on this encoder? Measure before wiring.

WHY THIS GATE EXISTS BEFORE ANY WIRING
    The approved item is "wire delta_sagnac_observational into search()". Before
    wiring a veto, the veto must be shown to SEPARATE cases that should pass from
    cases that should fail. If it cannot, wiring it produces a gate that looks green
    and decides nothing -- the pattern this project has already paid for twice
    (r = 0.707 that never opened; tau = 0.35 on a match-rate metric).

WHY DISCRIMINATION IS IN DOUBT
    `HENRIVisionEncoder.encode_grid` is NOT an invertible pixel code. Reading it:
      * values are CLAMPED to 0..15 (line 112), and ARC uses 0..9, so adjacent
        symbols can collide,
      * a ParityContourMask is applied to enclosed regions (lines 120-124),
      * ConnectedComponentSegmenter introduces topology-dependent structure.
    It is a topological superposition. Two different grids can therefore share a
    large fraction of their wave structure, which would make an observable
    agreement metric weakly informative no matter how the threshold is set.

WHAT IS MEASURED
    For a base grid G and perturbations of increasing severity, and for an unrelated
    random grid, compute
      * the waveform-cosine stress (the EXISTING metric, now scale-corrected)
      * the quantized-bin agreement stress at several k_bins (a candidate OBSERVABLE
        metric), with its exact binomial null q_99 and derived tau
    and report the SEPARATION between "should pass" and "should fail" for each.

DECISION RULE, PRE-REGISTERED
    Wire the observable channel as a veto ONLY IF, at the deployment k_bins, its
    separation exceeds the waveform cosine's. Otherwise report that the encoder
    limits observable discrimination, and do NOT wire a gate that cannot decide.
"""
from __future__ import annotations

import json
import math
import os
import sys
from math import comb

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(HERE, "observable_discrimination_observed.json")
ALPHA = 0.01


def binom_q(n: int, p: float, alpha: float) -> int:
    pmf = [comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in range(n + 1)]
    s = sum(pmf)
    pmf = [x / s for x in pmf] if s else pmf
    cum = 0.0
    for k in range(n + 1):
        cum += pmf[k]
        if cum >= 1.0 - alpha:
            return k
    return n


def quantize(w: torch.Tensor, k_bins: int) -> torch.Tensor:
    x = torch.clamp(w.flatten().to(torch.float32), -1.0, 1.0)
    return torch.floor((x + 1.0) / 2.0 * (k_bins - 1)).to(torch.long)


def bin_stress(a: torch.Tensor, b: torch.Tensor, k_bins: int) -> float:
    qa, qb = quantize(a, k_bins), quantize(b, k_bins)
    return float(1.0 - (qa == qb).float().mean().item())


def make_base(H: int = 8, W: int = 8, seed: int = 0) -> np.ndarray:
    g = np.random.default_rng(seed)
    grid = g.integers(0, 10, size=(H, W))
    # Give it ARC-like structure: a distinguishable object on a background.
    grid[:, :] = 0
    grid[1:4, 1:4] = 3
    grid[5:7, 5:7] = 7
    return grid


def perturb(grid: np.ndarray, n_cells: int, seed: int) -> np.ndarray:
    g = np.random.default_rng(seed)
    out = grid.copy()
    idx = g.choice(out.size, size=n_cells, replace=False)
    for i in idx:
        r, c = divmod(int(i), out.shape[1])
        out[r, c] = int(g.integers(0, 10))
    return out


def main() -> int:
    from henri_vision_encoder import HENRIVisionEncoder
    from sagnac_mcts_planner import SagnacMCTSPlanner

    D_MODEL = 1024
    enc = HENRIVisionEncoder(d_model=D_MODEL, k_blocks=128, device="cpu")
    planner = SagnacMCTSPlanner(d_model=D_MODEL, k_blocks=128, tau_veto=0.35,
                                device="cpu")

    base = make_base()
    g = np.random.default_rng(999)
    unrelated = g.integers(0, 10, size=(8, 8))

    variants = {
        "identical": base,
        "1_cell_changed": perturb(base, 1, 1),
        "2_cells_changed": perturb(base, 2, 2),
        "4_cells_changed": perturb(base, 4, 3),
        "8_cells_changed": perturb(base, 8, 4),
        "unrelated_random": unrelated,
    }

    base_wave = enc.encode_grid(base)
    rows = []
    for name, grid in variants.items():
        w = enc.encode_grid(grid)
        # Existing metric: waveform cosine, scale-corrected.
        cos_stress = float(1.0 - enc.compute_sagnac_similarity(base_wave, w))
        row = {"variant": name, "wave_cosine_stress": round(cos_stress, 6)}
        for k in (11, 16, 64):
            row[f"bin_stress_k{k}"] = round(bin_stress(base_wave, w, k), 6)
        rows.append(row)

    # Nulls and derived thresholds per k_bins, at the actual wave dimension.
    D = int(base_wave.numel())
    nulls = {}
    for k in (11, 16, 64):
        p = 1.0 / k
        q99 = binom_q(D, p, ALPHA)
        tau = 1.0 - q99 / D
        nulls[f"k{k}"] = {
            "dim": D, "null_p": round(p, 6),
            "null_expected_match_rate": round(p, 6),
            "q99_matched_bins": q99, "tau_derived": round(tau, 6),
            "inherited_0.35_requires_match_rate": 0.65,
        }

    # SEPARATION analysis. "Should pass" = identical. "Should fail" = unrelated.
    sep = {}
    id_row = next(r for r in rows if r["variant"] == "identical")
    un_row = next(r for r in rows if r["variant"] == "unrelated_random")
    for key in ("wave_cosine_stress", "bin_stress_k11", "bin_stress_k16",
                "bin_stress_k64"):
        sep[key] = {
            "identical": id_row[key],
            "unrelated": un_row[key],
            "separation": round(un_row[key] - id_row[key], 6),
        }
    # Monotonicity: does stress increase with damage?
    mono = {}
    order = ["identical", "1_cell_changed", "2_cells_changed", "4_cells_changed",
             "8_cells_changed", "unrelated_random"]
    for key in ("wave_cosine_stress", "bin_stress_k16"):
        vals = [next(r for r in rows if r["variant"] == v)[key] for v in order]
        mono[key] = {
            "values": vals,
            "monotone_nondecreasing": all(vals[i] <= vals[i + 1] + 1e-9
                                          for i in range(len(vals) - 1)),
        }

    best_bin_sep = max(sep[k]["separation"] for k in sep if k.startswith("bin_stress"))
    wave_sep = sep["wave_cosine_stress"]["separation"]
    observable_wins = best_bin_sep > wave_sep

    out = {
        "module": "observable_discrimination",
        "evidence_class": "OBSERVED",
        "dim": D, "alpha": ALPHA, "grid": "8x8",
        "rows": rows, "nulls_and_derived_tau": nulls,
        "separation": sep, "monotonicity": mono,
        "best_bin_separation": round(best_bin_sep, 6),
        "wave_cosine_separation": round(wave_sep, 6),
        "observable_metric_more_discriminative": bool(observable_wins),
        "decision": (
            "WIRE the observable veto" if observable_wins else
            "DO NOT WIRE as a veto: the observable metric does not separate cases "
            "better than the existing scale-corrected waveform cosine. Report that "
            "encode_grid's topological superposition limits observable "
            "discrimination, and that the ingress encoder is the enabling change."),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 92)
    print(f"OBSERVABLE DISCRIMINATION ON encode_grid  (d_model={D_MODEL}, dim={D})")
    print("=" * 92)
    print(f"{'variant':>18} {'wave_cos':>10} {'bin_k11':>10} {'bin_k16':>10} "
          f"{'bin_k64':>10}")
    for r in rows:
        print(f"{r['variant']:>18} {r['wave_cosine_stress']:>10.4f} "
              f"{r['bin_stress_k11']:>10.4f} {r['bin_stress_k16']:>10.4f} "
              f"{r['bin_stress_k64']:>10.4f}")
    print()
    print("DERIVED THRESHOLDS (exact binomial null at this dim):")
    for k in ("k11", "k16", "k64"):
        n = nulls[k]
        print(f"  k_bins={k[1:]:>3}  null match rate {n['null_expected_match_rate']:.4f}  "
              f"q99 matched {n['q99_matched_bins']:>4}/{D}  ->  tau = {n['tau_derived']:.4f}"
              f"   (inherited 0.35 needs match 0.65)")
    print()
    print("SEPARATION (identical vs unrelated, higher = more discriminative):")
    for key, v in sep.items():
        print(f"  {key:<20} identical {v['identical']:.4f}  unrelated "
              f"{v['unrelated']:.4f}  separation {v['separation']:.4f}")
    print()
    print("MONOTONICITY under damage:")
    for key, v in mono.items():
        print(f"  {key:<20} {[round(x,3) for x in v['values']]}  "
              f"monotone={v['monotone_nondecreasing']}")
    print()
    print(f"observable metric more discriminative than waveform cosine: {observable_wins}")
    print(f"DECISION: {out['decision']}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
