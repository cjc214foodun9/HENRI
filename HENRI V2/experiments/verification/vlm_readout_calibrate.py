"""CALIBRATE the readability readout, then re-read psi on the real geometry.

WHY THIS EXISTS
===============
vlm_readout_remeasure_receipt.json (commit 02887c7) measured, on REAL Qwen3-VL
features, that BOTH readouts fail their permutation null:

    input_mean_pooled_1024   r2_fixed 0.7139   null_fixed -0.7743
    psi_65536_unitnorm       r2_fixed 0.6900   null_fixed -0.7979
    binding_term_65536       r2_fixed 0.6657   null_fixed -0.7184
    verdict: instrument_validated_on_psi = false

So no psi claim was permitted (reported BLOCKED). henri_readout.self_test PASSED
its nulls only because it used NON-COLLINEAR synthetic data: the readout's validity
is DATA-DEPENDENT, and it was never validated on the real geometry until then.

MECHANISM (hypothesis, tested here): per-dimension standardisation divides each
coordinate by its own residual std. On a near-collinear cloud (mean off-diag cos
~0.999999) the residual signal in most coordinates is tiny and noise-dominated, so
those coordinates are amplified enormously. The ridge then fits that amplified
noise in-sample and predicts badly out-of-sample -> a strongly NEGATIVE null.

THE TEST -- a regularisation sweep, the standard way to calibrate a ridge readout
  For each lam_frac in a grid, for each representation, measure:
      obs  = r2_ridge(x, S,             lam_frac)
      null = r2_ridge(x, S[perm_k],     lam_frac)   for K permutations (same split)
  The readout is TRUSTWORTHY at the smallest lam_frac where |mean(null)| < 0.10 for
  EVERY representation (including the pure-noise control). Read `obs` THERE.

PRE-REGISTERED (fixed before running; not adjusted afterwards)
  null_pass            : |mean(null)| < 0.10 across all reps at that lam_frac
  psi_readable         : obs(psi)  >= 0.50 at the calibrated lam_frac
  pooled_readable      : obs(pooled) >= 0.50 at the calibrated lam_frac
  calibrated_drop      : pooled_obs - psi_obs  (the readability the bridge costs)
  noise_control_is_zero: obs(pure_noise) <= 0.20 at the calibrated lam_frac
  IF no lam_frac passes: report BLOCKED again, with the sweep as evidence.

Pure-noise rows are included as the control that makes the sweep meaningful: a
lam_frac that makes NOISE readable is over-shrunk or broken, not calibrated.
"""

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", ".."))
if V2 not in sys.path:
    sys.path.insert(0, V2)

from henri_readout import r2_ridge, offdiag_cos, participation_ratio  # noqa: E402

D_MODEL = 65536
LAM_GRID = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
K_PERM = 8
NULL_TOL = 0.10


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/root/henri/vlm_feat_cache.pt")
    ap.add_argument("--out", default="/root/henri/vlm_readout_calibrated.json")
    ap.add_argument("--d-pool", type=int, default=512)
    a = ap.parse_args()

    R = {"schema": "henri.vlm_readout_calibrated/1",
         "when": time.strftime("%Y-%m-%d %H:%M:%S"),
         "purpose": ("sweep ridge regularisation until the permutation null passes, "
                     "then read psi readability at the calibrated setting"),
         "lam_grid": LAM_GRID, "k_permutations": K_PERM, "null_tolerance": NULL_TOL,
         "pre_registered": {
             "null_pass": "|mean(null)| < 0.10 across ALL reps",
             "psi_readable": "obs(psi) >= 0.50 at the calibrated lam_frac",
             "pooled_readable": "obs(pooled) >= 0.50 at the calibrated lam_frac",
             "noise_control_is_zero": "obs(pure_noise) <= 0.20",
             "no_lam_passes": "report BLOCKED with the sweep as evidence"}}

    if not os.path.isfile(a.cache):
        R["fatal"] = f"cache not found: {a.cache}"
        json.dump(R, open(a.out, "w"), indent=2)
        print(json.dumps(R, indent=2))
        return 4

    C = torch.load(a.cache, map_location="cpu")
    feats = C["feats"].float()
    S = C["states"].float()
    n, NP, DV = feats.shape
    R["cache"] = {"n": int(n), "n_patches": int(NP), "d_vit": int(DV),
                  "bytes": os.path.getsize(a.cache),
                  "fp16_expectation": int(n * NP * DV * 2)}

    import henri_vla_vlm_bridge as B
    torch.manual_seed(77)
    bridge = B.VLMPhaseBridge(d_vit=DV, n_patches=NP, d_model=D_MODEL,
                              d_pool=a.d_pool)
    bridge.eval()
    with torch.no_grad():
        psi = bridge(feats).float()
        pooled_attn = bridge.pool(feats)
        z = bridge.to_pool(pooled_attn)
        p = bridge.to_phase(z) + bridge.phase_offset
        bsum = bridge.binding_term(feats) if bridge.binding_active else None
        acc = (bsum + p) if bsum is not None else p
    R["bridge"] = {"binding_active": bool(bridge.binding_active),
                   "binding_ratio": round(float(bridge.binding_ratio), 6)}

    reps = {
        "input_mean_pooled_1024": feats.mean(dim=1),
        "attention_pooled_1024": pooled_attn,
        "to_pool_512": z,
        "to_phase_65536": p,
        "accumulator_65536": acc,
        "psi_65536_unitnorm": psi,
        "CONTROL_pure_noise_65536": torch.randn(
            n, 65536, generator=torch.Generator().manual_seed(9)),
    }
    if bsum is not None:
        reps["binding_term_65536"] = bsum

    for name, x in reps.items():
        R.setdefault("geometry", {})[name] = {
            "d": int(x.reshape(x.shape[0], -1).shape[-1]),
            "cos": round(offdiag_cos(x), 8),
            "pr": round(participation_ratio(x), 2)}

    perms = [torch.randperm(n, generator=torch.Generator().manual_seed(100 + k))
             for k in range(K_PERM)]

    sweep = {}
    for lam in LAM_GRID:
        row = {}
        for name, x in reps.items():
            obs = r2_ridge(x, S, lam_frac=lam)
            nulls = [r2_ridge(x, S[pi], lam_frac=lam) for pi in perms]
            mn = sum(nulls) / len(nulls)
            var = sum((v - mn) ** 2 for v in nulls) / max(len(nulls) - 1, 1)
            sd = var ** 0.5
            row[name] = {"obs": round(obs, 4),
                         "null_mean": round(mn, 4),
                         "null_sd": round(sd, 4),
                         "calibrated": round((obs - mn) / (1.0 - mn), 4)
                         if mn < 1.0 else None,
                         "z": round((obs - mn) / sd, 2) if sd > 1e-9 else None}
        worst = max(abs(v["null_mean"]) for v in row.values())
        row["_worst_abs_null_mean"] = round(worst, 4)
        row["_null_passes"] = bool(worst < NULL_TOL)
        sweep[f"lam_frac_{lam:g}"] = row
        print(f"[lam {lam:g}] worst|null|={worst:.4f} passes={row['_null_passes']} "
              f"psi_obs={row['psi_65536_unitnorm']['obs']} "
              f"pooled_obs={row['input_mean_pooled_1024']['obs']}", flush=True)

    R["sweep"] = sweep

    passing = [k for k, v in sweep.items() if v["_null_passes"]]
    R["lam_fracs_passing_null"] = passing
    if passing:
        chosen = min(passing, key=lambda k: float(k.split("_")[-1]))
        c = sweep[chosen]
        R["chosen_lam_frac"] = chosen
        psi_o = c["psi_65536_unitnorm"]["obs"]
        pool_o = c["input_mean_pooled_1024"]["obs"]
        noise_o = c["CONTROL_pure_noise_65536"]["obs"]
        R["verdicts"] = {
            "null_pass": True,
            "chosen": chosen,
            "pooled_obs": pool_o, "psi_obs": psi_o, "noise_obs": noise_o,
            "psi_readable": bool(psi_o >= 0.50),
            "pooled_readable": bool(pool_o >= 0.50),
            "noise_control_is_zero": bool(noise_o <= 0.20),
            "calibrated_drop_pooled_minus_psi": round(pool_o - psi_o, 4),
            "psi_calibrated": c["psi_65536_unitnorm"]["calibrated"],
            "pooled_calibrated": c["input_mean_pooled_1024"]["calibrated"],
        }
        v = R["verdicts"]
        R["conclusion"] = (
            f"Readout CALIBRATED at {chosen}. On the real geometry psi is "
            f"{'READABLE' if v['psi_readable'] else 'NOT readable'} "
            f"(obs {psi_o}) vs input {pool_o}; the bridge costs "
            f"{v['calibrated_drop_pooled_minus_psi']} readability. "
            + ("Noise control clean." if v["noise_control_is_zero"]
               else "WARNING: noise control not clean -- sweep not trustworthy."))
    else:
        R["verdicts"] = {"null_pass": False,
                         "no_lam_passes": True,
                         "best_worst_null": min(
                             v["_worst_abs_null_mean"] for v in sweep.values())}
        R["conclusion"] = (
            "NO lam_frac in the grid makes the permutation null pass for every "
            "representation, so this readout cannot be validated on this geometry. "
            "Any psi readability claim remains BLOCKED. The sweep is the evidence; "
            "the next instrument to try is a rank-truncated (PCA) ridge or PLS, "
            "which does not rely on per-dimension standardisation of a near-collinear "
            "cloud.")

    json.dump(R, open(a.out, "w"), indent=2)
    print("\n" + json.dumps({"chosen_lam_frac": R.get("chosen_lam_frac"),
                             "verdicts": R["verdicts"],
                             "conclusion": R["conclusion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
