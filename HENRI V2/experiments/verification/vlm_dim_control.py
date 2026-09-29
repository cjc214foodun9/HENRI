"""DIMENSION-MATCHED CONTROL for the stage 4 -> 5 drop. Cheap; runs on CPU.

WHY THIS EXISTS
===============
vlm_stage_localise_receipt.json (my real run, n=256, RTX 5090) measured:

    to_phase (65536 dims)      r2_std 0.6037
    accumulator (65536 dims)   r2_std 0.3961     <- a 34% relative drop
    psi normalised             r2_std 0.3949

and my own receipt flagged this gap in the same run:
`capacity_explains_drop` compares stage 3 only against the 512-dim control. There
is NO dimension-matched control at 65536 dims. Without one, two rival explanations
predict the same number:

  (a) the BINDING/accumulator step really costs readability, or
  (b) the readout simply behaves differently on a 65536-dim, unit-norm, mostly
      uninformative vector (the scalar mean-of-std standardisation is dominated by
      dimensions that carry no state).

This test separates them using the REAL cached Qwen3-VL features from that run.
It trains nothing.

PRE-REGISTERED DECISION RULE (fixed before running)
  Let X = r2_std of a RANDOM 512 -> 65536 expansion, unit-normalised, same readout.
    X <= 0.50  -> GENERIC: geometry+readout explains the drop; binding exonerated.
    X >= 0.55  -> the added superposition itself costs readability; the beta sweep
                  then shows how much term mass reproduces the measured 0.3949.
  A third outcome, 0.50 < X < 0.55, is reported as UNRESOLVED rather than assigned.
"""

import argparse
import json
import os
import sys
import time

import torch


def r2_std(x, y, frac=0.25, lam_rel=1e-3):
    """Scale-aware held-out R^2 -- the SAME readout used by every prior probe."""
    n = x.shape[0]
    nte = max(1, int(n * frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    te, tr = perm[:nte], perm[nte:]
    xtr, xte = x[tr].double(), x[te].double()
    ytr, yte = y[tr].double(), y[te].double()
    xm, ym = xtr.mean(0, keepdim=True), ytr.mean(0, keepdim=True)
    s = xtr.std(0, keepdim=True).mean().clamp(min=1e-12)
    xtr, xte = (xtr - xm) / s, (xte - xm) / s
    ytr, yte = ytr - ym, yte - ym
    k = xtr @ xtr.T
    a = torch.linalg.solve(k + lam_rel * n * torch.eye(k.shape[0], dtype=k.dtype), ytr)
    pred = (xte @ xtr.T) @ a
    res = ((yte - pred) ** 2).sum(0)
    tot = (yte ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - res / tot).mean())


def unit(h: torch.Tensor) -> torch.Tensor:
    return h / h.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/root/henri/vlm_feat_cache.pt")
    ap.add_argument("--out", default="/root/henri/vlm_dim_control.json")
    ap.add_argument("--real-psi-r2", type=float, default=0.3949,
                    help="measured in vlm_stage_localise_receipt.json")
    ap.add_argument("--binding-ratio", type=float, default=0.03604,
                    help="measured binding_ratio from the same receipt")
    a = ap.parse_args()

    R = {"schema": "henri.vlm_dim_control/1",
         "when": time.strftime("%Y-%m-%d %H:%M:%S"),
         "cache": a.cache,
         "readout": "r2_std (scale-aware), held-out 25%, lam_rel=1e-3",
         "real_psi_r2_from_stage_probe": a.real_psi_r2,
         "real_binding_ratio": a.binding_ratio,
         "pre_registered": {
             "X_le_0.50": "generic geometry+readout explains the drop; binding exonerated",
             "X_ge_0.55": "the added superposition itself costs readability",
             "0.50_lt_X_lt_0.55": "UNRESOLVED -- reported, not assigned"}}

    if not os.path.isfile(a.cache):
        R["fatal"] = f"cache not found: {a.cache}"
        json.dump(R, open(a.out, "w"), indent=2)
        print(json.dumps(R, indent=2))
        return 4

    C = torch.load(a.cache, map_location="cpu")
    feats = C["feats"].float()          # [n, n_patches, d_vit] fp16 -> fp32
    S = C["states"].float()             # [n, 4]
    n, NP, DV = feats.shape
    R["cache_meta"] = {"n": int(n), "n_patches": int(NP), "d_vit": int(DV),
                       "env_class": C.get("env_class"),
                       "bytes": os.path.getsize(a.cache)}
    R["cache_size_consistency"] = {
        "n_x_np_x_dv_x_2_fp16": int(n * NP * DV * 2),
        "matches_measured_cache": int(n * NP * DV * 2) == os.path.getsize(a.cache)}

    pooled = feats.mean(dim=1)                                  # [n, 1024]
    R["r2_pooled_1024_reference"] = round(r2_std(pooled, S), 4)

    g = torch.Generator().manual_seed(1234)
    # ---- control A: random 1024 -> 512 (matches to_pool width) --------------
    Wa = torch.randn(int(pooled.shape[-1]), 512, generator=g) / (512 ** 0.5)
    zA = pooled @ Wa
    R["r2_random_1024_to_512"] = round(r2_std(zA, S), 4)

    # ---- control B: random 512 -> 65536 (matches to_phase width) ------------
    Wb = torch.randn(512, 65536, generator=g) / (65536 ** 0.5)
    h = zA @ Wb                                                 # [n, 65536]
    R["r2_random_65536_raw"] = round(r2_std(h, S), 4)
    h = unit(h)
    X = r2_std(h, S)
    R["r2_random_65536_unitnorm"] = round(X, 4)

    # ---- control C: add a random binding-like superposition of matched mass --
    b = torch.randn(n, 65536, generator=g)
    b = b - b.mean(dim=0, keepdim=True)
    b = unit(b)
    sweep = {}
    for beta in (a.binding_ratio, 0.1, 0.3, 1.0):
        acc = unit(h + beta * b)
        sweep[f"beta_{beta}"] = round(r2_std(acc, S), 4)
    R["r2_unitnorm_plus_random_superposition"] = sweep
    R["r2_at_measured_binding_ratio"] = sweep[f"beta_{a.binding_ratio}"]

    # ---- verdicts (pre-registered rule) ------------------------------------
    R["verdicts"] = {
        "X_random_65536_unitnorm": round(X, 4),
        "GENERIC_geometry_explains_drop": bool(X <= 0.50),
        "ADDED_SUPERPOSITION_COSTS": bool(X >= 0.55),
        "UNRESOLVED": bool(0.50 < X < 0.55),
        "real_psi_r2_reproduced_by_random_geometry":
            bool(abs(X - a.real_psi_r2) <= 0.10),
    }
    json.dump(R, open(a.out, "w"), indent=2)
    print(json.dumps(R, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
