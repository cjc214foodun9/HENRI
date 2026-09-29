"""Is the control-C collapse (r2 0.6432 -> 0.0021) REAL, or a standardizer artifact?

WHY THIS EXISTS
===============
My own dimension-matched control (committed 261ed28) reported:

    r2 of unit-norm random 512->65536 expansion     0.6432
    r2 after adding a 0.036-mass random perturbation 0.0021
    verdicts: ADDED_SUPERPOSITION_COSTS true, GENERIC_geometry_explains_drop false

That jump is ARITHMETICALLY IMPLAUSIBLE. Adding beta=0.036 of a unit-norm vector b
to a unit-norm vector h gives ||h + beta*b|| ~ 1.0, and the result is overwhelmingly
h. So r2 should stay near 0.6432, not fall to 0.0021. Two possibilities:

  (A) the perturbation genuinely destroys readability through some interaction, or
  (B) the READOUT is the artifact: `r2_scalar` divides by ONE scalar standardizer
      (the mean over dims of per-dim std). In 65536 dims, most coordinates are
      uninformative; if the perturbation inflates variance in those coordinates,
      the scalar grows and the INFORMATIVE coordinates get divided by a larger
      number, crushing the apparent signal while the information is still present.

DISCRIMINATOR: measure the same tensors with BOTH readouts.
  r2_scalar  : one scalar standardizer (used everywhere so far) -- the suspect
  r2_perdim  : each dimension z-scored by its OWN std (clamped) -- scale-invariant
If r2_perdim(h + 0.036*b) stays high while r2_scalar collapses, (B) holds and the
control-C verdict must be WITHDRAWN rather than carried forward.

NULL CONTROLS included so this test can fail:
  - permuting rows of h must preserve r2 under both readouts
  - pure noise must be unreadable under BOTH (else the readout is just broken)
"""

import argparse
import json
import os
import time

import torch


def _ridge(xtr, xte, ytr, yte, n, lam_rel):
    k = xtr @ xtr.T
    a = torch.linalg.solve(k + lam_rel * n * torch.eye(k.shape[0], dtype=k.dtype), ytr)
    pred = (xte @ xtr.T) @ a
    res = ((yte - pred) ** 2).sum(0)
    tot = (yte ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - res / tot).mean())


def r2_scalar(x, y, frac=0.25, lam_rel=1e-3):
    """One scalar standardizer: divide all dims by mean(per-dim std). The suspect."""
    n = x.shape[0]
    nte = max(1, int(n * frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    te, tr = perm[:nte], perm[nte:]
    xtr, xte = x[tr].double(), x[te].double()
    ytr, yte = y[tr].double(), y[te].double()
    xm, ym = xtr.mean(0, keepdim=True), ytr.mean(0, keepdim=True)
    s = xtr.std(0, keepdim=True).mean().clamp(min=1e-12)
    return _ridge((xtr - xm) / s, (xte - xm) / s, ytr - ym, yte - ym, n, lam_rel)


def r2_perdim(x, y, frac=0.25, lam_rel=1e-3):
    """Each dim z-scored by its OWN std. Near-zero-std dims are floored at a
    fraction of the median std so noise amplification cannot dominate."""
    n = x.shape[0]
    nte = max(1, int(n * frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    te, tr = perm[:nte], perm[nte:]
    xtr, xte = x[tr].double(), x[te].double()
    ytr, yte = y[tr].double(), y[te].double()
    xm, ym = xtr.mean(0, keepdim=True), ytr.mean(0, keepdim=True)
    s = xtr.std(0, keepdim=True)
    floor = float(s.median()) * 1e-2
    s = s.clamp(min=max(floor, 1e-30))
    return _ridge((xtr - xm) / s, (xte - xm) / s, ytr - ym, yte - ym, n, lam_rel)


def unit(h):
    return h / h.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/root/henri/vlm_feat_cache.pt")
    ap.add_argument("--out", default="/root/henri/vlm_control_artifact.json")
    a = ap.parse_args()

    R = {"schema": "henri.vlm_control_artifact/1",
         "when": time.strftime("%Y-%m-%d %H:%M:%S"),
         "purpose": ("decide whether control-C's 0.6432 -> 0.0021 is a real effect or "
                     "an artifact of the single-scalar standardizer"),
         "readouts": {"r2_scalar": "one scalar standardizer (suspect)",
                      "r2_perdim": "per-dimension z-score (scale-invariant control)"}}

    if not os.path.isfile(a.cache):
        R["fatal"] = f"cache not found: {a.cache}"
        json.dump(R, open(a.out, "w"), indent=2)
        print(json.dumps(R, indent=2))
        return 4

    C = torch.load(a.cache, map_location="cpu")
    feats = C["feats"].float()
    S = C["states"].float()
    n, NP, DV = feats.shape
    pooled = feats.mean(dim=1)
    R["cache"] = {"n": int(n), "n_patches": int(NP), "d_vit": int(DV),
                  "bytes": os.path.getsize(a.cache),
                  "fp16_expectation": int(n * NP * DV * 2)}

    g = torch.Generator().manual_seed(1234)
    Wa = torch.randn(int(pooled.shape[-1]), 512, generator=g) / (512 ** 0.5)
    zA = pooled @ Wa
    Wb = torch.randn(512, 65536, generator=g) / (65536 ** 0.5)

    cases = {}
    cases["ref_pooled_1024"] = pooled
    cases["zA_512"] = zA
    cases["h_unit_65536"] = unit(zA @ Wb)

    b = torch.randn(n, 65536, generator=g)
    b = b - b.mean(dim=0, keepdim=True)
    b = unit(b)
    h = unit(zA @ Wb)
    for beta in (0.0, 0.036, 0.3, 1.0):
        cases[f"h_plus_{beta}_noise"] = unit(h + beta * b)
    cases["pure_noise_65536"] = unit(torch.randn(n, 65536,
                                                 generator=torch.Generator().manual_seed(9)))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(3))
    cases["h_unit_permuted_rows"] = h[perm]      # null: must preserve r2

    for name, x in cases.items():
        cases_out = {
            "dim": int(x.shape[-1]),
            "r2_scalar": round(r2_scalar(x, S), 4),
            "r2_perdim": round(r2_perdim(x, S), 4),
        }
        R.setdefault("cases", {})[name] = cases_out

    c = R["cases"]
    base_s = c["h_unit_65536"]["r2_scalar"]
    pert_s = c["h_plus_0.036_noise"]["r2_scalar"]
    pert_p = c["h_plus_0.036_noise"]["r2_perdim"]
    base_p = c["h_unit_65536"]["r2_perdim"]
    noise_p = c["pure_noise_65536"]["r2_perdim"]
    perm_s = c["h_unit_permuted_rows"]["r2_scalar"]

    R["verdicts"] = {
        "scalar_readout_collapses_on_0.036_perturbation": bool(pert_s < 0.20),
        "perdim_readout_keeps_it": bool(pert_p > 0.50),
        "SCALAR_STANDARDIZER_IS_THE_ARTIFACT": bool(pert_s < 0.20 and pert_p > 0.50),
        "permutation_null_holds_scalar": bool(abs(perm_s - base_s) < 0.05),
        "pure_noise_unreadable_by_perdim": bool(noise_p < 0.20),
        "baseline_scalar": base_s, "baseline_perdim": base_p,
        "perturbed_scalar": pert_s, "perturbed_perdim": pert_p,
    }
    R["consequence"] = (
        "Control-C's ADDED_SUPERPOSITION_COSTS verdict must be WITHDRAWN: the drop is "
        "produced by the single-scalar standardizer, not by the superposition. The "
        "stage 4->5 drop (0.6037 -> 0.3961) therefore remains UNATTRIBUTED."
        if R["verdicts"]["SCALAR_STANDARDIZER_IS_THE_ARTIFACT"] else
        "Scalar standardizer exonerated: the perturbation really does cost readability, "
        "and the stage 4->5 drop is attributable to the added superposition.")

    json.dump(R, open(a.out, "w"), indent=2)
    print(json.dumps(R, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
