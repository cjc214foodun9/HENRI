"""RE-MEASURE psi readability with a NULL-VALIDATED readout.

WHY THIS EXISTS
===============
Every psi number in this session (0.3949 fresh, -0.5 trained, 0.6432 control) came
from a ridge readout regularised by a FIXED penalty `lam_rel * n`. My own instrument
test then measured that readout's PERMUTATION NULL -- feature row i deliberately
paired with state row j != i, which MUST score ~0 -- on near-collinear unit-norm
features (psi's actual geometry):

    h_unit_permuted_rows   r2_scalar -1.4811     r2_perdim -1.4822

A strongly negative held-out R^2 is catastrophic overfitting, not a real effect. At
d=65536 the Gram diagonal after standardisation is ~n_train while the penalty is
1e-3*n = 0.256, i.e. essentially no shrinkage, so the solve explodes in the many
near-zero-eigenvalue directions. The permutation null of the legacy readout on
SYNTHETIC non-collinear data passed (-0.0278), so the defect is geometry-specific --
which is exactly why it matters here: psi is near-collinear AND unit-norm.

CONSEQUENCE: the absolute psi r2 values above are NOT trustworthy, and the commit
that used them to conclude "the bridge discards state" rests on a broken instrument.

THIS SCRIPT
  1. loads the cached real Qwen3-VL features (no VLM reload needed)
  2. rebuilds psi from a fresh bridge at the same seed
  3. scores every representation with BOTH readouts
  4. scores each with a PERMUTATION NULL under BOTH readouts
  5. verdicts on whether psi's readability loss survives a trustworthy instrument

PRE-REGISTERED (fixed before running)
  instrument_validated_X : |permutation null| < 0.10 for readout X on this data
  psi_readable_under_fixed : r2_ridge(psi) >= 0.50
  legacy_numbers_are_artifacts : legacy null fails AND fixed null passes
If the fixed readout ALSO shows psi unreadable, the conclusion stands and the
instrument was merely imprecise. If psi becomes readable, the conclusion REVERSES.
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


def r2_legacy(x: torch.Tensor, y: torch.Tensor, frac: float = 0.25,
              lam_rel: float = 1e-3) -> float:
    """The SUPERSEDED readout -- kept here only to quantify its defect."""
    n = int(x.shape[0])
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


def score(x: torch.Tensor, S: torch.Tensor, seed: int = 3) -> dict:
    """Both readouts, plus a permutation null for each. x: [n, d], S: [n, 4]."""
    x = x.reshape(x.shape[0], -1)
    n = int(x.shape[0])
    pidx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    return {
        "d": int(x.shape[-1]),
        "r2_fixed": round(r2_ridge(x, S), 4),
        "r2_legacy": round(r2_legacy(x, S), 4),
        "null_fixed": round(r2_ridge(x, S[pidx]), 4),
        "null_legacy": round(r2_legacy(x, S[pidx]), 4),
        "cos": round(offdiag_cos(x), 8),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/root/henri/vlm_feat_cache.pt")
    ap.add_argument("--out", default="/root/henri/vlm_readout_remeasure.json")
    ap.add_argument("--d-pool", type=int, default=512)
    a = ap.parse_args()

    R = {"schema": "henri.vlm_readout_remeasure/1",
         "when": time.strftime("%Y-%m-%d %H:%M:%S"),
         "purpose": ("re-measure psi readability with a permutation-null-validated "
                     "readout; the legacy fixed-penalty readout failed its own null "
                     "on near-collinear unit-norm features (r2 -1.4811)"),
         "pre_registered": {
             "instrument_validated": "|permutation null| < 0.10",
             "psi_readable_under_fixed": "r2_ridge(psi) >= 0.50",
             "legacy_numbers_are_artifacts": "legacy null fails AND fixed null passes"}}

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
    print(f"[cache] n={n} NP={NP} DV={DV}", flush=True)

    # ---------------------------------------------------------- representations
    reps = {}
    reps["input_mean_pooled_1024"] = feats.mean(dim=1)

    import henri_vla_vlm_bridge as B
    torch.manual_seed(77)                       # same seed as the aux experiment
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
                   "binding_ratio": round(float(bridge.binding_ratio), 6),
                   "dc_share": round(float(getattr(bridge, "dc_share", float("nan"))), 6),
                   "trainable_params": bridge.trainable_params()}
    reps["attention_pooled_1024"] = pooled_attn
    reps["to_pool_512"] = z
    reps["to_phase_65536"] = p
    if bsum is not None:
        reps["binding_term_65536"] = bsum
    reps["accumulator_65536"] = acc
    reps["psi_65536_unitnorm"] = psi

    R["representations"] = {}
    for name, x in reps.items():
        R["representations"][name] = score(x, S)
        print(f"[rep] {name:28s} {R['representations'][name]}", flush=True)

    # ------------------------------------------------------------- verdicts
    r = R["representations"]
    psi_s = r["psi_65536_unitnorm"]
    pooled_s = r["input_mean_pooled_1024"]
    fixed_valid = abs(psi_s["null_fixed"]) < 0.10
    legacy_broken = abs(psi_s["null_legacy"]) >= 0.10
    R["verdicts"] = {
        "psi_permutation_null_fixed": psi_s["null_fixed"],
        "psi_permutation_null_legacy": psi_s["null_legacy"],
        "instrument_validated_on_psi": bool(fixed_valid),
        "legacy_readout_broken_on_psi": bool(legacy_broken),
        "psi_readable_under_fixed": bool(psi_s["r2_fixed"] >= 0.50),
        "pooled_readable_under_fixed": bool(pooled_s["r2_fixed"] >= 0.50),
        "legacy_numbers_are_artifacts": bool(legacy_broken and fixed_valid),
        "readability_drop_fixed_readout": round(
            pooled_s["r2_fixed"] - psi_s["r2_fixed"], 4),
    }
    v = R["verdicts"]
    if v["legacy_numbers_are_artifacts"]:
        R["conclusion"] = (
            "The legacy readout is INVALID for psi's geometry (its own permutation "
            "null fails), so the previously committed psi r2 values are artifacts. "
            "Use the fixed-readout numbers, which pass their null.")
    elif fixed_valid:
        R["conclusion"] = (
            "The fixed readout is valid here and its numbers govern. Compare "
            f"pooled {pooled_s['r2_fixed']} vs psi {psi_s['r2_fixed']} to decide "
            "whether readability is genuinely lost.")
    else:
        R["conclusion"] = (
            "NEITHER readout passes its null on psi's geometry, so no psi "
            "readability claim can be made with this instrument. Report BLOCKED.")

    json.dump(R, open(a.out, "w"), indent=2)
    print("\n" + json.dumps({"bridge": R["bridge"], "verdicts": v,
                             "conclusion": R["conclusion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
