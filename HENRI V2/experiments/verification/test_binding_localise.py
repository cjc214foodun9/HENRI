"""LOCALISE the state loss: does the BINDING term destroy readability? ZERO GPU.

MEASURED (receipt vlm_bridge_trainability_receipt.json, RTX 5090):
    pooled (mean over patches)  r2_std  0.7999     input IS readable
    psi_at_init                 r2_std  0.3553
    psi_AFTER training          r2_std -0.5        WORSE than init
    psi mean off-diag cos               0.99977    still near-collinear
    binding_ratio                       0.037975   psi is ~96% pooled
So with the PROVEN scale-aware readout, a TRAINED bridge still discards state that
its input carries. Ruled out already: DC/bias (H1), L2-normalisation (H3),
fixed-penalty readout (H4).

HYPOTHESIS H5 (this probe). The binding term is a VSA SUPERPOSITION over
n_patches=256 role-filler pairs:

    bsum = sum_{n=0..255} phi(f_n) (.) pos_codes[n]

Superposing 256 high-dimensional pairs is noise-dominated: each fill is diluted
1/sqrt(256) while 256 independent noise directions accumulate. That term is added
into the same accumulator as the pooled term and then L2-normalised, so its noise
can dominate the coordinates that carry state.

TEST: identical synthetic features carrying a known 4-D state, pushed through the
REAL bridge with binding ON vs OFF. If ON loses readability and OFF keeps it, the
binding term is the destroyer and the fix is scoped.

FEATURE DESIGN (matched to the measured geometry):
  - one dominant common direction so mean off-diag cos ~ 0.9999 (as measured)
  - the state lives in the mean over patches (so mean-pooling is readable, as the
    pooled r2 0.7999 shows)
  - per-patch deviations are state-INDEPENDENT, so only the pooled component can
    carry state -- exactly the condition under which the bridge must not lose it

NOTE ON A WEAKNESS IN MY EARLIER H3 TEST, stated rather than hidden: that test
repeated ONE patch vector across all N, so with centred pos_codes the binding sum
collapsed to ~0 (binding_ratio was 4.3e-05). It therefore never exercised the
binding path. This probe uses per-patch variation so the path is live.
"""

import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", ".."))
if V2 not in sys.path:
    sys.path.insert(0, V2)

from henri_vla_vlm_bridge import VLMPhaseBridge  # noqa: E402

D_MODEL = 65536


def r2_std(x, y, frac=0.25, lam_rel=1e-3):
    """Scale-aware held-out R^2 (the readout proven by test_readout_instrument.py)."""
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


def offdiag_cos(f):
    c = F.normalize(f.reshape(f.shape[0], -1).double(), dim=-1)
    c = c @ c.T
    n = c.shape[0]
    return float((c.sum() - c.diag().sum()) / (n * (n - 1)))


def build_features(n, npatch, dvit, seed=0):
    """Synthetic [n, npatch, dvit] whose MEAN over patches carries a 4-D state."""
    g = torch.Generator().manual_seed(seed)
    S = torch.rand(n, 4, generator=g) * 0.5 + 0.25
    common = torch.randn(dvit, generator=g)
    common = common / common.norm()
    W = torch.randn(4, dvit, generator=g)
    base = (S - S.mean(0, keepdim=True)) @ W          # state lives here
    base = base / base.std().clamp(min=1e-12) * 1.0
    eps_b = 1e-3
    eps_n = 1e-3
    noise = torch.randn(n, npatch, dvit, generator=g)
    feats = common[None, None, :] + eps_b * base[:, None, :] + eps_n * noise
    return feats, S, {"eps_base": eps_b, "eps_noise": eps_n}


def main() -> int:
    torch.manual_seed(0)
    n, NP, DV = 256, 256, 1024
    feats, S, geo = build_features(n, NP, DV)

    R = {"schema": "henri.bridge_binding_localise/1",
         "geometry": {"n": n, "n_patches": NP, "d_vit": DV, "d_model": D_MODEL, **geo},
         "readout": "r2_std (scale-aware), held-out 25%, lam_rel=1e-3"}

    pooled = feats.mean(dim=1)
    R["A_pooled_r2"] = round(r2_std(pooled, S), 4)
    R["A_feats_offdiag_cos"] = round(offdiag_cos(feats.reshape(n, -1)), 8)

    for name, use_bind in (("B_binding_ON", True), ("C_binding_OFF", False)):
        b = VLMPhaseBridge(d_vit=DV, n_patches=NP, d_model=D_MODEL, d_pool=512,
                           use_position_binding=use_bind).eval()
        with torch.no_grad():
            psi = b(feats).float()
        R[name] = {
            "r2_std": round(r2_std(psi, S), 4),
            "mean_offdiag_cos": round(offdiag_cos(psi), 8),
            "binding_active": bool(getattr(b, "binding_active", False)),
            "binding_ratio": round(float(getattr(b, "binding_ratio", 0.0)), 6),
            "dc_share": round(float(getattr(b, "dc_share", float("nan"))), 6),
        }
        # is the state readable from the BINDING TERM alone?
        if use_bind:
            with torch.no_grad():
                bsum = b.binding_term(feats.to(torch.float32))
            R["B_terms"] = {
                "binding_term_r2": round(r2_std(bsum, S), 4),
                "binding_term_norm_over_psi": round(
                    float(bsum.norm(dim=-1).mean()), 6),
            }

    on = R["B_binding_ON"]["r2_std"]
    off = R["C_binding_OFF"]["r2_std"]
    a = R["A_pooled_r2"]
    R["verdicts"] = {
        "input_readable": bool(a > 0.50),
        "binding_ON_loses_state": bool(a - on > 0.20),
        "binding_OFF_keeps_state": bool(off > 0.50),
        "H5_BINDING_TERM_IS_THE_DESTROYER": bool((a - on > 0.20) and (off > 0.50)),
        "r2_delta_on_minus_off": round(on - off, 4),
    }
    R["next_action"] = (
        "Scope the fix to the binding term: the accumulator mixes a noise-dominated "
        "256-way VSA superposition into the same vector as the pooled signal, then "
        "L2-normalises. Candidates: carry binding in a SEPARATE channel rather than "
        "summing into the pooled accumulator; weight the two terms by their norms; "
        "or reduce superposition load."
        if R["verdicts"]["H5_BINDING_TERM_IS_THE_DESTROYER"] else
        "Binding is NOT the destroyer; the loss is in to_pool/to_phase or the "
        "attention pool. Next: read the state from the bridge's internal pooled "
        "vector (attention-pooled, not mean-pooled) to split those two.")

    out = os.path.join(os.environ.get("LOCALAPPDATA", "/tmp"), "Temp",
                       "henri_audit", "binding_localise.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(R, open(out, "w", encoding="utf-8"), indent=2)
    print(json.dumps(R, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
