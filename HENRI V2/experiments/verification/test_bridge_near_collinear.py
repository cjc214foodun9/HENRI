"""Where does the bridge lose state? -- near-collinearity probe. ZERO GPU.

MEASURED ON THE RTX 5090 (experiments/verification/vlm_repr_diag.py):
    cos_sim_vlm_pooled      0.999999   Qwen3-VL pooled features are near-collinear
    r2_vlm_pooled_features  0.7559     yet the 4-D state IS linearly readable
    r2_psi_vlm_bridge       0.0509     after MY bridge it is NOT
    r2_psi_random_ingress   0.9886     the random ingress PRESERVES state
A bias/DC hypothesis was proposed, shipped, then FALSIFIED for the GPU case
(readability did not recover: 0.0691 -> 0.0509; and the random ingress also
carries dc_share 0.9111 while staying readable).

HYPOTHESIS H3 (this probe). The state lives in the TINY deviations from a
dominant direction. L2-normalising an accumulator dominated by that direction
maps every sample to nearly the same point, so the deviation -- and with it the
state -- is crushed.

This probe needs no VLM. It builds synthetic features with the MEASURED geometry
(cos ~ 0.999999) that carry a known 4-D state, and asks:
    A. is the state readable in the input?                    (validity check)
    B. readable after the bridge, as shipped?
    C. readable if the INPUT is high-passed (mean removed)?
    D. readable if `psi` is centred before normalising?
    E. readable if the state is read from the pre-normalisation accumulator?

If A is high and B is low, H3 holds. If C/D/E recover readability, the fix is
identified rather than guessed.

READOUT NOTE: the readout is a STANDARDISED ridge. A fixed ridge penalty is not
scale-invariant, so on near-collinear inputs it would suppress a low-variance
signal and report a false negative. That would be a bug in the probe, not a fact
about the bridge, so the probe standardises and says so.
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


def r2_std(x: torch.Tensor, y: torch.Tensor, frac: float = 0.25,
           lam_rel: float = 1e-3) -> float:
    """Held-out R^2 from a STANDARDISED kernel ridge (scale-invariant)."""
    n = x.shape[0]
    nte = max(1, int(n * frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    te, tr = perm[:nte], perm[nte:]
    xtr, xte = x[tr].double(), x[te].double()
    ytr, yte = y[tr].double(), y[te].double()
    xm = xtr.mean(0, keepdim=True)
    ym = ytr.mean(0, keepdim=True)
    s = xtr.std(0, keepdim=True).mean().clamp(min=1e-12)
    xtr, xte = (xtr - xm) / s, (xte - xm) / s
    ytr, yte = ytr - ym, yte - ym
    k = xtr @ xtr.T
    a = torch.linalg.solve(k + lam_rel * n * torch.eye(k.shape[0], dtype=k.dtype), ytr)
    pred = (xte @ xtr.T) @ a
    res = ((yte - pred) ** 2).sum(0)
    tot = ((yte) ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - res / tot).mean())


def mean_offdiag_cos(f: torch.Tensor) -> float:
    c = F.normalize(f.reshape(f.shape[0], -1), dim=-1)
    c = c @ c.T
    n = c.shape[0]
    return float((c.sum() - c.diag().sum()) / (n * (n - 1)))


def make_collinear(n: int, d: int, sdim: int, cos_target: float, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    S = torch.rand(n, sdim, generator=g) * 0.5 + 0.25          # state in [0.25,0.75]
    m = torch.randn(d, generator=g)
    m = m / m.norm()
    W = torch.randn(sdim, d, generator=g)
    dev = (S - S.mean(0, keepdim=True)) @ W
    v = float((dev ** 2).sum(1).mean())
    eps = ((2.0 * (1.0 - cos_target)) / max(v, 1e-30)) ** 0.5
    return m[None, :] + eps * dev, S, eps


def main() -> int:
    torch.manual_seed(0)
    n, N, dvit, sdim = 128, 8, 1024, 4
    COS = 0.999999
    S_ref = torch.rand(n, sdim, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.25
    Fcol, S, eps = make_collinear(n, dvit, sdim, COS)
    cos_in = mean_offdiag_cos(Fcol)

    R = {"schema": "henri.bridge_near_collinear/1",
         "geometry": {"n": n, "n_patches": N, "d_vit": dvit, "d_model": D_MODEL,
                      "cos_target": COS, "cos_measured": round(cos_in, 8),
                      "deviation_scale_eps": eps},
         "readout": "standardised kernel ridge, held-out 25%, lam_rel=1e-3"}

    # ---- A. the synthetic input must be readable (validity gate) -------------
    R["A_r2_input"] = round(r2_std(Fcol, S), 4)
    R["A_highpass_r2"] = round(r2_std(Fcol - Fcol.mean(0, keepdim=True), S), 4)

    # ---- build patch features the bridge can consume ------------------------
    g = torch.Generator().manual_seed(7)
    feats = Fcol[:, None, :].repeat(1, N, 1) + 1e-4 * torch.randn(
        n, N, dvit, generator=g)
    R["feats_mean_offdiag_cos"] = round(mean_offdiag_cos(feats), 8)

    bridge = VLMPhaseBridge(d_vit=dvit, n_patches=N, d_model=D_MODEL,
                            d_pool=512).eval()

    with torch.no_grad():
        psi = bridge(feats)                                  # B: as shipped
        # pre-normalisation accumulator, same code path as forward()
        pooled = bridge.pool(feats.to(torch.float32))
        p = bridge.to_phase(bridge.to_pool(pooled)) + bridge.phase_offset
        bsum = bridge.binding_term(feats.to(torch.float32)) if bridge.binding_active else 0.0
        acc = (bsum + p) if bridge.binding_active else p

    R["B_r2_psi_shipped"] = round(r2_std(psi, S), 4)
    R["B_cos_psi_shipped"] = round(mean_offdiag_cos(psi), 8)
    R["B_binding_ratio"] = round(float(bridge.binding_ratio), 6)

    # ---- C. high-pass the INPUT -------------------------------------------
    feats_h = feats - feats.mean(dim=0, keepdim=True)
    with torch.no_grad():
        psi_h = bridge(feats_h)
    R["C_r2_psi_highpassed_input"] = round(r2_std(psi_h, S), 4)
    R["C_cos_psi_highpassed_input"] = round(mean_offdiag_cos(psi_h), 8)

    # ---- D. centre the ACCUMULATOR before normalising ----------------------
    acc_c = acc - acc.mean(dim=0, keepdim=True)
    psi_c = acc_c / acc_c.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    R["D_r2_psi_centred"] = round(r2_std(psi_c, S), 4)
    R["D_cos_psi_centred"] = round(mean_offdiag_cos(psi_c), 8)

    # ---- E. read the state from the RAW accumulator (no normalisation) -----
    R["E_r2_accumulator_raw"] = round(r2_std(acc, S), 4)
    R["E_cos_accumulator_raw"] = round(mean_offdiag_cos(acc), 8)

    # ---- verdicts ---------------------------------------------------------
    a, b = R["A_r2_input"], R["B_r2_psi_shipped"]
    R["verdicts"] = {
        "input_is_readable": bool(a > 0.50),
        "bridge_loses_state": bool(a - b > 0.20),
        "H3_normalisation_crushes_deviation": bool(a > 0.50 and (b < 0.20)),
        "highpass_input_restores": bool(R["C_r2_psi_highpassed_input"] > 0.50),
        "centring_restores": bool(R["D_r2_psi_centred"] > 0.50),
        "raw_accumulator_readable": bool(R["E_r2_accumulator_raw"] > 0.50),
    }
    R["fix_identified"] = (
        "centre before normalising" if R["verdicts"]["centring_restores"] else
        "high-pass the input" if R["verdicts"]["highpass_input_restores"] else
        "normalisation is NOT the whole cause -- investigate upstream")

    out = os.path.join(os.environ.get("LOCALAPPDATA", "/tmp"), "Temp",
                       "henri_audit", "near_collinear.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(R, open(out, "w", encoding="utf-8"), indent=2)
    print(json.dumps(R, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
