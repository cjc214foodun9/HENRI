"""REGRESSION GUARD: psi must not collapse to a single direction.

WHY THIS EXISTS
===============
Measured (2026-09-29), on the RTX 5090, by experiments/verification/vlm_repr_diag.py:

    r2_vlm_pooled_features  0.7559   the VLM features ARE readable for state
    r2_psi_vlm_bridge       0.0691   after THIS bridge, readability is gone
    r2_psi_random_ingress   0.9886   the random ingress PRESERVES state

so psi discards state information its own input contains. That is why the VLM arm
scored 0.078125 while the random-feature arm scored 0.4688.

CONTRIBUTING CAUSE (measured locally, not assumed): biases on the projection
stack inject an input-independent vector -- to_phase.bias, plus to_phase.bias
dotted with sum_n pos_codes[n] via the binding sum, the codes being uncentered.
Removing those sources (bias=False, centered codes) dropped the DC energy share
from ~0.57 to 0.0163 and cos_sim from ~0.57 to 0.015.

HONEST SCOPE -- the GPU cos_sim 0.99987 is CONFOUNDED: the VLM's pooled features
are themselves cos 0.999999 across observations for this synthetic scene. The
absolute 0.99987 is therefore NOT attributable to this bridge alone. This guard
asserts what is actually established: the output geometry is healthy, and the
legacy configuration measurably WORSENS it.

The positive control is FAITHFUL: it takes a real VLMPhaseBridge and restores
exactly the two legacy properties, so the comparison isolates that delta. A guard
whose control cannot fail proves nothing.
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.abspath(os.path.join(HERE, "..", ".."))
if V2 not in sys.path:
    sys.path.insert(0, V2)

from henri_vla_vlm_bridge import VLMPhaseBridge  # noqa: E402


def _cos_offdiag(psi: torch.Tensor) -> float:
    b = psi.shape[0]
    c = F.normalize(psi, dim=-1) @ F.normalize(psi, dim=-1).T
    return float((c.sum() - c.diag().sum()) / (b * (b - 1)))


def _dc_share(x: torch.Tensor) -> float:
    """Energy fraction along the single common (mean) direction of the batch."""
    if x.shape[0] < 2:
        return float("nan")
    m = x.mean(dim=0)
    n = m.norm()
    if float(n) < 1e-12:
        return 0.0
    mh = m / n
    proj = x @ mh
    return float((proj ** 2).mean() / (x ** 2).sum(dim=-1).mean().clamp(min=1e-12))


def _legacy_variant(d_vit: int, n_patches: int, d_model: int, d_pool: int = 512,
                    seed: int = 0) -> VLMPhaseBridge:
    """FAITHFUL positive control: a real bridge with the two legacy properties.

    Everything else (architecture, code path, v_proj/attention weights) is
    identical, so any measured difference isolates bias=True + uncentered codes.
    Bias is left at nn.Linear's default init, which is what the defect had.
    """
    b = VLMPhaseBridge(d_vit=d_vit, n_patches=n_patches, d_model=d_model,
                       d_pool=d_pool, seed=seed).eval()
    with torch.no_grad():
        lin_pool = nn.Linear(d_vit, d_pool, bias=True)      # default bias init
        lin_pool.weight.data.copy_(b.to_pool.weight.data)
        lin_phase = nn.Linear(d_pool, d_model, bias=True)   # default bias init
        lin_phase.weight.data.copy_(b.to_phase.weight.data)
        b.to_pool = lin_pool
        b.to_phase = lin_phase
        g = torch.Generator().manual_seed(seed)
        pos = torch.randn(n_patches, d_model, generator=g)  # legacy: UNIT-norm,
        pos = pos / pos.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # not centered
        b.pos_codes = pos.to(torch.float32)
    return b


def main() -> int:
    torch.manual_seed(0)
    B, N, DVIT, D = 64, 256, 1024, 65536
    feats = torch.randn(B, N, DVIT)

    out = {"schema": "henri.bridge_dc_share/2",
           "geometry": {"B": B, "N": N, "d_vit": DVIT, "d_model": D},
           "scope_note": ("the GPU cos_sim 0.99987 is confounded by VLM pooled "
                          "features that are themselves cos 0.999999; this guard "
                          "asserts local geometry + a faithful legacy control")}
    ok = True

    # ---------------- 1. FIXED bridge: geometry must be healthy ----------------
    b = VLMPhaseBridge(d_vit=DVIT, n_patches=N, d_model=D, d_pool=512).eval()
    with torch.no_grad():
        psi = b(feats)
        psi_perm = b(feats[torch.randperm(B, generator=torch.Generator().manual_seed(1))])
    fixed_cos, fixed_dc = _cos_offdiag(psi), _dc_share(psi)
    out["fixed"] = {
        "cos_sim_offdiag": round(fixed_cos, 6),
        "dc_share": round(fixed_dc, 6),
        "psi_unit_norm": round(float(psi.norm(dim=-1).mean()), 6),
        "inputs_distinguishable": round(float((psi - psi_perm).abs().mean()), 6),
        "binding_active": bool(b.binding_active),
        "binding_ratio": round(float(b.binding_ratio), 6),
        "pos_codes_rowsum_absmax": float(b.pos_codes.sum(dim=0).abs().max()),
    }
    c_fixed = {
        "cos_offdiag_lt_0.50": fixed_cos < 0.50,
        "dc_share_lt_0.50": fixed_dc < 0.50,
        "unit_norm": abs(out["fixed"]["psi_unit_norm"] - 1.0) < 1e-3,
        "binding_active": bool(b.binding_active),
        "pos_codes_centered": out["fixed"]["pos_codes_rowsum_absmax"] < 1e-2,
        "inputs_distinguishable": out["fixed"]["inputs_distinguishable"] > 1e-3,
    }
    out["checks"] = {"fixed_healthy": {k: bool(v) for k, v in c_fixed.items()}}
    ok = ok and all(c_fixed.values())

    # ---------------- 2. FAITHFUL control: the defect must show ----------------
    leg = _legacy_variant(DVIT, N, D, d_pool=512)
    with torch.no_grad():
        psi_l = leg(feats)
    leg_cos, leg_dc = _cos_offdiag(psi_l), _dc_share(psi_l)
    ratio = leg_dc / max(fixed_dc, 1e-9)
    c_ctrl = {
        "control_dc_share_gt_5x_fixed": leg_dc > 5.0 * max(fixed_dc, 1e-9),
        "control_cos_worse_than_fixed": leg_cos > fixed_cos + 0.05,
    }
    out["positive_control_legacy"] = {
        "cos_sim_offdiag": round(leg_cos, 6),
        "dc_share": round(leg_dc, 6),
        "dc_ratio_vs_fixed": round(ratio, 3),
        "note": ("bias=True + uncentered codes on the SAME bridge class; a "
                 "relative criterion is used because the absolute collapse level "
                 "depends on input geometry"),
        **{k: bool(v) for k, v in c_ctrl.items()},
    }
    out["checks"]["control_shows_defect"] = {k: bool(v) for k, v in c_ctrl.items()}
    ok = ok and all(c_ctrl.values())

    out["GUARD_HOLDS"] = bool(ok)
    import json
    print(json.dumps(out, indent=2))
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
