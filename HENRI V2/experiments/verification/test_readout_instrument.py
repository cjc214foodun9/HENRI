"""INSTRUMENT CHECK: is `r2_psi_vlm_bridge 0.0509` real, or a readout artifact?

WHY THIS EXISTS
===============
Three explanations for the VLM arm's failure have been tested and FALSIFIED:
  1. DC / bias in the bridge         -- GPU re-run: 0.0691 -> 0.0509, no recovery
  2. H3: L2-normalisation crushes the deviation
     -- test_bridge_near_collinear.py: bridge_preserves_readability, r2 0.9843
So the surviving claim is the diagnostic's headline verdict,
`H1_bridge_destroys_information: true` (r2 0.7559 -> 0.0509).

THE SUSPECT IS THE INSTRUMENT, NOT THE BRIDGE.
vlm_repr_diag.py::_r2 solves kernel ridge with a FIXED penalty:
    k = xtr @ xtr.T ;  a = solve(k + 1e-2 * I, ytr)
`psi` is UNIT-NORM by construction (forward() divides by the norm), so every
diagonal of `k` is 1.0 and, given cos ~ 0.9999, `k` is near-singular with a large
top eigenvalue (~n). A fixed 1e-2 is then effectively ZERO regularisation: the
solve is dominated by the tiny eigenvalues and held-out prediction collapses.
Raw pixels, by contrast, have large feature norms, so the same penalty is
comparatively negligible and the readout looks healthy.

That is a readout that reports "no information" on a near-collinear unit-norm
representation REGARDLESS of whether information is present. This test decides
it by construction: identical data, identical state, two readouts.

  readout FIXED   : solve(k + 1e-2 * I, y)          (vlm_repr_diag.py, as used)
  readout SCALED  : solve(k + 1e-3*n * I, y), features standardised

PASS (artifact confirmed) iff FIXED reports low R^2 while SCALED reports high
R^2, on data known to carry the state. If BOTH are low, the artifact story is
wrong and the bridge really does lose information.
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


def r2_fixed(x, y, frac=0.25, lam=1e-2):
    """The readout as used in vlm_repr_diag.py -- fixed penalty, raw features."""
    n = x.shape[0]
    nte = max(1, int(n * frac))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    te, tr = perm[:nte], perm[nte:]
    xtr, ytr = x[tr].double(), y[tr].double()
    xte, yte = x[te].double(), y[te].double()
    k = xtr @ xtr.T
    a = torch.linalg.solve(k + lam * torch.eye(k.shape[0], dtype=k.dtype), ytr)
    pred = (xte @ xtr.T) @ a
    res = ((yte - pred) ** 2).sum(0)
    tot = ((yte - yte.mean(0, keepdim=True)) ** 2).sum(0).clamp(min=1e-12)
    return float((1.0 - res / tot).mean())


def r2_scaled(x, y, frac=0.25, lam_rel=1e-3):
    """Scale-aware readout: standardise, then regularise RELATIVE to n."""
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


def main() -> int:
    torch.manual_seed(0)
    n, d, sdim = 256, 4096, 4
    col = torch.rand(n, sdim, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.25

    cases = {}

    # (i) UNIT-NORM near-collinear -- the geometry of `psi`
    m = F.normalize(torch.randn(d, generator=torch.Generator().manual_seed(2)), dim=0)
    dev = (col - col.mean(0, keepdim=True)) @ torch.randn(sdim, d,
          generator=torch.Generator().manual_seed(3))
    dev = dev - dev.mean(0, keepdim=True)
    c_t = 0.9999
    eps = ((2 * (1 - c_t)) / max(float((dev ** 2).sum(1).mean()), 1e-30)) ** 0.5
    u = m[None, :] + eps * dev
    u = u / u.norm(dim=-1, keepdim=True)               # UNIT NORM, like psi
    cases["unit_norm_near_collinear"] = u

    # (ii) LARGE-NORM feature block (raw-pixel-like scale)
    cases["large_norm_features"] = u * 30.0

    # (iii) well-conditioned unit-norm control (must be readable by BOTH)
    w = torch.randn(n, d, generator=torch.Generator().manual_seed(4))
    cases["well_conditioned_unit_norm"] = F.normalize(w, dim=-1)

    out = {"schema": "henri.readout_instrument_check/1",
           "n": n, "d": d, "state_dim": sdim,
           "readouts": {"FIXED": "solve(k + 1e-2*I, y)  [vlm_repr_diag.py as used]",
                        "SCALED": "standardise, solve(k + 1e-3*n*I, y)"},
           "cases": {}}

    for name, x in cases.items():
        out["cases"][name] = {
            "mean_offdiag_cos": round(offdiag_cos(x), 8),
            "feat_norm_mean": round(float(x.norm(dim=-1).mean()), 4),
            "r2_FIXED": round(r2_fixed(x, col), 4),
            "r2_SCALED": round(r2_scaled(x, col), 4),
        }

    a = out["cases"]["unit_norm_near_collinear"]
    art = (a["r2_FIXED"] < 0.20) and (a["r2_SCALED"] > 0.80)
    out["verdicts"] = {
        "same_data_state_present": True,
        "FIXED_collapses_on_unit_norm_near_collinear": bool(a["r2_FIXED"] < 0.20),
        "SCALED_reads_it": bool(a["r2_SCALED"] > 0.80),
        "READOUT_ARTIFACT_CONFIRMED": bool(art),
        "well_conditioned_readable_by_both":
            bool(out["cases"]["well_conditioned_unit_norm"]["r2_FIXED"] > 0.80
                 and out["cases"]["well_conditioned_unit_norm"]["r2_SCALED"] > 0.80),
    }
    out["consequence"] = (
        "If READOUT_ARTIFACT_CONFIRMED, then r2_psi_vlm_bridge 0.0509 in "
        "vlm_repr_diag_receipt.json is an artifact of the fixed-penalty readout on "
        "unit-norm near-collinear psi, NOT evidence that the bridge discards state. "
        "The verdict H1_bridge_destroys_information must be re-issued with the "
        "SCALED readout before any architectural change is justified."
        if art else
        "If NOT confirmed, the fixed-penalty readout is exonerated and the bridge "
        "really does lose state; the search continues in the bridge.")

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
