"""ARC Task Functor Compiler (Phase 7.2 Step 1) - continuous-domain, default-off.

Compiles the Inductive Task Functor W_task from public (X, Y) grid-pair
demonstrations in the complex half-space of the live continuous UWE
([D/2] complex, stored as [Re, Im] in [D] real).

Protocol (per Phase 7.2 PDF Lens A):
    W_task = normalize( sum_i conj(Psi_X,i) * Psi_Y,i )   (elementwise,
    complex domain = circular convolution of the two waves in time domain)

Goal anchor for the live episode (no X_test available):
    Psi_goal = normalize( mean_i Psi_Y,i )   (prototype of outputs)

Falsifiable kill experiment (pre-registered):
    FUNCTOR_OK iff on a HELD-OUT demo pair (X_h, Y_h):
        cos(normalize(W_task * Psi_X,h), Psi_Y,h) > 0.3
        AND > cos(Psi_X,h, Psi_Y,h) + 0.1   (beats identity baseline)
    else FUNCTOR_FALSIFIED.

All storage/provenance fields are sha256 digests; the module is read-only
over the corpus and never writes to the repo.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

STATUS_OK = "FUNCTOR_OK"
STATUS_FALSIFIED = "FUNCTOR_FALSIFIED"
STATUS_NO_DEMOS = "BLOCKED_NO_DEMONSTRATIONS"
STATUS_EMPTY = "BLOCKED_EMPTY_DEMOS"
STATUS_IMPORT = "BLOCKED_IMPORT_FAILED"

_RECOVERY_COS_THRESHOLD = 0.3
_IDENTITY_MARGIN = 0.1


@dataclass
class TaskFunctorResult:
    status: str = ""
    reason: str = ""
    task_id: str = ""
    demo_pair_count: int = 0
    held_out_cos: Optional[float] = None
    identity_cos: Optional[float] = None
    w_task_sha256: str = ""
    goal_wave_sha256: str = ""
    pairs_digest: str = ""
    provenance: dict = field(default_factory=dict)


def _to_complex(real_wave: torch.Tensor) -> torch.Tensor:
    """[D] real (Re|Im concatenation) -> [D/2] complex, unit norm."""
    d = real_wave.numel()
    half = d // 2
    r = real_wave[:half].to(torch.float32)
    i = real_wave[half:].to(torch.float32)
    z = torch.complex(r, i)
    return F.normalize(z, p=2, dim=-1)


def _to_real(complex_wave: torch.Tensor) -> torch.Tensor:
    """[D/2] complex -> [D] real (Re|Im concatenation), unit norm."""
    return F.normalize(torch.cat([complex_wave.real, complex_wave.imag], dim=-1), p=2, dim=-1)


def _pairs_digest(demo_pairs: Sequence[Tuple]) -> str:
    h = hashlib.sha256()
    for x, y in demo_pairs:
        h.update(b"X")
        h.update(str(x.tolist() if hasattr(x, "tolist") else x).encode("utf-8"))
        h.update(b"Y")
        h.update(str(y.tolist() if hasattr(y, "tolist") else y).encode("utf-8"))
    return h.hexdigest()


def compile_task_functor(
    demo_pairs: Sequence[Tuple],
    tokenizer: object,
    device: str = "cpu",
    task_id: str = "",
    hold_out_index: int = -1,
) -> TaskFunctorResult:
    """Compile W_task and a goal anchor from (X, Y) grid pairs.

    demo_pairs: sequence of (input_grid, output_grid) list/np/array pairs.
    hold_out_index: index of the pair held out for the falsifiable check
        (default -1 = last). Compile uses all other pairs.
    """
    res = TaskFunctorResult(task_id=task_id, demo_pair_count=len(demo_pairs))
    if not demo_pairs:
        res.status = STATUS_NO_DEMOS
        res.reason = "no demonstration pairs supplied"
        return res
    res.pairs_digest = _pairs_digest(demo_pairs)
    encode = getattr(tokenizer, "encode_spatial_grid", None)
    if encode is None:
        res.status = STATUS_IMPORT
        res.reason = "tokenizer lacks encode_spatial_grid"
        return res

    # Encode all pairs in the complex domain.
    waves = []
    for x, y in demo_pairs:
        _x = x.tolist() if hasattr(x, "tolist") else x
        _y = y.tolist() if hasattr(y, "tolist") else y
        wx = encode(_x).squeeze(0).reshape(-1).to(device)
        wy = encode(_y).squeeze(0).reshape(-1).to(device)
        if wx.dim() != 1 or wy.dim() != 1:
            res.status = STATUS_IMPORT
            res.reason = f"encode produced shape {tuple(wx.shape)}/{tuple(wy.shape)}"
            return res
        waves.append((_to_complex(wx), _to_complex(wy)))

    n = len(waves)
    if hold_out_index < 0:
        hold_out_index = n - 1
    hold_out_index = min(hold_out_index, n - 1)
    train = [w for i, w in enumerate(waves) if i != hold_out_index]
    hold_x, hold_y = waves[hold_out_index]

    if not train:
        res.status = STATUS_EMPTY
        res.reason = "no training pairs after hold-out"
        return res

    # W_task = normalize( sum_i conj(Psi_X,i) * Psi_Y,i )
    # Carrier F7 (default-OFF, HENRI_F7_AFFINE=1): per-task non-unitary affine
    # operator (implicit dual ridge, real domain) + supervised egress
    # (docs/spec/f7_affine_egress_preregistration.md, Appendix C). Carrier F6
    # (HENRI_F6_FUNCTOR=1) remains available. The legacy path is byte-identical
    # when BOTH flags are unset (Gate G6-class differential, contract C5).
    f6_mask: Optional[torch.Tensor] = None
    f6_ns_err: Optional[float] = None
    f6_ns_iters: Optional[int] = None
    f6_recon: Optional[float] = None
    _f7_held_cos: Optional[float] = None
    _f7_identity_cos: Optional[float] = None
    _f7_active = os.environ.get("HENRI_F7_AFFINE") == "1" and len(train) >= 2
    if _f7_active:
        from f7_affine_egress import AffineEgress
        Xtr = torch.stack([_to_real(wx) for wx, _ in train]).to(device)
        Ytr = torch.stack([_to_real(wy) for _, wy in train]).to(device)
        eg = AffineEgress(lam=1e-3).fit(Xtr, Ytr)
        eg.to(device)
        hx_r = _to_real(hold_x).to(device).unsqueeze(0)
        hy_r = F.normalize(_to_real(hold_y).to(device), p=2, dim=-1)
        z_hold = F.normalize(eg.predict(hx_r).squeeze(0), p=2, dim=-1)
        w_task = _to_complex(z_hold)
        _f7_held_cos = float(torch.abs(torch.dot(z_hold, hy_r)).item())
        _f7_identity_cos = float(torch.abs(
            torch.dot(F.normalize(hx_r.squeeze(0), p=2, dim=-1), hy_r)).item())
        _f7_factor_sha = hashlib.sha256(
            eg._GinvX.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    elif os.environ.get("HENRI_F6_FUNCTOR") == "1" and len(train) >= 2:
        from f6_adaptive_functor import compile_adaptive_functor
        Xtr = torch.stack([wx for wx, _ in train]).to(device)
        Ytr = torch.stack([wy for _, wy in train]).to(device)
        w_task, f6_mask, f6_ns_err, f6_ns_iters, f6_recon = compile_adaptive_functor(
            Xtr, Ytr, max_iters=8, tol=1e-5, eps_floor=1e-3)
    else:
        w_task = torch.zeros_like(train[0][0])
        for wx, wy in train:
            w_task = w_task + torch.conj(wx) * wy
        w_task = F.normalize(w_task, p=2, dim=-1)

    # Goal anchor = prototype of training outputs.
    goal_c = torch.zeros_like(train[0][1])
    for _, wy in train:
        goal_c = goal_c + wy
    goal_c = F.normalize(goal_c, p=2, dim=-1)

    # Falsifiable held-out check.
    with torch.no_grad():
        if _f7_active:
            held_out_cos = _f7_held_cos
            identity_cos = _f7_identity_cos
        else:
            pred = F.normalize(w_task * hold_x, p=2, dim=-1)
            held_out_cos = float(torch.real(torch.vdot(pred, hold_y)).item())
            identity_cos = float(torch.real(torch.vdot(hold_x, hold_y)).item())

    res.held_out_cos = held_out_cos
    res.identity_cos = identity_cos
    res.w_task_sha256 = _wave_digest(_to_real(w_task))
    res.goal_wave_sha256 = _wave_digest(_to_real(goal_c))
    res.provenance = {
        "schema_id": "henri.task-functor.v1",
        "task_id": task_id,
        "demo_pair_count": n,
        "hold_out_index": hold_out_index,
        "pairs_digest": res.pairs_digest,
        "threshold_cos": _RECOVERY_COS_THRESHOLD,
        "identity_margin": _IDENTITY_MARGIN,
        "device": device,
    }
    if held_out_cos > _RECOVERY_COS_THRESHOLD and held_out_cos > identity_cos + _IDENTITY_MARGIN:
        res.status = STATUS_OK
        res.reason = (
            f"held-out recovery cos={held_out_cos:.4f} > "
            f"identity {identity_cos:.4f} + {_IDENTITY_MARGIN}"
        )
    else:
        res.status = STATUS_FALSIFIED
        res.reason = (
            f"held-out recovery cos={held_out_cos:.4f} vs identity "
            f"{identity_cos:.4f} (threshold {_RECOVERY_COS_THRESHOLD})"
        )
    if _f7_active:
        # egress provenance must be merged AFTER the dict assignment above
        res.provenance["egress"] = {
            "schema_id": "f7-affine-egress.v1",
            "implicit": True,
            "factor_sha256": _f7_factor_sha,
        }
    return res


def _wave_digest(wave: torch.Tensor) -> str:
    h = hashlib.sha256()
    h.update(wave.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def goal_anchor_from_result(res: TaskFunctorResult) -> Optional[torch.Tensor]:
    """Reconstruct the real-domain goal anchor tensor from provenance."""
    return None  # anchors are materialized only inside the runner at init
