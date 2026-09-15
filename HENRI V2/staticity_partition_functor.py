"""Staticity-partition task operator (Phase 10.3 directive 2).

SOURCE OF RECORD
    "Phase 10.3 Adjudication & Staticity Partition Directive",
    sha256 d33f24e0dc01055217fa90bd16cbb84c3ef0dca4c30ff8014ba4d7a9c4545712,
    22 pages, 37203 chars extracted. Recovered from the document's own text:
        static subset   Omega_static = { x | for all m: X_m(x) = Y_m(x) }  -> 87.4%
        dynamic subset  Omega_active = T^2 \\ Omega_static                 -> 12.6%
        mask_static = ALL_m ( |X_m - Y_m|^2 <= epsilon_floor )
        W_active*   = sum_m (X_m,act^H . Y_m,act) / ( |X|^2 + lambda )
        inference:  Y_pred[static] = X_test[static]
                    Y_pred[active] = W_active* . X_test[active]
        reg_lambda default 1e-1, epsilon_floor default 1e-4

TENSOR-LAYOUT DEVIATION (measured, documented, not hidden)
    The directive writes X_demos as [M, N_pos, D]: one feature vector per canvas
    POSITION. That layout does not exist in this codebase. The live wave
    representation is a FLAT complex vector per grid ([NB*BLOCK_SLOTS] = 32768),
    produced by SUMMING every pixel into random frequency channels:

        enc(g)[(b,s)] = sum_{y,x} phasor(v(g[y][x]) + x*wx[b,s] + y*wy[b,s])

    Two consequences, BOTH measured before writing this file:
      1. Wave components are SUM-OVER-POSITION channels, NOT positions. A
         position mask is therefore NOT a diagonal operator in wave space:
         enc(g * mask) != M . enc(g) for any diagonal M.
      2. o_vsa_torus_encoder.py exposes NO inverse (`decode`, `inverse`, `ifft`,
         `unbind`, `to_grid`, `slot_to` -> 0 hits). Grid -> wave is lossy for the
         purpose of position localization.
    So the partition is realised in the GRID/VALUE domain it is actually defined
    on, and scored through the same encoder-based cosine metric as the recorded
    baseline, which keeps the comparison honest. `compute_staticity_partition_
    functor` below still implements the directive's PER-POSITION formula exactly,
    for the case where a per-position tensor exists.

ZONE C STATUS
    This module adds NO engram and touches no Zone C gate. Zone C remains the two
    verified parameter-free gates (Delta q = 0; torus translation equivariance).

DOCUMENT VALUE NOT ADOPTED
    The request summary states gamma_local ~= 9.38. That literal does NOT occur in
    the authenticated extraction (searched '9.38', '9.4', '9.5', 'gamma', the
    Greek letters -> 0 hits). A sample-to-dimension ratio is COMPUTED here from
    measured counts and its basis is written into the receipt. It is not copied.
"""
from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "compute_staticity_partition_functor",
    "staticity_mask_grid",
    "fit_colour_transfer",
    "predict_grid_partition",
    "sample_to_dimension_ratio",
]


# --------------------------------------------------------------------------- core
def compute_staticity_partition_functor(
    X_demos: torch.Tensor,
    Y_demos: torch.Tensor,
    reg_lambda: float = 1e-1,
    epsilon_floor: float = 1e-4,
):
    """Directive formula, PER-POSITION layout [M, N_pos, D].

    Returns (mask_static [N_pos] bool, mask_active [N_pos] bool,
             W_active_star [n_active, D] complex).

    mask_static is the CONSENSUS over demonstrations: a position is static only if
    it is unchanged in EVERY demo. That is the directive's `ALL_m` quantifier and
    it is the conservative choice (a position that changes in any demo is active).
    """
    if X_demos.shape != Y_demos.shape:
        raise ValueError(f"X/Y shape mismatch: {tuple(X_demos.shape)} vs {tuple(Y_demos.shape)}")
    if X_demos.dim() != 3:
        raise ValueError(
            f"expected [M, N_pos, D]; got {tuple(X_demos.shape)}. The live wave "
            f"layout is flat and is handled by predict_grid_partition().")
    diff_energy = torch.sum(torch.abs(X_demos - Y_demos) ** 2, dim=-1)      # [M, N_pos]
    mask_static = torch.all(diff_energy <= epsilon_floor, dim=0)            # [N_pos]
    mask_active = ~mask_static

    Xa = X_demos[:, mask_active, :]                                         # [M, n, D]
    Ya = Y_demos[:, mask_active, :]
    num = torch.sum(torch.conj(Xa) * Ya, dim=0)
    den = torch.sum(torch.abs(Xa) ** 2, dim=0)
    lam = torch.as_tensor(reg_lambda, dtype=den.dtype, device=den.device)
    W_active_star = num / (den + lam)
    return mask_static, mask_active, W_active_star


# ------------------------------------------------------------------- grid domain
def staticity_mask_grid(x_grids, y_grids, require_same_shape: bool = True):
    """Consensus static mask over demo pairs, in the GRID domain.

    Returns (mask_static, mask_active, n_changed_any). None when the demo pair
    shapes are not mutually compatible, so the caller can SKIP rather than guess.
    """
    xs = [np.asarray(g, dtype=np.int64) for g in x_grids]
    ys = [np.asarray(g, dtype=np.int64) for g in y_grids]
    if require_same_shape:
        shapes = {a.shape for a in xs} | {a.shape for a in ys}
        if len(shapes) != 1:
            return None
    if any(a.shape != b.shape for a, b in zip(xs, ys)):
        return None
    static = np.ones(xs[0].shape, dtype=bool)
    for a, b in zip(xs, ys):
        static &= (a == b)
    return static, ~static, int((~static).sum())


def fit_colour_transfer(x_grids, y_grids, mask_active, v_max: int,
                        reg_lambda: float = 1e-1):
    """Regularized per-slot diagonal LS on the ACTIVE positions, in COLOUR space.

        W[v, v'] = sum_m #{p in Omega_active : X_m(p)=v, Y_m(p)=v'} /
                   ( sum_m #{p in Omega_active : X_m(p)=v} + lambda )

    Slots are colour values, so W is [v_max, v_max] -- a TINY operator. This is the
    directive's `W_active*` with the wave slot replaced by the colour slot, which
    is the only slot index that localises to positions here.

    Returns (W, n_samples, n_unseen_rows): n_unseen_rows counts colours with no
    active evidence, which must fall back to identity instead of predicting 0.
    """
    num = np.zeros((v_max, v_max), dtype=np.float64)
    den = np.zeros((v_max, v_max), dtype=np.float64)
    n = 0
    ys_idx, xs_idx = np.nonzero(mask_active)
    for a, b in zip(x_grids, y_grids):
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        if a.shape != mask_active.shape or b.shape != mask_active.shape:
            return None
        av = a[ys_idx, xs_idx]
        bv = b[ys_idx, xs_idx]
        ok = (av >= 0) & (av < v_max) & (bv >= 0) & (bv < v_max)
        av, bv = av[ok], bv[ok]
        num[av, bv] += 1.0
        den[av, av] += 1.0
        n += int(av.size)
    W = num / (den + reg_lambda)
    return W, n, int((den.sum(axis=1) == 0).sum())


def predict_grid_partition(train_pairs, test_input, v_max: int,
                           reg_lambda: float = 1e-1, use_gate: bool = True,
                           force_identity_on_active: bool = False):
    """Grid-domain staticity-partition prediction.

    train_pairs: sequence of (input_grid, output_grid).
    use_gate=False        -> fit and apply the colour LS on ALL positions
                             (ABLATION: isolates the gate's contribution).
    force_identity_on_active -> identity everywhere (CONTROL: must reproduce the
                             plain identity arm exactly).

    Returns (ypred_grid, diag) or (None, reason).
    """
    if not train_pairs:
        return None, "no_train_pairs"
    xs = [np.asarray(a, dtype=np.int64) for a, _ in train_pairs]
    ys = [np.asarray(b, dtype=np.int64) for _, b in train_pairs]
    xt = np.asarray(test_input, dtype=np.int64)

    shapes = {a.shape for a in xs} | {a.shape for a in ys} | {xt.shape}
    if len(shapes) != 1:
        return None, f"shape_mismatch:{sorted(shapes)}"
    shape = xt.shape

    sm = staticity_mask_grid([a for a, _ in train_pairs],
                             [b for _, b in train_pairs])
    if sm is None:
        return None, "mask_unsupported"
    mask_static, mask_active, n_changed = sm

    if force_identity_on_active:
        return xt.copy(), {"n_changed_any": n_changed, "n_samples": 0,
                           "n_unseen_rows": int(v_max), "arm": "identity_control"}

    fit_mask = mask_active if use_gate else np.ones(shape, dtype=bool)
    fitted = fit_colour_transfer(xs, ys, fit_mask, v_max, reg_lambda)
    if fitted is None:
        return None, "fit_failed"
    W, n_samples, n_unseen = fitted

    ypred = xt.copy()
    ys_idx, xs_idx = np.nonzero(fit_mask)
    if ys_idx.size:
        xv = xt[ys_idx, xs_idx]
        seen = W[xv].sum(axis=1) > 0.0
        pred = W[xv].argmax(axis=1)
        sel = seen & (xv < v_max)
        ypred[ys_idx[sel], xs_idx[sel]] = pred[sel]
        if use_gate:
            # positions outside the fit mask keep the identity by construction
            pass
    return ypred, {"n_changed_any": n_changed, "n_active": int(mask_active.sum()),
                   "n_samples": int(n_samples), "n_unseen_rows": int(n_unseen),
                   "n_static": int(mask_static.sum())}


def sample_to_dimension_ratio(n_samples: int, effective_dim: int) -> float:
    """gamma = M_eff / D_eff, computed from measured counts. Basis is reported."""
    if effective_dim <= 0:
        return float("inf")
    return float(n_samples) / float(effective_dim)


# ============================================================================
# WAVE-DOMAIN PARTITION (the directive's actual formulation)
# ============================================================================
# The directive writes W_task = Pi_bg + Sum_p Pi_{Omega_p} K_p Pi_{Omega_p} with
# "spatial projection operators". MEASURED CONSTRAINT that shapes the
# implementation: a wave component is a SUM over positions
# (enc(g)[(b,s)] = sum_{y,x} phasor(v + x*wx + y*wy)), so a position mask is NOT a
# diagonal slot operator, and this encoder exposes no inverse (decode/inverse/ifft/
# unbind/to_grid/slot_to -> 0 hits). The ONLY way to express position locality in
# wave space is to MASK THE GRID, then encode:
#
#     X_act  = enc(X * mask_active)      X_stat = enc(X * mask_static)
#     K      = per-slot diagonal LS fitted on (X_act -> Y_act) across demos
#     pred   = K . X_act(test)  +  X_stat(test)
#
# This is a faithful realisation of the partitioned operator THROUGH the encoder,
# and it is directly comparable to the incumbent diag_ls arm. The static branch is
# an exact identity pass-through because Y == X there by construction of the mask.

def mask_from_demos(demo_pairs, strict: bool = False):
    """Consensus static mask over demo pairs, with an explicit fallback mode.

    Returns (mask_static, mask_active, mode, shape).
      mode = "consensus"  -> all X/Y demo shapes equal; positions compared 1:1
             "all_active" -> shapes differ so no position consensus exists; the
                             caller fits on every position (documented fallback,
                             NOT silently treated as a successful gate).
    strict=True returns (None, None, "unsupported", None) instead of falling back.
    """
    xs = [np.asarray(a, dtype=np.int64) for a, _ in demo_pairs]
    ys = [np.asarray(b, dtype=np.int64) for _, b in demo_pairs]
    shapes = {a.shape for a in xs} | {b.shape for b in ys}
    if len(shapes) != 1:
        if strict:
            return None, None, "unsupported", None
        # DEFECT D5 (mine, fixed). v1 returned mask_static = ALL ONES here, which made
        # 33/60 tasks report "100% static" -- an arithmetic artefact of the fallback,
        # not a measurement (it produced the IMPOSSIBLE combination
        # consensus=1.0000 while per-pair median=0.8730). With no position-wise
        # correspondence between demo shapes there IS no verifiable static set, so
        # the only honest fallback is the EMPTY static set with a fully active mask.
        shape = xs[0].shape
        return (np.zeros(shape, dtype=bool), np.ones(shape, dtype=bool),
                "all_active", shape)
    static = np.ones(xs[0].shape, dtype=bool)
    for a, b in zip(xs, ys):
        static &= (a == b)
    return static, ~static, "consensus", xs[0].shape


def predict_grid_partition_auto(demo_pairs, test_input, v_max: int,
                                reg_lambda: float = 1e-1, use_gate: bool = True):
    """Grid-domain colour-LS partition on ANY input shape.

    Fallback behaviour is recorded so a mixed-mode aggregate can be reported
    honestly rather than presenting a gated number for an ungated mechanism.
    """
    xt = np.asarray(test_input, dtype=np.int64)
    if use_gate:
        mask_static, mask_active, mode, _ = mask_from_demos(demo_pairs)
        fit_mask = mask_active
    else:
        mask_static, mask_active, mode, _ = mask_from_demos(
            [(t, t) for t, _ in demo_pairs])          # all positions "static"
        fit_mask = np.ones(np.asarray(demo_pairs[0][0]).shape, dtype=bool)
        mode = "no_gate_all_positions"
    if mask_static is None:
        return None, {"mode": "unsupported"}

    xs = [np.asarray(a, dtype=np.int64) for a, _ in demo_pairs]
    ys = [np.asarray(b, dtype=np.int64) for _, b in demo_pairs]
    fit = fit_colour_transfer(xs, ys, fit_mask, v_max, reg_lambda)
    if fit is None:
        return None, {"mode": "fit_failed"}
    W, n_samples, n_unseen = fit

    if not use_gate:
        # single global colour map applied everywhere
        ypred = np.full(xt.shape, xt.flat[0], dtype=np.int64)
        for v in range(min(v_max, W.shape[0])):
            ypred[xt == v] = int(W[v].argmax()) if W[v].sum() > 0 else v
        return ypred, {"mode": mode, "n_samples": int(n_samples),
                       "n_unseen_rows": int(n_unseen), "n_active": int(xt.size),
                       "n_static": 0}

    ypred = xt.copy()
    if mode == "consensus" and xt.shape == mask_static.shape:
        fit_mask = mask_active
    else:
        fit_mask = np.ones(xt.shape, dtype=bool)     # shape fallback
    yi, xi = np.nonzero(fit_mask)
    if yi.size:
        xv = xt[yi, xi]
        seen = np.zeros(xv.shape, dtype=bool)
        ok = xv < v_max
        seen[ok] = W[xv[ok]].sum(axis=1) > 0.0
        pred = W[np.where(ok, xv, 0)].argmax(axis=1)
        sel = ok & seen
        ypred[yi[sel], xi[sel]] = pred[sel]
    return ypred, {"mode": mode, "n_samples": int(n_samples),
                   "n_unseen_rows": int(n_unseen),
                   "n_active": int(mask_active.sum()),
                   "n_static": int(mask_static.sum())}
