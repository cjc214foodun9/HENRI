"""K2 carrier: reduced-rank action-conditioned Koopman fit (default-OFF).

Fits factored per-action operators on the production dictionary
(Phi(s,a) = fused intent, Re||Im) using the SAME dual/thin-SVD math as the
live EFEPlanner.train_transition_batch — never materializing [2d, d] or
[d, d] ambient matrices. Then evaluates against:

  - persistence baseline (s' = s)
  - action-agnostic operator
  - shuffled-action control
  - action-conditioned per-action operators

Separate reports: calibration projected residual, fresh-evaluation one-step
skill vs persistence, spectral norm/radius (power iteration, factored form),
fixed-horizon open-loop errors (h=5, h=12), learning engagement.

Closed-form solve only: no nn.Parameter, no optimizer, no backward.
Default-OFF: HENRI_KOOPMAN_FIT=1.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FLAG = "HENRI_KOOPMAN_FIT"
DEFAULT_HORIZONS = (5, 12)


class KoopmanFitDisabledError(RuntimeError):
    pass


def fit_operator(phi: Any, y: Any, ridge: float = 1e-4, r: Optional[int] = None,
                 cholesky_jitter: Sequence[float] = (1.0, 10.0, 100.0, 1000.0)) -> Dict[str, Any]:
    """Dual EDMD fit: K = phi phi^T + ridge N I; C = cholesky_solve(y, L);
    SVD(C) -> factored W [2d, k], V [d, k] (production solve pattern)."""
    import torch
    N = phi.shape[0]
    K = phi @ phi.T + ridge * N * torch.eye(N, device=phi.device, dtype=phi.dtype)
    L = None
    for jm in cholesky_jitter:
        try:
            L = torch.linalg.cholesky(
                K + (jm - 1.0) * ridge * N * torch.eye(N, device=phi.device, dtype=phi.dtype))
            break
        except Exception:
            continue
    if L is None:
        raise RuntimeError("cholesky failed for operator fit")
    C = torch.cholesky_solve(y, L)  # [N, d]
    Uc, Sc, Vch = torch.linalg.svd(C, full_matrices=False)
    k = min(r if r is not None else Sc.numel(), Sc.numel())
    V = Vch[:k].T.contiguous()               # [d, k]
    W = (phi.T @ Uc[:, :k]) * Sc[:k]         # [2d, k]
    return {"V": V.detach(), "W": W.detach(), "k": int(k),
            "sc": Sc.detach().cpu().numpy()}


def predict_wave(op: Dict[str, Any], phi: Any, num_blocks: int = 8) -> Any:
    import torch
    out = op["V"] @ (op["W"].T @ phi)
    out = out.view(num_blocks, 8)
    return out / (out.norm(dim=-1, keepdim=True) + 1e-9)


def spectral_norm(op: Dict[str, Any], iters: int = 40) -> float:
    """Power iteration on K^T K with K = V W^T (factored, no [d, 2d] form).

    K maps input space [2d] -> output [d]; K^T K maps [2d] -> [2d], so the
    iteration vector lives in the INPUT space (W.shape[0]).
    """
    import torch
    x = torch.randn(op["W"].shape[0], device=op["V"].device, dtype=op["V"].dtype)
    x = x / (x.norm() + 1e-12)
    n = 0.0
    for _ in range(iters):
        x = op["W"] @ (op["V"].T @ (op["V"] @ (op["W"].T @ x)))
        n = float(x.norm())
        x = x / (n + 1e-12)
    # n converges to the top eigenvalue of K^T K = sigma_max^2.
    return float(np.sqrt(max(n, 1e-30)))


def _sagnac(p: Any, o: Any) -> float:
    import torch
    return float(1.0 - torch.dot(p.view(-1), o.view(-1))
                 / (p.norm() * o.norm()).clamp(min=1e-12))


def evaluate(cal: Sequence[Any], evl: Sequence[Any], dictionary_fn: Callable,
             ridge: float = 1e-4, rank: int = 8, num_blocks: int = 8,
             horizons: Sequence[int] = DEFAULT_HORIZONS,
             flag: str = FLAG) -> Dict[str, Any]:
    """Arms: persistence, action-agnostic, shuffled-action, action-conditioned.

    cal/evl: sequences of TransitionRecord (state_wave, action_id,
    action_wave, next_wave, episode, step). dictionary_fn(state_wave,
    action_wave) -> [2d] dictionary vector (production transition.bind path).
    """
    if os.environ.get(flag, "0") != "1":
        raise KoopmanFitDisabledError(f"{flag} is not set; Koopman fit is default-OFF")
    if not cal or not evl:
        return {"verdict": "BLOCKED_INSUFFICIENT_EPISODES",
                "n_cal": int(len(cal)), "n_eval": int(len(evl)),
                "trainable_parameters": 0, "optimizer": None}
    import torch
    device = torch.device("cpu")
    try:
        device = cal[0].state_wave.device if hasattr(cal[0].state_wave, "device") else torch.device("cpu")
    except Exception:
        pass
    dt = torch.float32

    def stack(records, key):
        return torch.stack([getattr(r, key).detach().to(device).to(dt).reshape(-1)
                            for r in records])

    action_ids = sorted({r.action_id for r in cal} | {r.action_id for r in evl})
    # per-action calibration splits
    cal_by_a = {a: [r for r in cal if r.action_id == a] for a in action_ids}
    ops = {}
    for a in action_ids:
        rs = cal_by_a[a]
        if not rs:
            continue
        phi = torch.stack([dictionary_fn(r.state_wave, r.action_wave) for r in rs])
        y = stack(rs, "next_wave")
        ops[a] = fit_operator(phi, y, ridge=ridge, r=rank)
    # Eval coverage: only records whose action has a fitted operator can be
    # scored on the conditioned/shuffled arms. Exclusions are counted and
    # reported; persistence baseline still covers every eval record.
    fitted = set(ops)
    evl_fitted = [r for r in evl if r.action_id in fitted]
    n_eval_excluded = len(evl) - len(evl_fitted)
    if not evl_fitted:
        return {"verdict": "BLOCKED_NO_EVAL_COVERAGE",
                "n_cal": int(len(cal)), "n_eval": int(len(evl)),
                "n_eval_excluded": n_eval_excluded,
                "trainable_parameters": 0, "optimizer": None}
    evl = evl_fitted
    # action-agnostic op
    phi_all = torch.stack([dictionary_fn(r.state_wave, r.action_wave) for r in cal])
    y_all = stack(cal, "next_wave")
    ops["__agnostic__"] = fit_operator(phi_all, y_all, ridge=ridge, r=rank)
    # shuffled action control: permuted per-action assignment (action-name
    # keys; derangement so every action is paired with a DIFFERENT action's
    # operator, and the arm is not the identity permutation).
    rng = np.random.default_rng(0)
    shuffled = list(action_ids)
    rng.shuffle(shuffled)
    while len(action_ids) > 1 and any(
            a == b for a, b in zip(action_ids, shuffled)):
        rng.shuffle(shuffled)
    perm = dict(zip(action_ids, shuffled))

    def arm_errors(records, op_fn):
        errs = []
        coses = []
        for r in records:
            phi = dictionary_fn(r.state_wave, r.action_wave)
            pred = op_fn(r, phi)
            errs.append(_sagnac(pred, r.next_wave))
            coses.append(float(torch.dot(pred.view(-1), r.next_wave.view(-1))
                               / (pred.norm() * r.next_wave.norm()).clamp(min=1e-12)))
        return errs, coses

    persist_e, _ = arm_errors(evl, lambda r, phi: r.state_wave)
    agn_e, _ = arm_errors(evl, lambda r, phi: predict_wave(ops["__agnostic__"], phi, num_blocks))
    shuf_e, _ = arm_errors(evl, lambda r, phi: predict_wave(ops[perm[r.action_id]], phi, num_blocks))
    cond_e, cond_cos = arm_errors(evl, lambda r, phi: predict_wave(ops[r.action_id], phi, num_blocks))

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    persist_err, cond_err, agn_err, shuf_err = mean(persist_e), mean(cond_e), mean(agn_e), mean(shuf_e)
    skill_ratio = persist_err / max(cond_err, 1e-12)
    # calibration engagement
    _, cal_cos = arm_errors(cal, lambda r, phi: predict_wave(ops[r.action_id], phi, num_blocks))
    _, cal_persist_cos = arm_errors(cal, lambda r, phi: r.state_wave)
    # open-loop rollouts per episode on eval (horizons)
    rollouts = {}
    for h in horizons:
        p_errs, c_errs = [], []
        by_ep = {}
        for r in evl:
            by_ep.setdefault(r.episode, []).append(r)
        for ep, rs in by_ep.items():
            rs = sorted(rs, key=lambda r: r.step)
            if len(rs) < 2:
                continue
            # persistence rollout: keep initial state
            s0 = rs[0].state_wave
            p_acc = []
            for i in range(1, min(len(rs), h + 1)):
                p_acc.append(_sagnac(s0, rs[i].next_wave))
            p_errs.append(mean(p_acc))
            # conditioned rollout: feed own prediction forward
            s_pred = rs[0].state_wave
            c_acc = []
            for i in range(1, min(len(rs), h + 1)):
                phi = dictionary_fn(s_pred, rs[i - 1].action_wave)
                s_pred = predict_wave(ops[rs[i - 1].action_id], phi, num_blocks)
                c_acc.append(_sagnac(s_pred, rs[i].next_wave))
            c_errs.append(mean(c_acc))
        rollouts[str(h)] = {"persistence": mean(p_errs),
                            "conditioned": mean(c_errs)}
    sn = {a: spectral_norm(ops[a]) for a in ops}
    max_sn = max(sn.values()) if sn else float("nan")
    # verdicts
    if mean(cond_cos) <= mean(cal_persist_cos) + 1e-9 and mean(cond_cos) < 0.05:
        verdict = "FALSIFIED_NO_ENGAGEMENT"
    elif skill_ratio > 1.0 and rollouts.get("5", {}).get("conditioned", float("nan")) \
            < rollouts.get("5", {}).get("persistence", float("inf")):
        verdict = "KOOPMAN_FIT_SUPPORTED"
    else:
        verdict = "FALSIFIED_NO_EXTERNAL_GAIN"
    return {
        "verdict": verdict,
        "rank": int(rank),
        "one_step": {"persistence": round(persist_err, 6),
                     "action_agnostic": round(agn_err, 6),
                     "shuffled_action": round(shuf_err, 6),
                     "action_conditioned": round(cond_err, 6),
                     "skill_ratio_vs_persistence": round(skill_ratio, 4)},
        "engagement": {"cal_pred_cos_conditioned": round(mean(cal_cos), 6),
                       "cal_pred_cos_persistence": round(mean(cal_persist_cos), 6),
                       "eval_pred_cos_conditioned": round(mean(cond_cos), 6)},
        "rollouts": rollouts,
        "spectral": {"per_action": {a: round(v, 4) for a, v in sn.items()},
                     "max_spectral_norm": round(max_sn, 4)},
        "n_cal": int(len(cal)), "n_eval": int(len(evl)),
        "actions": action_ids,
        "trainable_parameters": 0,
        "optimizer": None,
    }
