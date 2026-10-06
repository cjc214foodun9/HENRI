"""HENRI gate runner, VERSION 2. Additive. gates.py is NOT modified.

Why a new file: a committed receipt's number belongs to the exact code state it
names. Editing gates.py in place would silently re-label every prior G-U4
receipt. So the fixes live here under new gate ids, and gates.py stays frozen.

C1/C2/C3 fixes (from the independent review, findings.json):
  C1  non-finite R^2 is REJECTED -> BLOCKED. The old code clamped with
      max(0.0, min(1.0, r2)), and min(1.0, nan) == 1.0 in Python, so a broken
      estimator silently read as a ceiling PASS. No clamp here. No guess.
  C2  the verdict branches on status/estimator_sane, never on the raw value.
      A BLOCKED arm can no longer be counted as a pass.
  C3  a per-system gate CANNOT know the multi-seed denominator, so the
      aggregation rule belongs to the DRIVER. This module does not implement
      one; the caller must apply the pass rule over the receipts and must not
      hardcode a literal count. Disclosed, not silently claimed.
  NOTE: the estimator is NOT byte-identical to v1. The v1 exception path returns
      0.0; this one returns nan -> BLOCKED. Same happy path, different failure
      path, and v2 drops v1's provenance fields. Compare verdicts, not raw R2.
  F4  the negative control needs a MARGIN, not a 1-seed cutoff.
  F5  the gate's own shuffled-pairing control enters the verdict.
  F11 an explicit outcome vocabulary: PASS / FAIL / VACUOUS / BLOCKED. A control
      that passes yields VACUOUS, never a claim of FALSIFIED.
"""
from __future__ import annotations

import math

import torch

from . import substrate as sub

PASS, FAIL, VACUOUS, BLOCKED = "PASS", "FAIL", "VACUOUS", "BLOCKED"


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _verdict_v2(real_ok: bool, ctl_ok: bool, control_name: str,
                margin_ok: bool = True) -> tuple[str, str]:
    """Combine the real result with the negative control. No raw-value path."""
    if not real_ok:
        return FAIL, "real mechanism did not meet the bound"
    if ctl_ok:
        return VACUOUS, (f"control '{control_name}' also passed: the gate cannot "
                         "distinguish the mechanism from its absence")
    if not margin_ok:
        return VACUOUS, (f"control '{control_name}' failed, but WITHOUT the "
                         "pre-registered margin: the separation is inside noise")
    return PASS, f"real passes; control '{control_name}' fails with margin"


# --------------------------------------------------------------------- KSG MI
def ksg_mi(X: torch.Tensor, Y: torch.Tensor, k: int = 5) -> float:
    """KSG (Kraskov-Stoegbauer-Grassberger) MI estimator, continuous-continuous.

    I(X;Y) = psi(k) + psi(N) - <psi(n_x + 1) + psi(n_y + 1)>

    Distances are Chebyshev (max-norm) over standardized coordinates; counts are
    STRICTLY less than the k-th neighbour radius, self excluded. k=5 and
    standardization are pinned defaults; change them only in a new pre-registration.

    Returns nan when N is too small for the draw. Callers must treat nan as
    BLOCKED, never as a value.
    """
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64)
    if X.dim() == 1:
        X = X[:, None]
    if Y.dim() == 1:
        Y = Y[:, None]
    N = X.shape[0]
    if Y.shape[0] != N or N <= k + 1:
        return float("nan")
    X = (X - X.mean(0)) / X.std(0).clamp_min(1e-12)
    Y = (Y - Y.mean(0)) / Y.std(0).clamp_min(1e-12)
    Z = torch.cat([X, Y], dim=-1)
    dz = torch.cdist(Z, Z, p=float("inf"))
    dx = torch.cdist(X, X, p=float("inf"))
    dy = torch.cdist(Y, Y, p=float("inf"))
    dz.fill_diagonal_(float("inf"))
    dx.fill_diagonal_(float("inf"))
    dy.fill_diagonal_(float("inf"))
    eps_k = dz.topk(k, largest=False).values[:, -1]          # [N]
    nx = (dx < eps_k[:, None]).sum(1).double()
    ny = (dy < eps_k[:, None]).sum(1).double()
    psi = torch.special.digamma
    val = (psi(torch.tensor(float(k))) + psi(torch.tensor(float(N)))
           - (psi(nx + 1.0) + psi(ny + 1.0)).mean())
    return float(val)


def ksg_mi_selftest(n: int = 512, rho: float = 0.6, seed: int = 20261005) -> dict:
    """Prove the estimator is not vacuous before any gate uses it.

    D-C1 (self-caught): the first draft compared a 4-D joint estimate against the
    BIVARIATE analytic MI, -0.5*log(1-rho^2). That is a unit error, not an
    estimator error. MI is additive over independent component pairs, so the
    correct multi-D value is d * (-0.5*log(1-rho^2)). The primary check here is
    1-D, where the analytic value is exact and unambiguous.

    Controls (all three must hold for estimator_sane):
      positive : 1-D Y = rho*X + sqrt(1-rho^2)*noise; |mi - analytic| <= 0.10
      negative : 1-D Y independent of X;                mi <= 0.10
      monotone : mi increases with rho
    """
    g = torch.Generator().manual_seed(seed)
    x1 = torch.randn(n, 1, generator=g)
    e1 = torch.randn(n, 1, generator=g)
    y_pos = rho * x1 + math.sqrt(1.0 - rho ** 2) * e1
    y_neg = torch.randn(n, 1, generator=g)

    mi_pos = ksg_mi(x1, y_pos)
    mi_neg = ksg_mi(x1, y_neg)
    analytic = -0.5 * math.log(1.0 - rho ** 2)

    gamma = torch.Generator().manual_seed(seed + 1)
    xs = torch.randn(512, 1, generator=gamma)
    by_rho = {}
    for r in (0.0, 0.3, 0.6, 0.9):
        en = torch.randn(512, 1, generator=gamma)
        by_rho[r] = ksg_mi(xs, r * xs + math.sqrt(max(0.0, 1.0 - r ** 2)) * en)
    mono = all(by_rho[a] <= by_rho[b] + 0.05
               for a, b in ((0.0, 0.3), (0.3, 0.6), (0.6, 0.9)))

    ok = (_finite(mi_pos) and _finite(mi_neg)
          and abs(mi_pos - analytic) <= 0.10 and mi_neg <= 0.10 and mono)
    return {"mi_pos_1d": mi_pos, "mi_neg_1d": mi_neg, "analytic_1d": analytic,
            "abs_err": abs(mi_pos - analytic) if _finite(mi_pos) else float("nan"),
            "monotone": mono, "mi_by_rho": by_rho,
            "estimator_sane": bool(ok), "n": n, "k": 5, "rho": rho,
            "note": "1-D primary; multi-D analytic is d-fold additive"}


# ------------------------------------------------------- gate_u4_retention_v2
def gate_u4_retention_v2(system, tokenizer, n_texts: int = 1024, ridge: float = 1.0,
                         n_comp: int = 32, bound: float = 0.95,
                         margin: float = 0.10) -> dict:
    """G-U4 v2: same happy-path estimator, fixed DECISION logic.

    Estimator parity is on the happy path only. The v1 exception path returns
    0.0; this one returns nan -> BLOCKED (C1). Receipt fields also differ, so
    compare verdicts, not raw R2, across versions.
    What changes here is how the number becomes a verdict: no clamp, non-finite
    -> BLOCKED, verdict keyed on sanity, and a margin requirement.

    C3 limit (disclosed): a single-system gate cannot know the multi-seed
    denominator. That aggregation belongs to the driver; this gate does not do
    it and does not pretend to.
    """
    texts = [f"retrieval transfer result {i} on the ladder" for i in range(n_texts)]
    with torch.no_grad():
        psi = torch.stack([system.wave_of(t, tokenizer) for t in texts])
        prof = (psi.abs() ** 2).reshape(n_texts, sub.N_SLOTS, -1).sum(-1)
        prof = prof / prof.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        _, tokens = system.decoder.encode_wave(psi)
        content = tokens.mean(dim=1).float()
        routing = tokens.mean(dim=-1).float()
        base = torch.cat([content, routing], dim=-1)
        F = torch.cat([base, base ** 2], dim=-1)

    half = n_texts // 2
    tr, te = slice(0, half), slice(half, n_texts)
    Ytr, Yte = prof[tr], prof[te]

    def r2(feats):
        mu = feats[tr].mean(0, keepdim=True)
        sd = feats[tr].std(0, keepdim=True).clamp_min(1e-6)
        Xtr, Xte = (feats[tr] - mu) / sd, (feats[te] - mu) / sd
        ybar = Ytr.mean(0, keepdim=True)
        Ytrc = Ytr - ybar
        try:
            _, _, vh = torch.linalg.svd(Xtr, full_matrices=False)
        except Exception:                          # noqa: BLE001
            return float("nan")                    # C1: NOT 0.0
        k = max(1, min(int(n_comp), vh.shape[0]))
        P = vh[:k]
        Ztr, Zte = Xtr @ P.T, Xte @ P.T
        gram = Ztr.T @ Ztr + ridge * torch.eye(k)
        w = torch.linalg.solve(gram, Ztr.T @ Ytrc)
        pred = Zte @ w + ybar
        ss_res = ((Yte - pred) ** 2).sum()
        ss_tot = ((Yte - Yte.mean(0)) ** 2).sum().clamp_min(1e-12)
        return float(1.0 - ss_res / ss_tot)

    value = r2(F)                                  # C1: no clamp
    g = torch.Generator().manual_seed(7)
    ctl_val = r2(F[torch.randperm(n_texts, generator=g)])
    pos_val = r2(prof.clone())

    sane = _finite(pos_val) and pos_val >= 0.99
    if not sane:
        status, why = BLOCKED, (f"positive control R2={pos_val!r} (<0.99 or "
                                "non-finite): estimator broken, no mechanism verdict")
    elif not _finite(value) or not _finite(ctl_val):
        status, why = BLOCKED, (f"non-finite estimator output value={value!r} "
                                f"control={ctl_val!r}")
    else:
        real_ok = value >= bound
        ctl_ok = ctl_val >= bound
        margin_ok = (value - ctl_val) >= margin
        status, why = _verdict_v2(real_ok, ctl_ok, "shuffled_pairing", margin_ok)
    return {"id": "G-U4-v2", "metric": "heldout_slot_profile_r2", "value": value,
            "op": ">=", "bound": bound, "margin": margin, "control_value": ctl_val,
            "positive_control": pos_val, "estimator_sane": bool(sane),
            "n_texts": n_texts, "status": status, "why": why,
            "supersedes": "G-U4 (gates.py); decision logic fixed; happy-path r2() same",
            "fixes": ["C1 no clamp; non-finite->BLOCKED (Happy-path r2() reused)",
                      "C2 verdict keyed on estimator_sane, never the raw value",
                      "C3 NOT implemented here: multi-seed aggregation is a driver duty",
                      "F4 margin on the control", "F5 shuffled control in the verdict",
                      "F11 explicit PASS/FAIL/VACUOUS/BLOCKED vocabulary"]}


# ------------------------------------------- gate_u5_perslot_mi (discriminating)
def gate_u5_perslot_mi(system, tokenizer, n_texts: int = 512, n_comp: int = 32,
                       k: int = 5, seed: int = 20261005) -> dict:
    """The discriminating metric G-U4 cannot be.

    Mechanism (confirmed by arithmetic): resolve_n_mem gives n_mem = d_model/d_k,
    so n_mem * d_k == d_model in BOTH the old rule and Spec B. Aggregate key rank
    is therefore identical, and a linear readout of the AGGREGATE structure (G-U4)
    cannot separate the arms. Only the SLOT/WIDTH split differs.

    Two statistics are measured on the pooled macro-tokens:
      collapse : mean pairwise |cos| -- the Gap#2/D149 defect. d_k=4 pooled
                 features measured 0.9553 (near-identical). Higher = worse.
      mi       : KSG MI between a TRAIN-only PCA projection of the pooled
                 features and the 4-slot energy target. Higher = more retained.

    Pre-registered separation rule: the arms are DISCRIMINATED only when
      |collapse_a - collapse_b| >= 0.20
    AND the KSG estimator self-test passes (else BLOCKED).

    This gate is CPU-only and takes one system at a time. The caller runs BOTH
    configs and compares the two receipts; no verdict is taken from one arm.
    """
    st = ksg_mi_selftest(seed=seed)
    texts = [f"retrieval transfer result {i} on the ladder" for i in range(n_texts)]
    with torch.no_grad():
        psi = torch.stack([system.wave_of(t, tokenizer) for t in texts])
        prof = (psi.abs() ** 2).reshape(n_texts, sub.N_SLOTS, -1).sum(-1)
        prof = prof / prof.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        _, tokens = system.decoder.encode_wave(psi)
        feat = tokens.mean(dim=1).float()                      # [N, d_model]

    # D-C2 (self-caught): the first draft built the pairwise list as
    # torch.cat([fn[i:i+1] @ fn[i+1:].T ...]). Each term has a SHRINKING second
    # dimension ([1,255], [1,254], ...), so torch.cat raised
    # "Expected size 255 but got size 254" and every arm returned BLOCKED.
    # Compute the Gram matrix instead; the off-diagonal mean is the same
    # statistic without an O(N^2) list.
    fn = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    m = int(min(n_texts, 256))
    fnm = fn[:m]
    gram = (fnm @ fnm.transpose(0, 1)).abs()
    off_sum = float(gram.sum() - torch.diagonal(gram).sum())
    collapse = off_sum / max(1, m * (m - 1))

    half = n_texts // 2
    mu = feat[:half].mean(0, keepdim=True)
    sd = feat[:half].std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (feat[:half] - mu) / sd
    try:
        _, _, vh = torch.linalg.svd(Xtr, full_matrices=False)
        P = vh[:max(1, min(n_comp, vh.shape[0]))]
        proj = ((feat - mu) / sd) @ P.T
    except Exception:                                  # noqa: BLE001
        proj = None
    mi = ksg_mi(proj, prof, k=k) if proj is not None else float("nan")

    # R-F2 (review fix): the MI was computed and returned but NEVER thresholded,
    # so the "retention measure" could not affect any decision. Add the shuffle
    # control the first version lacked, and BLOCK on a saturated MI.
    g = torch.Generator().manual_seed(seed + 91)
    prof_shuf = prof[torch.randperm(n_texts, generator=g)]
    mi_shuf = ksg_mi(proj, prof_shuf, k=k) if proj is not None else float("nan")

    if not st["estimator_sane"]:
        status, why = BLOCKED, "KSG estimator self-test failed; no MI verdict"
    elif not (_finite(mi) and _finite(mi_shuf) and _finite(collapse)):
        status, why = BLOCKED, f"non-finite output mi={mi!r} mi_shuf={mi_shuf!r}"
    elif mi_shuf > 0.15:
        status, why = BLOCKED, (f"MI shuffle control failed (mi_shuf={mi_shuf:.4f} "
                                "> 0.15): the MI statistic is saturated/leaky")
    else:
        status, why = "MEASURED", ("single arm; MI shuffle control passes. The "
                                   "pre-registered separation rule is on collapse; "
                                   "MI is reported and blocked-if-invalid, not gated")
    return {"id": "G-U5", "metric": "perslot_collapse_and_ksg_mi",
            "collapse_mean_pairwise_abs_cos": collapse, "ksg_mi": mi,
            "ksg_mi_shuffled": mi_shuf,
            "n_texts": n_texts, "n_comp": n_comp, "k": k,
            "estimator_selftest": st, "status": status, "why": why,
            "discrimination_rule": "arms DISCRIMINATED iff |dcollapse| >= 0.20 and both self-tests pass",
            "mi_role": "reported + shuffle-gated; NOT part of the separation rule",
            "note": "aggregate rank is identical across arms by construction; this "
                    "metric targets the per-slot/per-width structure instead"}
