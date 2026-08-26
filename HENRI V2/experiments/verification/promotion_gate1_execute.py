"""Gate 1 execution carrier: few-shot online adaptation on the T0 stream.

Implements the frozen contract promotion_gate1_contract.json (v1):

  arms     R (real action-conditioned), S (shuffled-action derangement),
           A (action-agnostic shared), N (no-update frozen), P (persistence)
  budgets  [1, 2, 5, 10, 32]; primary n* = 32
  metrics  L_a(n) = mean episode means of (1 - cos(psi_hat_next, psi_next))
           I_a(n) = L_N(n) - L_a(n)
           DeltaI(n) = L_S(n) - L_R(n)
  CI       episode-level paired bootstrap, 10,000 resamples, percentile
  trend    Spearman rho(I_R(n), log n) >= 0.60 AND I_R(32) - I_R(1) >= 0.02
  split    lexicographic by episode_id; calibration = first K_cal episodes,
           evaluation = last K_eval (K_eval = 3, frozen), K_eval >= 2
  support  at n*=32 every eval action must have N_a >= 2 cal transitions,
           else BLOCKED_NO_EVAL_COVERAGE; low-shot budgets diagnostic-only
  verdict  BLOCKED_INFRA > BLOCKED_NO_EVAL_COVERAGE >
           FALSIFIED_NO_ENGAGEMENT > ENGAGED > ACTION_INFORMATION_GAIN >
           FEW_SHOT_SCALING

Online update (identical across R/S/A, frozen before measurement):
  factored operator O = V W^T, V in Stiefel(r) via Cholesky retraction,
  W [d, r] (d = dictionary dim = num_blocks*8 = 65536; matches the
  production koopman_fit factored solve, never [2d, r]). One pass over
  the first n calibration transitions of the action (ledger order). Per
  transition:
    phi = fused_intent(s, wave_a)                # [d]
    z = W^T phi;  lin = V z -> [blocks, 8]       # per-block L2 normalize
    loss = 1 - cos(pred, next)
    surprise gate: skip update if loss < THETA
    grad_W = 2 phi (z - V^T y)^T                 # [d, r]
    grad_V = 2 (lin - y) z^T                     # [d, r]
    V <- chol_retract(V - lr*grad_V)
    W <- W - lr*grad_W + noise, noise ~ N(0, sqrt(2*T*dt))

Hyperparameters (frozen): rank r=8, lr=0.1, T=0.001, dt=1.0
(noise_std = sqrt(2*T*dt) = 0.044721), THETA=0.02, seed=20260826,
bootstrap B=10000, K_eval=3.

Default-OFF: HENRI_GATE1_ONLINE_ADAPTATION=1 required. Requires the T0
stream (HENRI_TEMPORAL_LEDGER=1, HENRI_LEDGER_PAYLOADS=1) and the Carrier A
production action-wave manifest (--action-waves). The corpus is the T0
replay stream regenerated with the same seed/flags as Carrier T0 (seed
20260826, discovery, rounds 2); the action-wave manifest comes from the
Carrier A exporter (origin = live_planner_boundary) so the runner is
verdict-capable (no placeholder rings).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

FLAG = "HENRI_GATE1_ONLINE_ADAPTATION"
BUDGETS = [1, 2, 5, 10, 32]
PRIMARY_BUDGET = 32
ARMS = ["R", "S", "A", "N", "P"]
K_EVAL = 3
BOOTSTRAP_B = 10000
SEED = 20260826

# Frozen update hyperparameters (see module docstring).
RANK = 8
LR = 0.1
T = 0.001
DT = 1.0
NOISE_STD = math.sqrt(2.0 * T * DT)
THETA = 0.02

VERDICT_ORDER = [
    "BLOCKED_INFRA",
    "BLOCKED_NO_EVAL_COVERAGE",
    "FALSIFIED_NO_ENGAGEMENT",
    "ENGAGED",
    "ACTION_INFORMATION_GAIN",
    "FEW_SHOT_SCALING",
]


class Gate1DisabledError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Pure helpers (importable by the contract test without torch)
# ---------------------------------------------------------------------------

def make_derangement(ids: Sequence[str], seed: int) -> Dict[str, str]:
    """Fixed derangement over action ids (no fixed points)."""
    ids = list(ids)
    if len(ids) < 2:
        raise ValueError("derangement impossible with fewer than 2 ids")
    rng = np.random.default_rng(seed)
    perm = list(ids)
    for _ in range(10000):
        rng.shuffle(perm)
        if all(a != b for a, b in zip(ids, perm)):
            return dict(zip(ids, perm))
    raise RuntimeError("failed to build a derangement")


def lexicographic_split(episode_ids: Sequence[str], k_eval: int,
                        ) -> Tuple[List[str], List[str]]:
    """Calibration = first episodes, evaluation = last k_eval (lex order)."""
    ids = sorted(episode_ids)
    if len(ids) < k_eval + 1 or k_eval < 1:
        raise ValueError(
            f"need at least {k_eval + 1} episodes for k_eval={k_eval}, "
            f"got {len(ids)}")
    eval_ids = ids[-k_eval:]
    cal_ids = ids[:-k_eval]
    return cal_ids, eval_ids


def delta_i(l_s: float, l_r: float) -> float:
    """Frozen algebra: DeltaI = L_S - L_R (positive = real beats shuffled)."""
    return l_s - l_r


def spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation with average-rank tie handling."""
    def ranks(v: Sequence[float]) -> List[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    rx = ranks(list(xs))
    ry = ranks(list(ys))
    n = len(xs)
    if n < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return 0.0
    # Exact Spearman: Pearson correlation of the average-rank vectors.
    # (The shortcut 1 - 6*sum(d^2)/(n^3 - n) is only valid tie-free and
    # was FALSIFIED by the contract test on the reversed case.)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0.0 or deny == 0.0:
        return 0.0
    return float(num / (denx * deny))


def bootstrap_percentile(paired: List[Tuple[float, float]], b: int = 10000,
                         seed: int = 0) -> Dict[str, float]:
    """Episode-level PAIRED bootstrap percentile interval of d_i = x_i - y_i."""
    rng = np.random.default_rng(seed)
    n = len(paired)
    if n == 0:
        return {"mean": float("nan"), "lb": float("nan"),
                "ub": float("nan")}
    diffs = np.array([x - y for x, y in paired], dtype=np.float64)
    draws = np.empty(b, dtype=np.float64)
    for k in range(b):
        idx = rng.integers(0, n, size=n)
        draws[k] = diffs[idx].mean()
    return {
        "mean": float(draws.mean()),
        "lb": float(np.percentile(draws, 2.5)),
        "ub": float(np.percentile(draws, 97.5)),
    }


def verdict_for(delta32: Dict[str, float], i_r: Dict[int, float],
                engaged: bool) -> str:
    """Frozen verdict mapping (pure)."""
    lb = delta32.get("lb", float("nan"))
    if not engaged:
        return "FALSIFIED_NO_ENGAGEMENT"
    if delta32.get("mean", float("-inf")) <= 0.0 or lb <= 0.0:
        return "ENGAGED"
    n_log = [math.log(n) for n in BUDGETS]
    rho = spearman_rho(n_log, [i_r[n] for n in BUDGETS])
    if rho >= 0.60 and i_r[PRIMARY_BUDGET] - i_r[BUDGETS[0]] >= 0.02:
        return "FEW_SHOT_SCALING"
    return "ACTION_INFORMATION_GAIN"


# ---------------------------------------------------------------------------
# Online operator (torch)
# ---------------------------------------------------------------------------

class FactoredOperator:
    """O = V W^T, V on Stiefel(r); Cholesky retraction; SGLD update."""

    def __init__(self, d: int, r: int, rank_seed: int, device: str = "cpu"):
        import torch
        self.torch = torch
        g = torch.Generator(device="cpu").manual_seed(rank_seed)
        q, _ = torch.linalg.qr(torch.randn(d, r, generator=g, device="cpu"))
        self.V = q.to(torch.float32)
        # W maps dictionary space [d] -> latent [r]; d = num_blocks*8
        # (65536) matching the production factored solve (koopman_fit:
        # W = phi^T U S, phi dim = d). NOT [2d, r].
        self.W = torch.zeros(d, r, dtype=torch.float32)
        self.updates = 0

    @staticmethod
    def _chol_retract(A):
        """V = A (A^T A)^{-1/2} via Cholesky (L L^T = A^T A)."""
        import torch
        M = A.T @ A
        M = M + 1e-6 * torch.eye(M.shape[0], dtype=M.dtype, device=M.device)
        L = torch.linalg.cholesky(M)
        # V = A L^{-T}
        X = torch.linalg.solve_triangular(L, A.T, upper=False)  # L X = A^T
        return X.T

    def predict(self, phi):
        torch = self.torch
        z = self.W.T @ phi
        lin = self.V @ z
        return lin, z

    def update(self, phi, y, num_blocks: int, rng) -> bool:
        """One SGLD step on (phi, y). Returns True if applied (surprise gate)."""
        torch = self.torch
        if phi.shape[0] != self.W.shape[0]:
            raise RuntimeError(
                f"operator input dim {phi.shape[0]} != W rows {self.W.shape[0]} "
                f"(dictionary dim mismatch; W must be [d, r])")
        lin, z = self.predict(phi)
        pred = lin.view(num_blocks, -1)
        nrm = pred.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        pred = pred / nrm
        yv = y.view(num_blocks, -1)
        cos = (pred * yv).sum(dim=-1) / (
            yv.norm(dim=-1, keepdim=True).clamp(min=1e-12).squeeze(-1)
            + 1e-12)
        loss = float(1.0 - cos.mean().clamp(max=1.0 - 1e-9))
        if loss < THETA:
            return False  # surprise gate
        yflat = y.reshape(-1)
        grad_w = 2.0 * torch.outer(phi, (z - self.V.T @ yflat))
        grad_v = 2.0 * torch.outer((lin - yflat), z)
        v_new = self._chol_retract(self.V - LR * grad_v)
        noise = torch.randn(self.W.shape, generator=rng, dtype=torch.float32)
        w_new = self.W - LR * grad_w + NOISE_STD * noise
        self.V = v_new
        self.W = w_new
        self.updates += 1
        return True


# ---------------------------------------------------------------------------
# Corpus and dictionary
# ---------------------------------------------------------------------------

def fused_intent(state_wave, action_wave, num_blocks: int = 8192):
    """Fused-intent dictionary: real||imag of circular-convolution binding
    (production LowRankCoupledTransition.bind geometry), CPU."""
    import torch
    s = torch.complex(state_wave[..., :4], state_wave[..., 4:])
    a = torch.complex(action_wave[..., :4], action_wave[..., 4:])
    bound = torch.fft.ifft(torch.fft.fft(s, dim=-1) * torch.fft.fft(a, dim=-1),
                           dim=-1)
    return torch.cat([bound.real.reshape(-1), bound.imag.reshape(-1)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--payload-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--action-waves", required=True)
    ap.add_argument("--num-blocks", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--eval-k", type=int, default=K_EVAL)
    args = ap.parse_args()

    if os.environ.get(FLAG, "0") != "1":
        print(json.dumps({"verdict": "BLOCKED_INFRA",
                          "reason": f"{FLAG}=1 required"}))
        return 2

    import torch
    torch.manual_seed(args.seed)

    try:
        from koopman_corpus_runner import validate_action_wave_manifest
        from ledger_payload_store import LedgerPayloadStore
        from koopman_identifiability import load_corpus
    except Exception as exc:
        print(json.dumps({"verdict": "BLOCKED_INFRA",
                          "reason": f"import failed: {exc}"}))
        return 2

    # Production action waves (Carrier A manifest; fail-closed).
    aw_map, aw_err = validate_action_wave_manifest(
        args.action_waves, args.num_blocks)
    if aw_err is not None:
        print(json.dumps({"verdict": "BLOCKED_INFRA",
                          "reason": f"action-wave manifest: {aw_err}"}))
        return 2

    try:
        store = LedgerPayloadStore(args.payload_root,
                                   flag="HENRI_LEDGER_PAYLOADS")
    except Exception as exc:
        print(json.dumps({"verdict": "BLOCKED_INFRA",
                          "reason": f"payload store: {exc}"}))
        return 2

    # Production lift (HENRIVisionEncoder, CPU, production basis).
    try:
        from arc_spatial_basis import resolve_spatial_basis
        from henri_vision_encoder import HENRIVisionEncoder
        basis_kind, bg_mask = resolve_spatial_basis()
        tokenizer = HENRIVisionEncoder(
            d_model=65536, k_blocks=args.num_blocks, device="cpu",
            spatial_basis_kind=basis_kind, bg_mask=bg_mask)
    except Exception as exc:
        print(json.dumps({"verdict": "BLOCKED_INFRA",
                          "reason": f"encoder init: {exc}"}))
        return 2

    def lift(grid):
        w = tokenizer.encode_spatial_grid(grid).squeeze(0)
        return w.detach().cpu().to(torch.float32)

    try:
        records, stats = load_corpus(
            args.ledger, store, lift, aw_map,
            dedupe_continuity=True, flag=FLAG)
    except Exception as exc:
        print(json.dumps({"verdict": "BLOCKED_INFRA",
                          "reason": f"corpus load: {exc}"}))
        return 2
    if not records:
        print(json.dumps({"verdict": "BLOCKED_INFRA",
                          "reason": "empty corpus",
                          "stats": stats}))
        return 2
    if stats.get("continuity_breaks", 0):
        print(json.dumps({"verdict": "BLOCKED_INFRA",
                          "reason": "chain continuity broken",
                          "stats": stats}))
        return 2
    ledger_sha = hashlib.sha256(
        Path(args.ledger).read_bytes()).hexdigest()
    aw_sha = hashlib.sha256(
        Path(args.action_waves).read_bytes()).hexdigest()

    # Lexicographic episode split (frozen rule).
    ep_ids = sorted({r.episode for r in records})
    try:
        cal_ids, eval_ids = lexicographic_split(ep_ids, args.eval_k)
    except ValueError as exc:
        print(json.dumps({"verdict": "BLOCKED_INFRA",
                          "reason": f"split: {exc}"}))
        return 2
    cal = [r for r in records if r.episode in cal_ids]
    evl = [r for r in records if r.episode in eval_ids]
    if len(cal) == 0 or len(evl) == 0:
        print(json.dumps({"verdict": "BLOCKED_INFRA",
                          "reason": "empty cal or eval partition"}))
        return 2

    # Per-action calibration slices (ledger order preserved).
    by_a_cal: Dict[str, List] = {}
    for r in cal:
        by_a_cal.setdefault(r.action_id, []).append(r)
    eval_actions = sorted({r.action_id for r in evl})
    cal_support = {a: len(by_a_cal.get(a, [])) for a in eval_actions}
    support_ok = all(cal_support[a] >= 2 for a in eval_actions)

    # Derangement for S arm.
    action_ids = sorted({r.action_id for r in records})
    try:
        perm = make_derangement(action_ids, args.seed)
    except Exception as exc:
        print(json.dumps({"verdict": "BLOCKED_NO_EVAL_COVERAGE",
                          "reason": f"derangement: {exc}"}))
        return 2

    d = args.num_blocks * 8

    def _run_budget(n: int) -> Dict[str, Any]:
        """Run all arms at budget n; return per-episode loss dicts."""
        # R: per-action operators on first n cal transitions of each action.
        ops_r = {}
        updates_r = 0
        for a in eval_actions:
            rs = by_a_cal.get(a, [])[:n]
            op = FactoredOperator(d, RANK, args.seed + 1)
            g = torch.Generator(device="cpu").manual_seed(args.seed)
            for r in rs:
                phi = fused_intent(r.state_wave, r.action_wave, args.num_blocks)
                if op.update(phi, r.next_wave, args.num_blocks, g):
                    updates_r += 1
            ops_r[a] = op
        # S: operator for a trained on perm(a) transitions (derangement).
        ops_s = {}
        updates_s = 0
        for a in eval_actions:
            src = perm.get(a, a)
            rs = by_a_cal.get(src, [])[:n]
            op = FactoredOperator(d, RANK, args.seed + 1)
            g = torch.Generator(device="cpu").manual_seed(args.seed)
            for r in rs:
                phi = fused_intent(r.state_wave, r.action_wave, args.num_blocks)
                if op.update(phi, r.next_wave, args.num_blocks, g):
                    updates_s += 1
            ops_s[a] = op
        # A: single shared operator on first n cal transitions per action.
        op_a = FactoredOperator(d, RANK, args.seed + 1)
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        updates_a = 0
        for a in eval_actions:
            for r in by_a_cal.get(a, [])[:n]:
                phi = fused_intent(r.state_wave, r.action_wave, args.num_blocks)
                if op_a.update(phi, r.next_wave, args.num_blocks, g):
                    updates_a += 1

        def loss_for(pred_fn) -> Dict[str, List[float]]:
            by_ep: Dict[str, List[float]] = {}
            for r in evl:
                pred = pred_fn(r)
                cosv = float((pred * r.next_wave).sum() / (
                    pred.norm() * r.next_wave.norm()).clamp(min=1e-12))
                by_ep.setdefault(r.episode, []).append(1.0 - cosv)
            return {ep: float(np.mean(vs)) for ep, vs in by_ep.items()}

        ep_r = loss_for(lambda r: _predict(ops_r[r.action_id], r, args))
        ep_s = loss_for(lambda r: _predict(ops_s[r.action_id], r, args))
        ep_a = loss_for(lambda r: _predict(op_a, r, args))
        # N: frozen-at-init operator, ZERO updates. Predicts the zero
        # vector under the frozen loss -> L_N = 1.0 by construction.
        # I_a = L_N - L_a measures improvement over the unadapted
        # predictor; DeltaI = L_S - L_R is baseline-independent.
        ops_n = {a: FactoredOperator(d, RANK, args.seed + 1)
                 for a in eval_actions}
        ep_n = loss_for(lambda r: _predict(ops_n[r.action_id], r, args))
        ep_p = loss_for(lambda r: r.state_wave)
        return {
            "R": ep_r, "S": ep_s, "A": ep_a, "N": ep_n, "P": ep_p,
            "updates": {"R": updates_r, "S": updates_s, "A": updates_a},
        }

    def _predict(op, r, args):
        phi = fused_intent(r.state_wave, r.action_wave, args.num_blocks)
        lin, _ = op.predict(phi)
        pred = lin.view(args.num_blocks, -1)
        return pred / pred.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    results: Dict[str, Any] = {}
    ep_list = sorted(eval_ids)
    for n in BUDGETS:
        out = _run_budget(n)
        arm_ep = {a: {ep: out[a].get(ep, float("nan")) for ep in ep_list}
                  for a in ARMS}
        def agg(a):
            vals = [arm_ep[a][ep] for ep in ep_list if not math.isnan(arm_ep[a][ep])]
            return float(np.mean(vals)) if vals else float("nan")
        L = {a: agg(a) for a in ARMS}
        # Paired episode bootstrap across arms on the SAME eval episodes.
        paired_sr = [(arm_ep["S"][ep], arm_ep["R"][ep]) for ep in ep_list
                     if not math.isnan(arm_ep["S"][ep])
                     and not math.isnan(arm_ep["R"][ep])]
        paired_nr = [(arm_ep["N"][ep], arm_ep["R"][ep]) for ep in ep_list
                     if not math.isnan(arm_ep["N"][ep])
                     and not math.isnan(arm_ep["R"][ep])]
        ci_sr = bootstrap_percentile(paired_sr, BOOTSTRAP_B, args.seed)
        ci_nr = bootstrap_percentile(paired_nr, BOOTSTRAP_B, args.seed)
        results[n] = {
            "L": {a: round(L[a], 6) for a in ARMS},
            "I": {"R": round(L["N"] - L["R"], 6),
                  "S": round(L["N"] - L["S"], 6),
                  "A": round(L["N"] - L["A"], 6)},
            "DeltaI": round(delta_i(L["S"], L["R"]), 6),
            "DeltaI_ci": {k: round(v, 6) for k, v in ci_sr.items()},
            "I_R_ci": {k: round(v, 6) for k, v in ci_nr.items()},
            "updates": out["updates"],
            "n_eval_records": int(len(evl)),
            "per_episode": {ep: {a: round(arm_ep[a][ep], 6) for a in ARMS}
                            for ep in ep_list},
        }

    # Primary endpoint (frozen n* = 32).
    primary = results[PRIMARY_BUDGET]
    engaged = sum(primary["updates"].values()) > 0
    i_r = {n: results[n]["I"]["R"] for n in BUDGETS}
    n_log = [math.log(n) for n in BUDGETS]
    rho = spearman_rho(n_log, [i_r[n] for n in BUDGETS])
    trend_ok = (rho >= 0.60
                and i_r[PRIMARY_BUDGET] - i_r[BUDGETS[0]] >= 0.02)
    delta32 = primary["DeltaI_ci"]

    verdict = "BLOCKED_NO_EVAL_COVERAGE"
    if support_ok and engaged:
        verdict = verdict_for(delta32, i_r, engaged)
    elif not engaged:
        verdict = "FALSIFIED_NO_ENGAGEMENT" if support_ok \
            else "BLOCKED_NO_EVAL_COVERAGE"

    # Kill experiments (pre-registered).
    kills = {
        "kill1_deltaI_leq_0_ci_includes_0": (
            primary["DeltaI"] <= 0.0 or delta32["lb"] <= 0.0),
        "kill2_I_R_leq_0_lb_gt_0": (
            primary["I"]["R"] <= 0.0 and primary["I_R_ci"]["lb"] > 0.0),
        "kill3_deltaI_lt_0_lb_gt_0": (
            primary["DeltaI"] < 0.0 and delta32["lb"] > 0.0),
    }

    result = {
        "contract_id": "promotion_gate1_few_shot_scaling_v1",
        "verdict": verdict,
        "head": os.environ.get("HENRI_COMMIT_SHA", "unknown"),
        "seed": args.seed,
        "hyperparameters": {"rank": RANK, "lr": LR, "T": T, "dt": DT,
                            "noise_std": round(NOISE_STD, 6),
                            "theta_surprise": THETA, "k_eval": args.eval_k,
                            "bootstrap_b": BOOTSTRAP_B},
        "N_baseline_note": "L_N = 1.0 by construction (frozen operator at init predicts the zero vector under the frozen loss); I_a = L_N - L_a measures improvement over the unadapted predictor; DeltaI = L_S - L_R is baseline-independent",
        "corpus": {"n_records": len(records), "n_cal": len(cal),
                   "n_eval": len(evl), "n_episodes": len(ep_ids),
                   "ledger_sha256": ledger_sha,
                   "action_waves_sha256": aw_sha,
                   "stats": stats, "cal_support": cal_support,
                   "support_ok_at_primary": bool(support_ok)},
        "split_manifest": {"ordering": "lexicographic_by_episode_id",
                           "cal_episodes": cal_ids,
                           "eval_episodes": eval_ids,
                           "record_counts_by_episode": {
                               ep: sum(1 for r in records if r.episode == ep)
                               for ep in sorted(ep_ids)}},
        "derangement": perm,
        "budgets": {str(n): results[n] for n in BUDGETS},
        "scaling": {"spearman_rho_I_R_vs_log_n": round(rho, 4),
                    "trend_ok": bool(trend_ok),
                    "I_R_32_minus_I_R_1": round(
                        i_r[PRIMARY_BUDGET] - i_r[BUDGETS[0]], 6),
                    "I_R_by_budget": {str(n): round(v, 6)
                                      for n, v in i_r.items()}},
        "primary": {"budget": PRIMARY_BUDGET,
                    "DeltaI": primary["DeltaI"],
                    "DeltaI_ci": delta32,
                    "I_R": primary["I"]["R"],
                    "I_R_ci": primary["I_R_ci"],
                    "updates": primary["updates"],
                    "engaged": bool(engaged)},
        "kill_experiments": kills,
        "verdict_precedence": VERDICT_ORDER,
        "evidence_labels": {"metrics": "OBSERVED", "delta_I": "DERIVED",
                            "verdict": "per precedence"},
        "corpus_consult": "BLOCKED_AUTH (NotebookLM ClientAuthenticationError 2026-08-26; user-authorized to proceed)",
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True),
                              encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "DeltaI_32": primary["DeltaI"],
        "DeltaI_32_ci": delta32,
        "I_R_32": primary["I"]["R"],
        "spearman_rho": round(rho, 4),
        "trend_ok": trend_ok,
        "engaged": bool(engaged),
        "support_ok": bool(support_ok),
        "kills": kills,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
