"""Causal Emergence telemetry for the HENRI live wave trajectory.

Packet: experiments/verification/ce_telemetry_packet_20260821.md (HENRI-CLASS47).
Diagnostic sidecar ONLY. Never influences policy, never gates scores, never
mutates weights. Returns None (with reason) instead of raising on bad input.

Formulation (Hoel 2013; Pigozzi & Levin 2026 arXiv:2605.06746):
    EI(X) = (1/n) * sum_i KL( TPM[i, :] || pbar ),  pbar = column means
    (uniform do-intervention on X_t; empirical TPM is a DERIVED approximation).
    CE = EI(macro) - EI(micro) in bits.

Micro state: thin-SVD PCA to r coordinates, sign-quantized (K=2) -> 2^r states.
Macro state: Hoel causal coarse-graining - cluster the VISITED states by their
    empirical TPM-row similarity; macro TPM = count-weighted row means. Pure
    noise rows are all the same random row -> macro EI ~= micro EI (CE ~ 0);
    structured rows separate -> CE != 0 possible.

Amended 2026-08-21 pre-CUDA (T2 smoke): r=8/K=3 terciles over T=64 gave
6,561 states with nearly unique transitions (micro EI collapsed to 1e-6) and
k-means on one-hot time rows manufactured spurious macro determinism from
white noise (noise CE 0.080 >= grid CE 0.068). Both defects fixed here.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

MICRO_R = 4          # PCA coordinates kept from the thin SVD.
MICRO_K = 2          # sign bins per coordinate.
MACRO_M = 4          # max macro clusters (hard cap; fewer than micro states).
WINDOW = 64          # default sliding window length (steps).
N_SHUFFLES = 8       # null-surrogate shuffles averaged for corrected CE.


def effective_information(tpm: torch.Tensor) -> float:
    """Hoel EI of a row-stochastic TPM under uniform do-intervention.

    EI = (1/n) sum_i KL(TPM[i,:] || pbar), pbar = column means of TPM.
    Exact for a true TPM; empirical TPMs are DERIVED approximations.
    """
    n = tpm.shape[0]
    pbar = tpm.mean(dim=0) + 1e-12
    kl = (tpm * torch.log2((tpm + 1e-12) / pbar)).sum(dim=1)
    return float(kl.mean())


def micro_states(waves: torch.Tensor, r: int = MICRO_R, k: int = MICRO_K) -> torch.Tensor:
    """PCA-reduce [T, D] waves to [T] micro-state indices (int64).

    waves: real float [T, D]. Each coordinate sign-quantized to k bins; state
    id = base-k encoding over r coordinates (2^r states for k=2).
    """
    T, D = waves.shape
    if T < 4 or D < r:
        raise ValueError(f"need T>=4 and D>=r: got T={T}, D={D}, r={r}")
    center = waves - waves.mean(dim=0, keepdim=True)
    U, _, _ = torch.linalg.svd(center, full_matrices=False)
    coords = U[:, :r]  # [T, r]
    codes = (coords > 0).to(torch.int64).clamp(0, k - 1)  # [T, r]
    state = torch.zeros(T, dtype=torch.int64, device=waves.device)
    for j in range(r):
        state = state * k + codes[:, j]
    return state


def _empirical_tpm(states: torch.Tensor, n_states: int) -> tuple:
    """Empirical transition count matrix + Laplace-smoothed row TPM.

    Returns (counts, tpm): counts [S, S] raw transition counts; tpm [S, S]
    row-stochastic with +1 Laplace smoothing. Rows with zero visits are left
    as zero rows in counts (excluded by callers via support).
    """
    n = states.numel()
    counts = torch.zeros(n_states, n_states, dtype=torch.float64)
    for t in range(n - 1):
        counts[states[t], states[t + 1]] += 1.0
    tpm = counts + 1.0
    tpm = tpm / tpm.sum(dim=1, keepdim=True)
    return counts, tpm


def estimate_ei_from_sequence(states: torch.Tensor, k: int = MICRO_K ** MICRO_R) -> Dict:
    """Empirical EI from a state sequence with Laplace smoothing.

    Returns {ei, support, n, status}. status == "ok" only when support >= 2.
    """
    n_steps = states.numel()
    n_states = int(states.max().item()) + 1 if n_steps else 0
    counts, tpm = _empirical_tpm(states, n_states)
    support = int((counts.sum(dim=1) > 0).sum().item())
    if support < 2 or n_steps < 8:
        return {"ei": None, "support": support, "n": n_steps, "status": "insufficient_support"}
    return {"ei": effective_information(tpm), "support": support, "n": n_steps, "status": "ok"}


def macro_tpm_ei(states: torch.Tensor, m: int = MACRO_M) -> tuple:
    """Hoel coarse-graining CE component.

    Clusters the VISITED states by their empirical TPM rows (k-means on the
    row-normalized transition rows), builds the macro TPM as the
    count-weighted row means, and returns (ei_macro, macro_m).

    For white noise all rows are similar random rows -> macro rows average to
    ~uniform -> EI_macro < EI_micro (sampling bias removed by averaging),
    giving CE <= 0. For structured data with distinguishable rows, CE > 0 is
    possible.
    """
    n_states = int(states.max().item()) + 1
    counts, tpm = _empirical_tpm(states, n_states)
    present = counts.sum(dim=1) > 0
    present_ids = torch.nonzero(present).flatten()
    p = int(present_ids.numel())
    m = max(1, min(m, p))
    if p < 2:
        return None, 0
    rows = tpm[present_ids]  # [P, S] TPM rows of visited states
    # k-means over rows.
    centers = rows[:1].clone()
    while centers.shape[0] < m:
        dists = (rows.unsqueeze(0) - centers.unsqueeze(1)).pow(2).sum(dim=2).min(dim=0).values
        centers = torch.cat([centers, rows[dists.argmax()].unsqueeze(0)], dim=0)
    for _ in range(50):
        assign = (rows.unsqueeze(0) - centers.unsqueeze(1)).pow(2).sum(dim=2).argmin(dim=0)
        new_centers = []
        for c in range(m):
            mask = assign == c
            new_centers.append(rows[mask].mean(dim=0) if mask.any() else centers[c])
        centers = torch.stack(new_centers)
    assign = (rows.unsqueeze(0) - centers.unsqueeze(1)).pow(2).sum(dim=2).argmin(dim=0)
    # Macro TPM: count-weighted row means over the raw counts.
    macro_counts = torch.zeros(m, m, dtype=torch.float64)
    for c in range(m):
        idx = present_ids[assign == c]
        sub = counts[idx].sum(dim=0)  # [S] raw counts from cluster c
        for a in range(m):
            tgt = present_ids[assign == a]
            macro_counts[c, a] = sub[tgt].sum().item()
    macro_tpm = macro_counts + 1.0
    macro_tpm = macro_tpm / macro_tpm.sum(dim=1, keepdim=True)
    return effective_information(macro_tpm), m


def causal_emergence(
    waves: torch.Tensor,
    r: int = MICRO_R,
    k: int = MICRO_K,
    m: int = MACRO_M,
) -> Dict:
    """Full CE report for a [T, D] wave window. Returns dict (never raises).

    Keys: ei_micro, ei_macro, ce_null, ce_raw, ce_bits, support, n, macro_m,
    status. ce_bits = ce_raw - ce_null (null-surrogate-corrected, amended
    2026-08-21 T2 v2). status != "ok" => measurement unavailable.
    """
    try:
        if waves.ndim != 2:
            raise ValueError(f"waves must be [T, D], got {tuple(waves.shape)}")
        micro = micro_states(waves, r=r, k=k)
        n_states = int(micro.max().item()) + 1
        counts, tpm = _empirical_tpm(micro, n_states)
        support = int((counts.sum(dim=1) > 0).sum().item())
        if support < 2 or micro.numel() < 8:
            return {"ei_micro": None, "ei_macro": None, "ce_null": None,
                    "ce_raw": None, "ce_bits": None, "support": support,
                    "n": int(micro.numel()), "macro_m": 0,
                    "status": "insufficient_support"}
        ei_micro = effective_information(tpm)
        ei_macro, macro_m = macro_tpm_ei(micro, m=m)
        ce_raw = (ei_macro - ei_micro) if (ei_macro is not None) else None
        # Null surrogate (amended T2 v3): mean over N_SHUFFLES seeded shuffles
        # of the same sequence (marginals preserved, temporal coupling
        # destroyed). Averaging cuts single-surrogate residual variance.
        nulls = []
        seed_base = int(micro.sum().item()) % (2 ** 31)
        for s in range(N_SHUFFLES):
            g = torch.Generator().manual_seed(seed_base + s)
            shuffled = micro[torch.randperm(micro.numel(), generator=g)]
            ei_micro_n = effective_information(_empirical_tpm(shuffled, n_states)[1])
            ei_macro_n, _ = macro_tpm_ei(shuffled, m=m)
            if ei_macro_n is not None:
                nulls.append(ei_macro_n - ei_micro_n)
        ce_null = float(torch.tensor(nulls).mean()) if nulls else None
        ce = (ce_raw - ce_null) if (ce_raw is not None and ce_null is not None) else None
        return {
            "ei_micro": round(ei_micro, 6),
            "ei_macro": round(ei_macro, 6) if ei_macro is not None else None,
            "ce_null": round(ce_null, 6) if ce_null is not None else None,
            "ce_raw": round(ce_raw, 6) if ce_raw is not None else None,
            "ce_bits": round(ce, 6) if ce is not None else None,
            "support": support,
            "n": int(micro.numel()),
            "macro_m": macro_m,
            "status": "ok",
        }
    except Exception as exc:  # noqa: BLE001 - diagnostic must never crash the runner
        return {"ei_micro": None, "ei_macro": None, "ce_null": None,
                "ce_raw": None, "ce_bits": None, "support": 0, "n": 0,
                "macro_m": 0, "status": f"error:{type(exc).__name__}"}


class CausalEmergenceTelemetry:
    """Streaming windowed CE telemetry. push(wave) per step; report() per window.

    Also computes the erasure probe: CE on the window with the last 25% dropped
    (forgetfulness-resistance diagnostic, packet gate T4).
    """

    def __init__(self, window: int = WINDOW, r: int = MICRO_R, k: int = MICRO_K, m: int = MACRO_M):
        self.window = window
        self.r, self.k, self.m = r, k, m
        self.buffer: List[torch.Tensor] = []
        self.first_report: Optional[Dict] = None
        self.reports: List[Dict] = []

    def push(self, wave: torch.Tensor) -> None:
        """wave: real [num_blocks, 8] (or flat [D]) float32, any device."""
        flat = wave.detach().reshape(-1).float().cpu()
        if flat.numel() == 0 or not torch.isfinite(flat).all():
            return  # non-finite waves are not evidence
        self.buffer.append(flat)

    def report(self) -> Optional[Dict]:
        """Compute CE on the current window; resets the buffer. None until full."""
        if len(self.buffer) < self.window:
            return None
        win = torch.stack(self.buffer[-self.window:])  # [W, D]
        self.buffer = []
        rep = causal_emergence(win, r=self.r, k=self.k, m=self.m)
        if rep["status"] == "ok":
            erase_n = max(1, self.window // 4)
            rep_erase = causal_emergence(win[: self.window - erase_n], r=self.r, k=self.k, m=self.m)
            rep["ce_after_erase"] = rep_erase.get("ce_bits")
            rep["erase_n"] = erase_n
        if self.first_report is None:
            self.first_report = rep
        self.reports.append(rep)
        return rep
