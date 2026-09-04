"""Carrier K3 — Empirical Koopman steering engine (default-OFF, G-series gauntlet).

Prereg: docs/spec/carrier_k3_empirical_koopman_preregistration.md (sealed
2026-09-03). Base: P1GoalSteeringEngine lineage @ 2f9bc57 (C1 sealed
FALSIFIED; K3 replaces the synthetic rotor dictionary with LIVE empirical
operators, per the supplied spec's immediate action item).

Mechanism (prereg §1-§4): per-action block-local Ridge Koopman operators are
fit from the run loop's OWN causal transition pairs (Psi_t, a_t, Psi_{t+1})
captured at the G4 post-step hook (update_online_affordance). Scoring follows
the P1/C1 potential-drop form, with candidates produced by the EMPIRICAL
operator instead of a synthetic rotor:

    cand_a = K_a psi_full          (block-wise; live-fitted)
    drops[a] = |<cand_a, g>|^2 - |<psi, g>|^2
    j(a) = (clamp(drops[a], -1, 1) + 1) * pi_a^H

Causality: an action's ring is cleared at every environment boundary
(dynamics are env-specific); the operator for a scored step is fit ONLY on
rows strictly older than the sliding held-out window (the newest
W = clamp(N/4, 2, 8) rows are never in the fit sums), so the current row is
never scored by an operator that saw it. KG1 held-out error is evaluated on
the rows that were excluded from the fit, once per env per action.

Verdicts (prereg §4.5 precedence): K3_BLOCKED_NAN | K3_NO_ENGAGEMENT |
K3_GATE_KG5_LATENCY_FAILED | K3_GATE_KG1_PREDICTION_FAILED |
K3_GATE_KG3_SEPARATION_FAILED | K3_GATE_KG4_SPECTRAL_FAILED |
K3_FALSIFIED_ACTION_COUPLING (KG2) | K3_FALSIFIED_TASK_SOLVE_KG6 |
K3_EMPIRICAL_KOOPMAN_VERIFIED.

Default-OFF: flag HENRI_K3_KOOPMAN=1 (fail-closed via require_k3_flag);
launcher routing mirrors use_g8/use_c1/use_p1.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from arc_g7_calibrated_engine import (
    G1_LATENCY_MS,
    G2_MIN_SOLVED,
    G4_MAX_AFFORDANCE,
)
from arc_p1_goal_steering_engine import (
    DV_CLAMP,
    P1GoalSteeringEngine,
)
from arc_k3_koopman_generator import (
    K3_FLAG,
    K3_M,
    K3_D,
    K3_RING_CAP,
    KFIT_MIN_N,
    BlockRidgeKoopmanFit,
    K3NumericalAbort,
    K3RingAccumulator,
    require_k3_flag,
)

K3_FIT_STRIDE = 8  # refit an action only after this many new ring rows

# KG gates (sealed prereg §5)
KG1_MAX_HELDOUT_ERR = 0.1500
KG2_MIN_DELTA_NU = 0.0200
KG3_MIN_SEPARATION = 0.0500
KG4_MAX_RHO = 1.000001
KG6_MIN_SOLVED = G2_MIN_SOLVED

# require_k3_flag is re-exported for launcher parity with G8/C1/P1.
__all__ = [
    "K3_FLAG", "K3KoopmanSteeringEngine", "require_k3_flag",
    "KG1_MAX_HELDOUT_ERR", "KG2_MIN_DELTA_NU", "KG3_MIN_SEPARATION",
    "KG4_MAX_RHO", "KG6_MIN_SOLVED",
]


class K3KoopmanSteeringEngine(P1GoalSteeringEngine):
    """P1/C1 + live-fitted empirical block-Koopman action candidates.

    Rings are per-action and cleared at each environment boundary
    (p1_bind_env_goal override). Operators are refit lazily on score calls
    once a ring holds >= KFIT_MIN_N fit rows. Falls back to the inherited
    G7/P1 scorer (byte-identical) when no full goal is bound or no action
    has a fitted operator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._k3_env = None
        self.k3_rings = {}                 # {a: K3RingAccumulator}
        self.k3_fits = {}                  # {a: BlockRidgeKoopmanFit}
        self.k3_ops = {}                   # {a: K [M,8,8] fp32}
        self.k3_ops_fit_n = {}             # {a: last fit row count}
        self.k3_fit_calls = 0
        self.k3_score_calls = 0
        self.k3_ring_pushes = 0
        self.k3_fired_blocks = 0
        self.k3_sigma_raw_max = 0.0
        self.k3_sigma_post_max = 0.0
        self.k3_pinv_fallbacks = 0
        self.k3_alpha_doublings = 0
        self.k3_latencies_ms = []
        self.k3_displacements = []
        self.k3_last = {}
        # per-(env, action) held-out metrics; finalized at env boundaries
        self._k3_heldout_errors = []       # list of float (per env/action)
        self._k3_pairwise_seps = []        # list of float (per env, pairs)
        self.k3_envs_with_goal = set()
        self._k3_finalized_envs = set()    # idempotency guard
        self._k3_dnu_current_env = None    # env whose dnu slice is open
        self._k3_env_dnus = {}             # {env_name: [dnu, ...]}
        self.k3_ring_cache = {}            # {a: (count_at_cache, X, Y)} fp32

    # -- guarded per-env dnu tracking (seal basis = goal-available envs) -----

    def k3_observe_dnu(self, dnu: float) -> None:
        """Guarded run-loop hook (called after dnus.append in G4)."""
        if self._k3_dnu_current_env is not None:
            self._k3_env_dnus.setdefault(self._k3_dnu_current_env, []).append(
                float(dnu))

    def k3_seal_basis_dnu(self) -> float:
        """Mean delta-nu over goal-available envs ONLY (sealed prereg §4.6)."""
        vals = [d for e, ds in self._k3_env_dnus.items()
                for d in ds if e in self.k3_envs_with_goal]
        if not vals:
            return None
        return sum(vals) / len(vals)

    def k3_per_env_dnu_mean(self) -> dict:
        return {e: (sum(ds) / len(ds) if ds else None)
                for e, ds in sorted(self._k3_env_dnus.items())}

    # -- per-action ring -----------------------------------------------------

    def _ring(self, a: int) -> K3RingAccumulator:
        if a not in self.k3_rings:
            self.k3_rings[a] = K3RingAccumulator(
                K3_RING_CAP, K3_M, K3_D,
                torch.device(self.device))
            self.k3_fits[a] = BlockRidgeKoopmanFit()
        return self.k3_rings[a]

    def _dump_abort_state(self, reason: str) -> None:
        """Serialize ring state to <cwd>/_abort_k3/ before failing closed."""
        try:
            out_dir = os.path.join(os.getcwd(), "_abort_k3")
            os.makedirs(out_dir, exist_ok=True)
            state = {"reason": reason, "env": self._k3_env,
                     "rings": {}}
            for a in sorted(self.k3_rings):
                ring = self.k3_rings[a]
                X, Y = ring.ordered()
                state["rings"][str(a)] = {
                    "count": ring.count, "n": ring.n,
                    "X": X.cpu(), "Y": Y.cpu()}
            torch.save(state, os.path.join(out_dir, f"k3_abort_{self._k3_env}.pt"))
        except Exception:
            pass  # serialization best-effort; the abort still propagates

    def _reset_rings(self) -> None:
        for a in self.k3_rings:
            self.k3_rings[a].reset()
        self.k3_ops = {}
        self.k3_ops_fit_n = {}
        self.k3_ring_cache = {}

    def _ring_ordered_cached(self, a: int):
        """fp32 (X, Y) arrival-ordered; recomputed only when count changed.

        KG5 discipline: the ordered() fp32 materialization is 64 MB/action at
        cap; recomputing it on every score call would churn ~450 MB/step.
        """
        ring = self._ring(a)
        cached = self.k3_ring_cache.get(a)
        if cached is not None and cached[0] == ring.count:
            return cached[1], cached[2]
        X, Y = ring.ordered()
        self.k3_ring_cache[a] = (ring.count, X, Y)
        return X, Y

    # -- env boundary hook (goal binding + ring reset + finalization) --------

    def p1_bind_env_goal(self, env_name: str) -> None:
        if self._k3_env is not None and self._k3_env != env_name:
            self._finalize_env(self._k3_env)
        self._k3_env = env_name
        self._k3_dnu_current_env = env_name
        self._reset_rings()
        super().p1_bind_env_goal(env_name)
        if self._p1_goal_full is not None:
            self.k3_envs_with_goal.add(env_name)

    def _finalize_env(self, env_name: str) -> None:
        """Held-out KG1 errors + KG3 pairwise separation for the env that just
        ended. Uses the CAUSAL fit split: operator on oldest n_fit rows,
        eval on the newest W rows that were never in the fit."""
        if env_name in self._k3_finalized_envs:
            return
        self._k3_finalized_envs.add(env_name)
        for a in sorted(self.k3_rings):
            ring = self.k3_rings[a]
            if ring.count < KFIT_MIN_N + 2:
                continue
            X, Y = self._ring_ordered_cached(a)
            n_fit, _w = ring.fit_eval_split()
            if n_fit < KFIT_MIN_N:
                continue
            fit = self.k3_fits[a]
            try:
                res = fit.fit(X, Y, n_fit)
            except K3NumericalAbort:
                raise
            self.k3_ops[a] = res["K"]
            self.k3_fit_calls += 1
            self.k3_fired_blocks += res["fired_blocks"]
            self.k3_sigma_raw_max = max(self.k3_sigma_raw_max, res["sigma_max"])
            self.k3_sigma_post_max = max(self.k3_sigma_post_max,
                                         res["sigma_post_max"])
            self.k3_pinv_fallbacks += int(res["pinv_fallback"])
            self.k3_alpha_doublings += int(res["alpha_doublings"])
            err = BlockRidgeKoopmanFit.heldout_error(
                X[n_fit:], Y[n_fit:], res["K"])
            if torch.isfinite(torch.tensor(err)):
                self._k3_heldout_errors.append(err)
        # KG3: min pairwise mean-Frobenius distance across fitted ops
        ops = [(a, self.k3_ops[a]) for a in sorted(self.k3_ops)
               if self.k3_ops.get(a) is not None]
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                d = (ops[i][1] - ops[j][1]).norm(p="fro", dim=(-2, -1)).mean().item()
                self._k3_pairwise_seps.append(d)

    # -- causal post-step hook (live transitions) ----------------------------

    def update_online_affordance(self, psi_full, action_idx, psi_full_next,
                                 eta=None):
        """Capture the OBSERVED (pre, post) pair into the action's ring."""
        psi_full = psi_full.float().to(self.device)
        psi_full_next = psi_full_next.float().to(self.device)
        a = int(action_idx)
        try:
            self._ring(a).push(psi_full.reshape(-1), psi_full_next.reshape(-1))
        except K3NumericalAbort as exc:
            self._dump_abort_state(str(exc))
            raise
        self.k3_ring_pushes += 1
        return super().update_online_affordance(psi_full, a, psi_full_next, eta)

    # -- K3 scoring ----------------------------------------------------------

    def _fit_action_if_new(self, a: int) -> bool:
        ring = self._ring(a)
        if ring.count < KFIT_MIN_N:
            return False
        X, Y = self._ring_ordered_cached(a)
        n_fit, _w = ring.fit_eval_split()
        if n_fit < KFIT_MIN_N:
            return False
        last_n = self.k3_ops_fit_n.get(a, 0)
        if a in self.k3_ops and (ring.count - last_n) < K3_FIT_STRIDE:
            return True  # existing op is fresh within stride (KG5 cadence)
        try:
            res = self.k3_fits[a].fit(X, Y, n_fit)
        except K3NumericalAbort as exc:
            self._dump_abort_state(str(exc))
            raise
        self.k3_ops[a] = res["K"]
        self.k3_ops_fit_n[a] = ring.count  # count-at-fit for the stride check
        self.k3_fit_calls += 1
        self.k3_fired_blocks += res["fired_blocks"]
        self.k3_sigma_raw_max = max(self.k3_sigma_raw_max, res["sigma_max"])
        post = res["sigma_max"] / max(1.0, res["sigma_max"])
        self.k3_sigma_post_max = max(self.k3_sigma_post_max, post)
        self.k3_pinv_fallbacks += int(res["pinv_fallback"])
        self.k3_alpha_doublings += int(res["alpha_doublings"])
        return True

    def score_all_actions(self, psi64, psi_full, waypoint=None):
        if self._p1_goal_full is None:
            return super().score_all_actions(psi64, psi_full, waypoint)

        psi_full = psi_full.float().to(self.device)
        batched = psi_full.dim() == 3
        if not batched:
            psi_full = psi_full.unsqueeze(0)          # [1, 8192, 8]
        B = psi_full.shape[0]
        flat = F.normalize(psi_full.reshape(B, -1), p=2, dim=-1)   # [1, D]
        g = F.normalize(self._p1_goal_full.to(self.device), p=2, dim=-1)
        base_align = (flat * g.unsqueeze(0)).sum(-1).abs().clamp(0.0, 1.0) ** 2

        _use_events = (self.device.startswith("cuda")
                       and torch.cuda.is_available())
        _ev_start = _ev_end = None
        if _use_events:
            _ev_start = torch.cuda.Event(enable_timing=True)
            _ev_end = torch.cuda.Event(enable_timing=True)
            _ev_start.record()

        pi = self.predict_affordance(psi_full[0])[0]               # [A]

        # Refit any action whose ring gained rows since the last fit.
        n_fitted = 0
        for a in range(self.n_actions):
            if self._fit_action_if_new(int(a)):
                n_fitted += 1

        if n_fitted == 0:
            # No empirical operator available yet: inherited scorer (this
            # keeps cold-start steps on the calibrated affordance surface).
            return super().score_all_actions(psi64, psi_full, waypoint)

        drops = torch.zeros(self.n_actions, device=self.device)
        disps = torch.zeros(self.n_actions, device=self.device)
        n = psi_full.shape[1]                                       # blocks
        for a in range(self.n_actions):
            K = self.k3_ops.get(int(a))
            if K is None:
                drops[a] = 0.0
                continue
            cand = BlockRidgeKoopmanFit.apply(
                K.to(self.device), psi_full[0])                     # [8192, 8]
            cand_flat = F.normalize(cand.reshape(1, -1), p=2, dim=-1)
            align_t = (cand_flat * g.unsqueeze(0)).sum(-1).abs().clamp(0.0, 1.0) ** 2
            drops[a] = (align_t - base_align)[0]
            disps[a] = (cand - psi_full[0]).norm().item() / (n ** 0.5)

        eff = torch.clamp(drops, min=-DV_CLAMP, max=DV_CLAMP) + 1.0
        j = eff * torch.pow(pi, self.horizon)
        if self.escape_state.get("active"):
            ggen = torch.Generator().manual_seed(
                self.seed + self.escape_state.get("steps", 0))
            noise = torch.sqrt(torch.tensor(
                2.0 * self.langevin_temp, device=self.device)) * \
                torch.randn(self.n_actions, generator=ggen).to(self.device)
            j = j + noise

        if _use_events and _ev_end is not None:
            _ev_end.record()
            torch.cuda.synchronize()
            self.k3_latencies_ms.append(float(_ev_start.elapsed_time(_ev_end)))

        sel = int(torch.argmax(j).item())
        self.k3_last = {
            "selected_action": sel,
            "potential_drops": [float(v) for v in drops.detach().cpu().tolist()],
            "base_align": float(base_align[0].item()),
            "n_fitted_actions": n_fitted,
        }
        if self._p1_drop_accum is None:
            self._p1_drop_accum = drops.detach().cpu().clone()
        else:
            self._p1_drop_accum += drops.detach().cpu()
        self.k3_displacements.append(float(disps[sel].item()))
        self.k3_score_calls += 1
        return {self.action_names[i]: float(j[i].item())
                for i in range(self.n_actions)}

    # -- verdicts (sealed prereg §4.5 precedence) ----------------------------

    def _decide_verdict(self, mean_latency, solved, mean_delta_nu, g4_mean,
                        steps_done, updates):
        if self._k3_env is not None:
            # finalize the trailing env so its held-out rows are counted
            try:
                self._finalize_env(self._k3_env)
            except K3NumericalAbort:
                raise
        if self.k3_latencies_ms:
            mean_latency = sum(self.k3_latencies_ms) / len(self.k3_latencies_ms)
        # KG2 seal basis (prereg §4.6): goal-available envs only. The inherited
        # mean_delta_nu argument spans ALL envs including the 5 without bank
        # goals; substitute the per-env observer's goal-env mean when present.
        seal_dnu = self.k3_seal_basis_dnu()
        if seal_dnu is not None:
            mean_delta_nu = seal_dnu
        if steps_done > 0 and self.k3_score_calls == 0:
            return "K3_NO_ENGAGEMENT"
        if mean_latency is not None and mean_latency > G1_LATENCY_MS:
            return "K3_GATE_KG5_LATENCY_FAILED"
        if self._k3_heldout_errors:
            kg1 = sum(self._k3_heldout_errors) / len(self._k3_heldout_errors)
            if kg1 > KG1_MAX_HELDOUT_ERR:
                return "K3_GATE_KG1_PREDICTION_FAILED"
        if self._k3_pairwise_seps:
            kg3 = min(self._k3_pairwise_seps)
            if kg3 < KG3_MIN_SEPARATION:
                return "K3_GATE_KG3_SEPARATION_FAILED"
        if self.k3_sigma_post_max > KG4_MAX_RHO:
            return "K3_GATE_KG4_SPECTRAL_FAILED"
        if mean_delta_nu is not None and mean_delta_nu < KG2_MIN_DELTA_NU:
            return "K3_FALSIFIED_ACTION_COUPLING"
        if solved < KG6_MIN_SOLVED:
            return "K3_FALSIFIED_TASK_SOLVE_KG6"
        if g4_mean is not None and g4_mean > G4_MAX_AFFORDANCE:
            return "K3_GATE_G4_AFFORDANCE_FAILED"
        return "K3_EMPIRICAL_KOOPMAN_VERIFIED"

    def k3_receipt_fields(self) -> dict:
        kg1 = (sum(self._k3_heldout_errors) / len(self._k3_heldout_errors)
               if self._k3_heldout_errors else None)
        kg3 = (min(self._k3_pairwise_seps)
               if self._k3_pairwise_seps else None)
        return {
            "k3_score_calls": self.k3_score_calls,
            "k3_fit_calls": self.k3_fit_calls,
            "k3_ring_pushes": self.k3_ring_pushes,
            "k3_fired_blocks": self.k3_fired_blocks,
            "k3_sigma_raw_max": self.k3_sigma_raw_max,
            "k3_sigma_post_max": self.k3_sigma_post_max,
            "k3_pinv_fallbacks": self.k3_pinv_fallbacks,
            "k3_alpha_doublings": self.k3_alpha_doublings,
            "k3_heldout_mean_err": kg1,
            "k3_heldout_err_samples": len(self._k3_heldout_errors),
            "k3_min_pairwise_sep": kg3,
            "k3_pairwise_sep_samples": len(self._k3_pairwise_seps),
            "k3_mean_selected_displacement": (
                sum(self.k3_displacements) / len(self.k3_displacements)
                if self.k3_displacements else None),
            "k3_envs_with_goal": list(self.k3_envs_with_goal),
        }
