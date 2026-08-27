"""HENRI Latent Explorer v1 — internal test-time demo discovery for the Goal Adapter.

Directive (2026-08-27): "Learning demos must come from latent exploration of
real world representations... from internal test time trial and error without
submitting incomplete answers to ARC-AGI-3."

Mechanism (arm D; single lever = the Goal Adapter's demo INPUT source; NO
learner or codec tuning):
  1. Source: REAL observed transitions (state_wave -> observed_next_wave) from
     the CURRENT episode — real frames encoded at the canonical [8192, 8]
     per-block unit-sphere boundary. No synthetic waves, no API submissions.
  2. Compile: the sealed Goal Adapter v1 (henri_goal_adapter.py, SHA-256
     0341a278... consumed byte-identical) compiles Channel G from the last
     <= MAX_DEMO_PAIRS non-degenerate real transitions (one-shot).
  3. Long horizon (minimum internal exploration): the goal is rolled forward
     internally through the LIVE transition operator (bounded horizon, default
     2), accepted ONLY when Sagnac coherence with the real-transition goal is
     preserved (roll_sagnac < ROLL_ACCEPT_SAGNAC = 0.05); otherwise the
     real-transition goal is kept (fail-closed). No internal candidate is ever
     submitted to the environment (ARC RHAE: internal reasoning costs zero
     actions; incomplete submissions would degrade the score).

Zero trainable parameters. Deterministic. Default-OFF: HENRI_GOAL_ADAPTER=1
AND HENRI_LATENT_EXPLORE=1 (runner gate; module never imported otherwise).

Fail-closed:
  - fewer than MIN_DEMO_PAIRS non-degenerate real transitions -> None (caller
    keeps GOAL_ADAPTER_NO_DEMOS semantics).
  - degenerate transitions (state == next within DEGENERATE_EPS) are filtered
    before compilation and can never enter a demo pair.
  - internal roll failures keep the un-rolled goal.
  - no environment/API access: imports are torch, math, os only (the sealed
    adapter is imported lazily).

Contract: HENRI-SPEC-2026-08-GOAL-ADAPTER-V1 demo-source amendment v1.
"""

import math  # noqa: F401  (kept for boundary documentation parity)
import os

import torch

NUM_BLOCKS = 8192
BLOCK_DIM = 8
MIN_DEMO_PAIRS = 2
MAX_DEMO_PAIRS = 4
ROLL_ACCEPT_SAGNAC = 0.05   # coherence >= 0.95 required to accept a rolled goal
DEGENERATE_EPS = 1e-4       # L2(state - next) below this = no-op transition
MAX_HORIZON = 4


def is_enabled() -> bool:
    """Both flags required (differential contract: default OFF)."""
    return (os.environ.get("HENRI_GOAL_ADAPTER", "0") == "1"
            and os.environ.get("HENRI_LATENT_EXPLORE", "0") == "1")


def _sagnac(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalized Sagnac delta = 1 - Re<pred, emp>/(||pred|| ||emp||), in [0, 2]."""
    a = a.reshape(-1).to(torch.float32)
    b = b.reshape(-1).to(torch.float32)
    return float(1.0 - (a @ b) / (a.norm() + 1e-12) / (b.norm() + 1e-12))


def _non_degenerate(transitions, eps: float = DEGENERATE_EPS):
    """Keep only REAL transitions whose observed next wave differs from the
    state wave (no-op actions carry no dynamics signal)."""
    out = []
    for triple in transitions:
        state, action_wave, nxt = triple[0], triple[1], triple[2]
        if state is None or nxt is None:
            continue
        if float((state - nxt).norm().item()) > eps:
            out.append((state, action_wave, nxt))
    return out


def compile_latent_goal(transitions, test_wave, transition, horizon=2,
                        max_pairs=MAX_DEMO_PAIRS, device=None):
    """Compile the sealed Goal Adapter v1 from REAL observed transitions.

    transitions: list of (state_wave, action_wave, observed_next_wave) from
        the CURRENT episode only. The CALLER enforces causal order: the buffer
        holds only steps < t; this module consumes exactly what it receives
        and cannot see future observations.
    test_wave:   [8192, 8] current real observation wave.
    transition:  live LowRankCoupledTransition (read-only forward calls) or
        None (no internal roll).
    horizon:     internal latent forward steps, clamped to [1, MAX_HORIZON].
    device:      torch device; defaults to test_wave.device.

    Returns {"goal_wave": [8192, 8], "info": {...}} or None (fail-closed).
    """
    if device is None:
        device = test_wave.device
    horizon = int(min(max(int(horizon), 1), MAX_HORIZON))

    from henri_goal_adapter import HenriGoalAdapter  # lazy; sealed module

    real = _non_degenerate(transitions)
    if len(real) < MIN_DEMO_PAIRS:
        return None
    pairs = real[-int(max_pairs):]
    xs = torch.stack([p[0].detach().to(torch.float32).to(device) for p in pairs])
    ys = torch.stack([p[2].detach().to(torch.float32).to(device) for p in pairs])
    test = test_wave.detach().to(torch.float32).to(device)

    adapter = HenriGoalAdapter(device=device)
    res = adapter.build_goal(xs, ys, test)
    goal = res["goal_wave"].to(device)

    info = {
        "demo_source": "latent_explore_real_transitions",
        "demo_pair_count": len(pairs),
        "buffer_size": len(transitions),
        "non_degenerate_count": len(real),
        "horizon": horizon,
        "roll_accepted": False,
        "roll_sagnac": None,
        "demo_recon_cos": res["demo_recon_cos"],
        "orthogonality_err": res["orthogonality_err"],
    }

    # Internal long-horizon roll (minimum latent exploration): forward the
    # goal through the LIVE transition operator using the most recent REAL
    # action wave. Accepted only if coherence with the real-transition goal
    # holds; otherwise fail-closed to the un-rolled goal.
    if transition is not None and pairs[-1][1] is not None:
        action_wave = pairs[-1][1].detach().to(torch.float32).to(device)
        rolled = goal
        try:
            for _ in range(horizon):
                rolled = transition.forward(rolled, action_wave)
            roll_sagnac = _sagnac(goal, rolled)
            info["roll_sagnac"] = round(roll_sagnac, 6)
            if roll_sagnac < ROLL_ACCEPT_SAGNAC:
                goal = rolled
                info["roll_accepted"] = True
        except Exception as exc:  # fail-closed: keep the real-transition goal
            info["roll_error"] = type(exc).__name__

    goal = goal / (goal.norm(dim=-1, keepdim=True) + 1e-12)
    return {"goal_wave": goal, "info": info}
