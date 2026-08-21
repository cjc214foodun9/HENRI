"""CE ratchet probe (packet HENRI-CLASS47-CE-TELEMETRY-2026-08-21, gates T3/T4).

Reuses the PRODUCTION learner (EFEPlanner.train_transition_step) on a
structured wave trajectory at D=65,536 (CUDA). Measures Causal Emergence of
the planner's emitted wave trajectory:

  Arm U (untrained):  predicted_t = transition(state_t, action_t); NO updates.
  Arm T (trained):    per-step train_transition_step(...) SGLD creep on the
                      same trajectory; collect post-update predicted waves.
  Erase arm (T4):     CE on the trained trajectory with the last 25% dropped
                      (engram-window erasure; forgetfulness resistance).

Gates (amended 2026-08-21):
  T3: ce_trained - ce_untrained > +0.01 bits
  T4: ce_after_erase >= ce_untrained - 0.02 bits
  S:  |CE| <= 3 bits, no NaN

Usage:
  python scripts/verification/ce_ratchet_probe.py --device cuda --d-model 65536
  python scripts/verification/ce_ratchet_probe.py --device cpu --d-model 4096  # local smoke
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch

from efe_planner import EFEPlanner
from causal_emergence_telemetry import CausalEmergenceTelemetry, causal_emergence


def build_trajectory(num_blocks: int, block_dim: int, steps: int, device: str, seed: int = 7):
    """Structured, temporally coupled wave trajectory (real [steps, B, 8]).

    Two alternating macro clusters (A/B deterministic alternation) with slow
    drift + per-step jitter. The ACTION encodes the NEXT cluster (a learnable
    causal relation: (state_t, action_t) -> state_{t+1}); observed[i] is the
    actual next-state wave. Without this, training on a non-causal action
    forces the action-conditioned transition into an action-averaged fixed
    point, destroying prediction structure (probe artifact fixed 2026-08-21).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = torch.linspace(0, 1, block_dim)  # per-block phase coordinate
    phases = [3.0, 3.7]  # clusters A, B
    cluster_seq = [0, 1] * (steps // 2 + 1)
    cluster_seq = cluster_seq[:steps]
    states, actions, observed = [], [], []
    for i in range(steps):
        phase = phases[cluster_seq[i]]
        base = torch.sin(2 * math.pi * phase * t + 0.05 * i)  # [block_dim]
        base = base.unsqueeze(0).repeat(num_blocks, 1)  # [num_blocks, block_dim]
        jitter = 0.15 * torch.randn(num_blocks, block_dim, generator=g)
        state = base + jitter
        state = state / (torch.norm(state, p=2, dim=-1, keepdim=True) + 1e-9)
        # Action encodes the NEXT cluster (causal content the transition can learn).
        next_phase = phases[cluster_seq[min(i + 1, steps - 1)]]
        action = torch.full((num_blocks, block_dim), math.cos(next_phase))
        action = action + 0.02 * torch.randn(num_blocks, block_dim, generator=g)
        action = action / (torch.norm(action, p=2, dim=-1, keepdim=True) + 1e-9)
        # Observed next = the actual next-state wave.
        nb = torch.sin(2 * math.pi * phases[cluster_seq[min(i + 1, steps - 1)]] * t + 0.05 * (i + 1))
        nb = nb.unsqueeze(0).repeat(num_blocks, 1)
        nxt = nb + 0.1 * torch.randn(num_blocks, block_dim, generator=g)
        nxt = nxt / (torch.norm(nxt, p=2, dim=-1, keepdim=True) + 1e-9)
        states.append(state)
        actions.append(action)
        observed.append(nxt)
    return (
        torch.stack(states).to(device),
        torch.stack(actions).to(device),
        torch.stack(observed).to(device),
    )


def collect_ce(planner, states, actions, observed, train: bool, window: int = 64):
    """Run the trajectory (optionally training per step); return (CE rep, pred-cos, losses).

    pred-cos = mean cosine(predicted_i, observed_i): the mechanism-engagement
    check. If the trained arm does not beat the untrained arm on pred-cos, the
    arm did NOT learn, and a negative T3 delta is a training-failure artifact,
    not a ratchet-absence measurement (BLOCKED_INFRASTRUCTURE for the gate).
    """
    ce_tele = CausalEmergenceTelemetry(window=window)
    pred_cos = []
    losses = []
    last = None
    with torch.enable_grad():
        for i in range(states.shape[0]):
            pred = planner.transition(states[i], actions[i]).detach()
            ce_tele.push(pred.reshape(-1))
            cos = torch.nn.functional.cosine_similarity(
                pred.reshape(-1), observed[i].reshape(-1), dim=0)
            pred_cos.append(float(cos.item()))
            if train and i < states.shape[0] - 1:
                loss = planner.train_transition_step(
                    states[i], actions[i], observed[i],
                    lr=0.05, surprise_modulate=True, valence=0.0 if i % 2 else 0.5,
                )
                losses.append(float(loss))
            rep = ce_tele.report()
            if rep is not None:
                last = rep
    import statistics
    return last, {
        "pred_cos_mean": round(statistics.mean(pred_cos), 6),
        "pred_cos_first": round(pred_cos[0], 6),
        "pred_cos_last": round(pred_cos[-1], 6),
        "loss_mean": round(statistics.mean(losses), 6) if losses else None,
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=65536)
    ap.add_argument("--num-blocks", type=int, default=8192)
    ap.add_argument("--block-dim", type=int, default=8)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--output", default="/tmp/ce_ratchet_probe.json")
    args = ap.parse_args()

    block_dim = args.block_dim
    num_blocks = args.num_blocks if args.d_model >= args.num_blocks * block_dim else args.d_model // block_dim
    torch.manual_seed(7)
    transition_rank = min(64, num_blocks * block_dim)
    planner = EFEPlanner(num_blocks=num_blocks, d_model=args.d_model, transition_rank=transition_rank)
    planner = planner.to(args.device)
    states, actions, observed = build_trajectory(num_blocks, block_dim, args.steps, args.device)

    # Arm U: untrained.
    rep_u, engage_u = collect_ce(planner, states, actions, observed, train=False)
    # Arm T: trained (fresh planner, same seed).
    planner2 = EFEPlanner(num_blocks=num_blocks, d_model=args.d_model, transition_rank=transition_rank).to(args.device)
    rep_t, engage_t = collect_ce(planner2, states, actions, observed, train=True)

    result = {
        "packet": "HENRI-CLASS47-CE-TELEMETRY-2026-08-21",
        "device": args.device,
        "d_model": args.d_model,
        "num_blocks": num_blocks,
        "steps": args.steps,
        "ce_untrained": rep_u,
        "ce_trained": rep_t,
        "engagement_untrained": engage_u,
        "engagement_trained": engage_t,
        "learning_engaged": (engage_t["pred_cos_mean"] > engage_u["pred_cos_mean"]),
        "t3_delta": (rep_t["ce_bits"] - rep_u["ce_bits"]) if rep_u and rep_t else None,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # T4: erasure probe on the trained trajectory (last 25% dropped).
    if rep_t and rep_t.get("status") == "ok":
        result["t4_ce_after_erase"] = rep_t.get("ce_after_erase")
        result["t4_delta_vs_untrained"] = (
            rep_t.get("ce_after_erase") - rep_u["ce_bits"] if rep_u else None)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print("RATCHET_PROBE_DONE")


if __name__ == "__main__":
    main()
