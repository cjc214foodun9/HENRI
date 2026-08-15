"""Phase 8.7 remote CUDA verification matrix (RTX 5090, D=65,536).

Pre-registration: HENRI V2/experiments/sweeps/phase87_typed_actions_design.md
Source PDF raw SHA-256 97d46428... (Phase 8.6 Final Postmortem & Phase 8.7 Blueprint)
All arms diagnostic_only=true; NO environment stepping outside physics rollouts.

Arms (paired discipline from Phase 8.6 — SAME eval trajectories across arms,
disjoint train/eval seeds):
  A0  baseline: random action waves + production FHRR bind (current default path)
  A1  Lever 8.7-A: TypedActionEmbedding + production FHRR bind
  A2  Lever 8.7-A: TypedActionEmbedding + Clifford (non-commutative) bind
  A3  Lever 8.7-B: valence-free (nu=0) pre-training; 50 un-docked steps; L < 0.10
  LAT latency probe: transition forward + train_transition_batch update at D=65,536

Gates (PDF, unchanged):
  8.7-A: held-out Sagnac loss < 0.15 over 32 eval trajectories AND >15% reduction
         vs A0 random-action baseline (promotion criterion #1).
  8.7-B: transition error L_Sagnac < 0.10 across 50 un-docked physical steps.
  LAT:   in-situ update cycle <= 45.0 ms at D=65,536 on RTX 5090.

DONE_MARKER written ONLY when all arms rc=0. Any nonzero arm ->
BLOCKED_INFRASTRUCTURE, no science verdict.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "HENRI V2"))

from efe_planner import EFEPlanner
from henri_typed_actions import TypedActionEmbedding, CliffordTransition, _torque_level
from physical_control_environments import InvertedPendulumEnvironment

OUT = os.environ.get("JEPA_DM_OUT", "/tmp/phase87_result.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("HENRI_SMOKE", "0") == "1"
NUM_BLOCKS = 512 if SMOKE else 8192
D = NUM_BLOCKS * 8
NUM_ACTIONS = 8
TRAIN_STEPS = 8 if SMOKE else 96
EVAL_TRAJECTORIES = 4 if SMOKE else 32
VALENCE_FREE_STEPS = 8 if SMOKE else 50
TRAIN_SEED = 87001
EVAL_SEED = 87002
VZ_SEED = 87003


def sagnac_loss(pred, target):
    """1 - cos on flattened waves; scalar."""
    p = pred.reshape(-1)
    t = target.reshape(-1)
    return float(1.0 - (p * t).sum() / ((p.norm() * t.norm()).clamp(min=1e-12)))


def physics_rollout(seed, steps):
    """Deterministic un-docked InvertedPendulum rollout. Returns (state waves,
    action tokens, next waves). NO reset penalties, NO reward coupling (nu=0)."""
    env = InvertedPendulumEnvironment()
    env.reset(theta_init=0.1, dtheta_init=0.0)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    states, acts, nexts = [], [], []
    for _ in range(steps):
        token = int(torch.randint(0, NUM_ACTIONS, (1,), generator=rng).item())
        torque = _torque_level(token, NUM_ACTIONS)
        s = env.state_to_wave(NUM_BLOCKS, DEVICE)
        env.step(torque)
        nxt = env.state_to_wave(NUM_BLOCKS, DEVICE)
        states.append(s)
        acts.append(token)
        nexts.append(nxt)
    return states, acts, nexts


def action_tensor(arm, tokens):
    if arm == "A0":
        return F.normalize(torch.randn(len(tokens), NUM_BLOCKS, 8, device=DEVICE), dim=-1)
    emb = TypedActionEmbedding(num_actions=NUM_ACTIONS, num_blocks=NUM_BLOCKS,
                               block_dim=8, device=DEVICE)
    return emb.embed(torch.tensor(tokens, device=DEVICE))


def build_planner(arm):
    planner = EFEPlanner(num_blocks=NUM_BLOCKS, d_model=D).to(DEVICE)
    if arm == "A2":
        planner.transition = CliffordTransition(
            num_blocks=NUM_BLOCKS, block_dim=8, rank=64).to(DEVICE)
    return planner


def arm_a0():
    t0 = time.perf_counter()
    st, tok, nx = physics_rollout(TRAIN_SEED, TRAIN_STEPS)
    st_e, tok_e, nx_e = physics_rollout(EVAL_SEED, EVAL_TRAJECTORIES)
    states = torch.stack(st)
    nexts = torch.stack(nx)
    actions = action_tensor("A0", tok)
    planner = build_planner("A0")
    planner.train_transition_batch(states, actions, nexts, iters=3, ridge=1e-4, blend=0.5)
    with torch.no_grad():
        losses = [sagnac_loss(planner.transition(st_e[i], action_tensor("A0", [tok_e[i]])[0]), nx_e[i])
                  for i in range(EVAL_TRAJECTORIES)]
    mean = float(torch.tensor(losses).mean().item())
    return {"verdict": "ok", "arm_rc": 0, "heldout_sagnac": round(mean, 6),
            "losses": [round(x, 6) for x in losses], "wall_s": round(time.perf_counter() - t0, 2)}


def arm_a1a2(arm):
    t0 = time.perf_counter()
    st, tok, nx = physics_rollout(TRAIN_SEED, TRAIN_STEPS)
    st_e, tok_e, nx_e = physics_rollout(EVAL_SEED, EVAL_TRAJECTORIES)
    states = torch.stack(st)
    nexts = torch.stack(nx)
    actions = action_tensor(arm, tok)
    planner = build_planner(arm)
    planner.train_transition_batch(states, actions, nexts, iters=3, ridge=1e-4, blend=0.5)
    with torch.no_grad():
        losses = [sagnac_loss(planner.transition(st_e[i], action_tensor(arm, [tok_e[i]])[0]), nx_e[i])
                  for i in range(EVAL_TRAJECTORIES)]
    mean = float(torch.tensor(losses).mean().item())
    # Baseline for the >15% reduction gate is A0's held-out loss (computed first).
    base = float(os.environ.get("P87_A0_HELDOUT", "1.0"))
    reduction = (base - mean) / base * 100.0 if base > 0 else 0.0
    gate = (mean < 0.15) and (reduction > 15.0) and math.isfinite(mean)
    return {"verdict": "PASS" if gate else "P87_FAIL", "arm_rc": 0,
            "heldout_sagnac": round(mean, 6), "reduction_vs_a0_pct": round(reduction, 4),
            "losses": [round(x, 6) for x in losses],
            "wall_s": round(time.perf_counter() - t0, 2)}


def arm_a3():
    """8.7-B valence-free pre-training: fit on un-docked rollout, eval on a
    FRESH 50-step un-docked rollout (same seed across arms — paired)."""
    t0 = time.perf_counter()
    st, tok, nx = physics_rollout(TRAIN_SEED, TRAIN_STEPS)
    st_e, tok_e, nx_e = physics_rollout(VZ_SEED, VALENCE_FREE_STEPS)
    states = torch.stack(st)
    nexts = torch.stack(nx)
    actions = action_tensor("A1", tok)
    planner = build_planner("A0")  # production FHRR path
    planner.train_transition_batch(states, actions, nexts, iters=3, ridge=1e-4, blend=0.5)
    with torch.no_grad():
        losses = [sagnac_loss(planner.transition(st_e[i], action_tensor("A1", [tok_e[i]])[0]), nx_e[i])
                  for i in range(VALENCE_FREE_STEPS)]
    mean = float(torch.tensor(losses).mean().item())
    gate = (mean < 0.10) and math.isfinite(mean)
    return {"verdict": "PASS" if gate else "P87_B_FAIL", "arm_rc": 0,
            "heldout_sagnac": round(mean, 6), "n_steps": VALENCE_FREE_STEPS,
            "wall_s": round(time.perf_counter() - t0, 2)}


def latency_probe():
    t0 = time.perf_counter()
    planner = build_planner("A0")
    st, tok, nx = physics_rollout(TRAIN_SEED, TRAIN_STEPS)
    states = torch.stack(st)
    nexts = torch.stack(nx)
    actions = action_tensor("A0", tok)
    s = states[0]
    a = actions[0]
    for _ in range(5):
        with torch.no_grad():
            _ = planner.transition(s, a)
    fwd_ms = (time.perf_counter() - t0) / 5.0 * 1000.0
    t1 = time.perf_counter()
    planner.train_transition_batch(states, actions, nexts, iters=1, ridge=1e-4)
    update_ms = (time.perf_counter() - t1) * 1000.0
    gate = update_ms <= 45.0
    return {"verdict": "PASS" if gate else "LAT_FAIL", "arm_rc": 0,
            "fwd_ms": round(fwd_ms, 3), "update_ms": round(update_ms, 3),
            "gate": gate}


def main():
    torch.manual_seed(20260814)
    header = {
        "schema": "henri.phase87.matrix.v1",
        "diagnostic_only": True,
        "cuda": torch.cuda.is_available(),
        "torch": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "source_pdf_sha": "97d46428d88988e940f548f4688b3a02b751d8e17a7ae9db86d693098b8997a7",
        "num_blocks": NUM_BLOCKS,
        "d_model": D,
        "smoke": SMOKE,
    }
    results = {"header": header}
    rc_total = 0
    arms = {"A0": arm_a0, "A1": lambda: arm_a1a2("A1"), "A2": lambda: arm_a1a2("A2"),
            "A3": arm_a3, "LAT": latency_probe}
    for name, fn in arms.items():
        try:
            r = fn()
        except Exception as e:
            r = {"verdict": "ERROR", "arm_rc": 1, "error": f"{type(e).__name__}: {e}"}
        results[name] = r
        if name == "A0":
            os.environ["P87_A0_HELDOUT"] = str(r.get("heldout_sagnac", 1.0))
        rc_total += int(r.get("arm_rc", 1))
        print(f"[{name}] {r.get('verdict')} ({r.get('wall_s', '?')}s): {r.get('verdict')}")
    results["done_marker_rc"] = rc_total
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("DONE_MARKER rc=" + str(rc_total))
    sys.exit(0 if rc_total == 0 else 1)


if __name__ == "__main__":
    main()
