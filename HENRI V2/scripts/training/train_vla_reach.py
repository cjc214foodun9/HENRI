"""HENRI VLA: end-to-end train + CLOSED-LOOP eval on the pixel reaching task.

THE CLAIM THIS SCRIPT TESTS
===========================
"HENRI can be a working machine learning algorithm: pixels in, continuous
actions out, and a learning signal that improves an EXTERNAL OUTCOME."

The external outcome is SUCCESS RATE on held-out environment seeds, compared
against:
    random policy  (measured baseline, ~0.14)
    analytic expert (upper bound, exactly 1.0)

A success rate ABOVE random on unseen seeds is the first real evidence that the
stack learns a policy from pixels. A success rate equal to random is reported as
equal to random.

PIPELINE
========
  obs [3,64,64] pixels  ->  VisionPhaseIngress (VSA bind+superpose)  ->  psi [D]
                        ->  FlowMatchingActionHead (velocity field)   ->  a [7]
                        ->  env.step(a)                               ->  reward/success

  Training = behaviour cloning on analytic-expert episodes (flow-matching MSE).
  Adaptation = the flow head only; the ingress sketch is frozen (Contract A:
  no [D,D] object is ever formed).

WHAT THIS DOES AND DOES NOT SHOW
================================
SHOWS: the vision->phase->continuous-action->world->outcome loop closes; the
head learns a policy that beats random on held-out seeds; VRAM and step times at
D=65536 on a 32 GB GPU; a train-loss curve.
DOES NOT SHOW: robot capability, SOTA, or any benchmark score. The task is
SYNTHETIC. Its only external number is its own success rate.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_continuous_action_head import ActionNormalizer, FlowMatchingActionHead
from henri_vla_env import N_DOF, EnvConfig, PixelReachEnv, random_policy_success
from henri_vla_vision_ingress import VisionPhaseIngress

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "vla_reach_receipt.json")


def resolve_out(cli: str | None) -> str:
    if cli:
        d = os.path.dirname(os.path.abspath(cli))
        if not os.path.isdir(d):
            print(f"MALFORMED --out: directory does not exist: {d}", file=sys.stderr)
            raise SystemExit(2)
        return os.path.abspath(cli)
    envd = os.environ.get("HENRI_RECEIPT_DIR")
    if envd:
        os.makedirs(envd, exist_ok=True)
        return os.path.join(envd, "vla_reach_receipt.json")
    return DEFAULT_OUT


def collect_expert(n_episodes: int, seed: int = 0):
    """Behaviour-cloning dataset from the analytic expert."""
    env = PixelReachEnv()
    obs_l, act_l = [], []
    for i in range(n_episodes):
        for o, a in env.expert_episode(seed + i):
            obs_l.append(o.cpu())
            act_l.append(a.cpu())
    return torch.stack(obs_l), torch.stack(act_l)


@torch.no_grad()
def encode_dataset(ingress, obs, batch: int = 8):
    """pixels -> psi. fp16 storage: at D=65536, 10k samples = ~1.2 GB."""
    out = []
    for i in range(0, obs.shape[0], batch):
        out.append(ingress(obs[i:i + batch]).to(torch.float16).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def closed_loop_success(ingress, head, norm, n_episodes: int, seed: int,
                        device, steps: int = 8) -> dict:
    """The external outcome: success rate on HELD-OUT seeds."""
    env = PixelReachEnv()
    wins = 0
    dists = []
    for i in range(n_episodes):
        env.reset(seed + i)
        done = False
        info = {"success": False, "dist": 1.0}
        while not done:
            obs = env.observe().unsqueeze(0).to(device)
            psi = ingress(obs)
            a = head.sample(psi, steps=steps)[0]
            a = norm.denormalize(a)
            _, _, done, info = env.step(a)
        wins += int(info["success"])
        dists.append(float(info["dist"]))
    return {"success_rate": wins / n_episodes,
            "mean_final_dist": sum(dists) / len(dists),
            "n_episodes": n_episodes, "seeds": f"{seed}..{seed + n_episodes - 1}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=65536)
    ap.add_argument("--train-episodes", type=int, default=192)
    ap.add_argument("--eval-episodes", type=int, default=48)
    ap.add_argument("--eval-seed", type=int, default=90000)   # disjoint from train
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--eval-every", type=int, default=750)
    ap.add_argument("--sample-steps", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: verifies the loop closes, not a result")
    args = ap.parse_args()
    if args.smoke:
        args.d_model, args.steps, args.train_episodes = 4096, 300, 24
        args.eval_episodes, args.eval_every = 16, 150
        args.batch = 32

    out = resolve_out(args.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R: dict = {"schema": "henri.vla-reach.v1",
               "device": dev, "d_model": args.d_model, "smoke": args.smoke,
               "config": vars(args)}
    if dev == "cuda":
        R["gpu"] = {"name": torch.cuda.get_device_name(0),
                    "capability": list(torch.cuda.get_device_capability(0)),
                    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)}
        torch.cuda.reset_peak_memory_stats()
    print(f"[vla] device={dev} D={args.d_model} smoke={args.smoke}", flush=True)

    # ---------------------------------------------------------------- ingress
    t0 = time.perf_counter()
    ingress = VisionPhaseIngress(d_model=args.d_model, image_size=64,
                                 patch=16, d_patch=256).to(dev).eval()
    head = FlowMatchingActionHead(n_dof=N_DOF, d_cond=512, d_hidden=1024,
                                  n_layers=4, phase_dim=args.d_model).to(dev)
    R["head_trainable_params"] = sum(p.numel() for p in head.parameters()
                                     if p.requires_grad)
    R["ingress_trainable_params"] = sum(p.numel() for p in ingress.parameters()
                                        if p.requires_grad)

    # ---------------------------------------------------------------- baselines
    R["baselines"] = {"random_policy_success": round(random_policy_success(64, seed=7), 4),
                      "analytic_expert_success": 1.0}
    print(f"[vla] baselines: {R['baselines']}", flush=True)

    # ---------------------------------------------------------------- dataset
    obs, act = collect_expert(args.train_episodes, seed=0)
    norm = ActionNormalizer(N_DOF).fit(act)
    act_n = norm.normalize(act)
    R["normalizer"] = norm.state_dict()
    print(f"[vla] expert samples={obs.shape[0]}  ops={obs.shape[1:]}", flush=True)
    psi = encode_dataset(ingress, obs.to(dev), batch=8)
    R["psi_shape"] = list(psi.shape)
    R["psi_dtype"] = str(psi.dtype)
    R["psi_gib"] = round(psi.numel() * psi.element_size() / 2 ** 30, 3)
    R["dataset_seconds"] = round(time.perf_counter() - t0, 1)
    print(f"[vla] psi {tuple(psi.shape)} {psi.dtype} {R['psi_gib']} GiB", flush=True)

    # ---------------------------------------------------------------- training
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    N = psi.shape[0]
    curve, evals = [], []
    t1 = time.perf_counter()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (min(args.batch, N),))
        p_b = psi[idx].to(dev).to(torch.float32)
        a_b = act_n[idx].to(dev)
        loss = head.loss(a_b, p_b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            curve.append({"step": step, "loss": round(float(loss.item()), 6)})
        if step % args.eval_every == 0 or step == args.steps:
            ev = closed_loop_success(ingress, head, norm, args.eval_episodes,
                                     args.eval_seed, dev, steps=args.sample_steps)
            ev["step"] = step
            evals.append(ev)
            print(f"[vla] step={step:5d} loss={float(loss.item()):.5f} "
                  f"SUCCESS={ev['success_rate']:.4f} dist={ev['mean_final_dist']:.4f}",
                  flush=True)
    R["train_seconds"] = round(time.perf_counter() - t1, 1)
    R["steps_per_second"] = round(args.steps / max(R["train_seconds"], 1e-9), 2)
    R["loss_curve"] = curve
    R["loss_first"] = curve[0]["loss"] if curve else None
    R["loss_last"] = curve[-1]["loss"] if curve else None
    R["closed_loop_evals"] = evals

    # ---------------------------------------------------------------- verdicts
    final = evals[-1] if evals else {"success_rate": 0.0}
    rnd = R["baselines"]["random_policy_success"]
    R["final_success_rate"] = final["success_rate"]
    R["verdicts"] = {
        "beats_random_baseline": bool(final["success_rate"] > rnd),
        "random_baseline": rnd,
        "delta_over_random": round(final["success_rate"] - rnd, 4),
        "loss_decreased": bool(curve and curve[-1]["loss"] < curve[0]["loss"]),
        "loss_ratio_last_over_first": (
            round(curve[-1]["loss"] / curve[0]["loss"], 4)
            if curve and curve[0]["loss"] else None),
        "LOOP_CLOSES": bool(curve and curve[-1]["loss"] < curve[0]["loss"]
                            and final["success_rate"] > rnd),
    }
    if dev == "cuda":
        R["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    R["honest_limits"] = [
        "SYNTHETIC pixel task; success rate here is NOT robot capability",
        "no benchmark score is claimed (ARC/SciCode/LIBERO absent)",
        "behaviour cloning from an ANALYTIC expert, not from a real corpus",
        "the ingress sketch is frozen; only the flow head adapts",
    ]

    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print("\n[vla] WROTE", out, flush=True)
    print("[vla] VERDICTS:", json.dumps(R["verdicts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
