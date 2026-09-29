"""HENRI continual-learning evaluation: does learning Task B destroy Task A?

THE QUESTION
============
The project claims a *continuous learning* VLA. That claim has a measurable
failure mode: catastrophic forgetting. This script measures it and tests whether
the architecture's own mechanisms (replay, EWC, Hopfield write-through) reduce it.

PROTOCOL
========
  Task A = reach goals in the LEFT half    Task B = reach goals in the RIGHT half
  Disjoint goal supports, so a policy cannot score on both by accident.

  Phase 1  train on A           -> eval A          = ACQUISITION(A)
  Phase 2  train on B           -> eval A, eval B  = RETENTION(A), ACQUISITION(B)
  FORGETTING = ACQUISITION(A) - RETENTION(A)

Four configurations, same seeds, same step count:
  NAIVE    sequential fine-tuning, no mechanism        <- the CONTROL
  REPLAY   + reservoir buffer mixed into each batch
  EWC      + diagonal Fisher penalty on high-F params
  BOTH     + replay and EWC together

The control is the point: if replay/EWC do not beat NAIVE, they do not work here,
and the script says so.

HOPFIELD WRITE-THROUGH
======================
Measured separately: engrams are written with NO gradient step and recalled by
softmax attention. Reported: recall cosine to the stored action, and mean
attention entropy (near 0 => one engram dominates).

HONEST LIMITS
=============
* Two tasks is the MINIMUM for a forgetting measurement. Not a task stream.
* Success rates near 0.5 carry a binomial standard error of ~0.06 at n=64; the
  script reports the SE and a 2-SE significance flag rather than implying
  precision it does not have.
* Synthetic pixel env: no robot claim.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_continual_learning import (EWC, HopfieldActionMemory, ReservoirBuffer,
                                      forgetting_report)
from henri_continuous_action_head import ActionNormalizer, FlowMatchingActionHead
from henri_vla_env import N_DOF, EnvConfig, PixelReachEnv
from henri_vla_vision_ingress import VisionPhaseIngress

D_MODEL = 65536
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "continual_receipt.json")


def resolve_out(cli):
    if cli:
        d = os.path.dirname(os.path.abspath(cli))
        if not os.path.isdir(d):
            print(f"MALFORMED --out: directory does not exist: {d}", file=sys.stderr)
            raise SystemExit(2)
        return os.path.abspath(cli)
    envd = os.environ.get("HENRI_RECEIPT_DIR")
    if envd:
        os.makedirs(envd, exist_ok=True)
        return os.path.join(envd, "continual_receipt.json")
    return DEFAULT_OUT


def make_env(region: str) -> PixelReachEnv:
    return PixelReachEnv(EnvConfig(goal_region=region))


def collect(task_region: str, n_episodes: int, seed: int):
    env = make_env(task_region)
    obs_l, act_l = [], []
    for i in range(n_episodes):
        for o, a in env.expert_episode(seed + i):
            obs_l.append(o.cpu())
            act_l.append(a.cpu())
    return torch.stack(obs_l), torch.stack(act_l)


@torch.no_grad()
def encode(ingress, obs, batch=8, device="cpu"):
    """pixels -> psi. DEFECT FIXED 2026-09-28: `obs` was left on CPU while the
    ingress buffers were on CUDA, raising "mat2 is on cuda:0, different from
    other tensors on cpu". The batch is moved to the device before the forward."""
    out = [ingress(obs[i:i + batch].to(device)).to(torch.float16).cpu()
           for i in range(0, obs.shape[0], batch)]
    return torch.cat(out, 0)


@torch.no_grad()
def success_rate(ingress, head, norm, task_region: str, n: int, seed: int,
                 device, steps=8) -> float:
    env = make_env(task_region)
    wins = 0
    for i in range(n):
        env.reset(seed + i)
        done = False
        info = {"success": False}
        while not done:
            psi = ingress(env.observe().unsqueeze(0).to(device))
            a = norm.denormalize(head.sample(psi, steps=steps)[0])
            _, _, done, info = env.step(a)
        wins += int(info["success"])
    return wins / n


def train_phase(head, psi, act_n, steps, batch, lr, device,
                buffer=None, ewc=None, replay_ratio=0.5, task_id=0,
                hopfield=None, collect_memory=False):
    """One training phase. Returns (loss_first, loss_last)."""
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],
                            lr=lr, weight_decay=0.0)
    N = psi.shape[0]
    first = last = None
    for step in range(1, steps + 1):
        n_new = batch if buffer is None or buffer.n_stored == 0 else \
            max(1, int(batch * (1 - replay_ratio)))
        idx = torch.randint(0, N, (min(n_new, N),))
        p_b = psi[idx].to(device).to(torch.float32)
        a_b = act_n[idx].to(device)
        loss = head.loss(a_b, p_b)

        if buffer is not None and buffer.n_stored > 0:
            n_old = batch - n_new
            ok, ov = buffer.sample(n_old, device=device)
            if ok is not None:
                loss = loss + head.loss(ov, ok)

        if ewc is not None and ewc.fisher:
            params = {k: p for k, p in head.named_parameters() if p.requires_grad}
            loss = loss + ewc.penalty(params)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        lv = float(loss.item())
        if step == 1:
            first = lv
        last = lv

        if collect_memory and hopfield is not None:
            with torch.no_grad():
                pp = psi[idx].to(device).to(torch.float32)
                aa = head.sample(pp[:1], steps=4)[0]
            hopfield.write(pp[0], aa, task=task_id)
    return first, last


def build(device, d_model: int = D_MODEL):
    ingress = VisionPhaseIngress(d_model=d_model, image_size=64,
                                 patch=16, d_patch=256).to(device).eval()
    head = FlowMatchingActionHead(n_dof=N_DOF, d_cond=512, d_hidden=1024,
                                  n_layers=4, phase_dim=d_model).to(device)
    return ingress, head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-episodes", type=int, default=96)
    ap.add_argument("--eval-episodes", type=int, default=64)
    ap.add_argument("--steps-per-task", type=int, default=800)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ewc-lambda", type=float, default=5e2)
    ap.add_argument("--buffer-size", type=int, default=2048)
    ap.add_argument("--memory-capacity", type=int, default=2048)
    ap.add_argument("--out", default=None)
    ap.add_argument("--d-model", type=int, default=D_MODEL)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.train_episodes, args.steps_per_task = 16, 150
        args.eval_episodes, args.buffer_size = 16, 256
        args.memory_capacity, args.d_model = 256, 4096

    out = resolve_out(args.out)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R: dict = {"schema": "henri.continual.v1", "device": dev, "config": vars(args),
               "tasks": {"A": "goal region LEFT", "B": "goal region RIGHT"}}
    if dev == "cuda":
        R["gpu"] = {"name": torch.cuda.get_device_name(0),
                    "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)}
        torch.cuda.max_memory_allocated()
    print(f"[cl] device={dev} smoke={args.smoke}", flush=True)

    # ------------------------------------------------------------------ data
    ingress, _ = build(dev, args.d_model)
    obsA, actA = collect("left", args.train_episodes, seed=0)
    obsB, actB = collect("right", args.train_episodes, seed=10000)
    norm = ActionNormalizer(N_DOF).fit(torch.cat([actA, actB], 0))
    psiA, psiB = encode(ingress, obsA, device=dev), encode(ingress, obsB, device=dev)
    nA, nB = norm.normalize(actA), norm.normalize(actB)
    R["data"] = {"n_A": int(psiA.shape[0]), "n_B": int(psiB.shape[0]),
                 "psi_dim": args.d_model}
    print(f"[cl] A samples={psiA.shape[0]}  B samples={psiB.shape[0]}", flush=True)

    configs = ["NAIVE", "REPLAY", "EWC", "BOTH"]
    results = {}
    for cfg in configs:
        torch.manual_seed(0)
        ingress, head = build(dev, args.d_model)
        base_params = None
        buf = ReservoirBuffer(args.buffer_size, args.d_model, N_DOF, seed=1) \
            if cfg in ("REPLAY", "BOTH") else None
        ewc = EWC(lambda_=args.ewc_lambda, n_batches=8) if cfg in ("EWC", "BOTH") else None
        hop = HopfieldActionMemory(args.d_model, N_DOF, capacity=args.memory_capacity,
                                   beta=8.0, device=dev) if cfg == "BOTH" else None

        # ---- phase 1: task A
        f1, l1 = train_phase(head, psiA, nA, args.steps_per_task, args.batch,
                             args.lr, dev, buf, None,
                             task_id=0, hopfield=hop, collect_memory=cfg == "BOTH")
        accA1 = success_rate(ingress, head, norm, "left", args.eval_episodes, 50000, dev)

        if buf is not None:
            n_add = min(psiA.shape[0], args.buffer_size)
            for i in range(n_add):
                buf.add(psiA[i], nA[i], task=0)

        if ewc is not None:
            def lf(batch=None):
                if batch is None:
                    b = torch.randint(0, psiA.shape[0], (128,))
                    return head.loss(nA[b].to(dev), psiA[b].to(dev).to(torch.float32))
                b = batch
                return head.loss(nA[b].to(dev), psiA[b].to(dev).to(torch.float32))

            def sampler():
                return torch.randint(0, psiA.shape[0], (128,))

            est = ewc.estimate(lf, {k: p for k, p in head.named_parameters()
                                    if p.requires_grad}, batch_sampler=sampler)
            base_params = copy.deepcopy({k: p.detach().clone()
                                         for k, p in head.named_parameters()})

        # ---- phase 2: task B
        f2, l2 = train_phase(head, psiB, nB, args.steps_per_task, args.batch,
                             args.lr, dev, buf, ewc,
                             task_id=1, hopfield=hop, collect_memory=cfg == "BOTH")
        accA2 = success_rate(ingress, head, norm, "left", args.eval_episodes, 50000, dev)
        accB2 = success_rate(ingress, head, norm, "right", args.eval_episodes, 60000, dev)

        if buf is not None:
            for i in range(min(psiB.shape[0], args.buffer_size)):
                buf.add(psiB[i], nB[i], task=1)

        rep = forgetting_report(accA1, accA2, args.eval_episodes)
        rep.update({"acquisition_task_B": round(accB2, 4),
                    "loss_A_first_last": [round(f1, 5), round(l1, 5)],
                    "loss_B_first_last": [round(f2, 5), round(l2, 5)],
                    "ewc_fisher": (est if ewc is not None and 'est' in dir() else None),
                    "mechanism_effective_vs_naive": None})
        if buf is not None:
            rep["buffer"] = buf.stats()
        if ewc is not None:
            rep["ewc"] = {"lambda": ewc.lambda_, "n_tasks_anchored": ewc.n_tasks}
        if hop is not None and hop.n > 0:
            idx = torch.randint(0, psiA.shape[0], (32,))
            q = psiA[idx].to(dev).to(torch.float32)
            rec = hop.recall(q)
            tgt = nA[idx].to(dev)
            rep["hopfield"] = hop.stats()
            rep["hopfield"]["recall_cosine"] = round(float(
                F.cosine_similarity(rec, tgt, dim=-1).mean().item()), 4)
            rep["hopfield"]["recall_attn_entropy_nats"] = round(hop.recall_entropy(q), 4)
        results[cfg] = rep
        print(f"[cl] {cfg:7s} ACC(A)={accA1:.4f} RET(A)={accA2:.4f} "
              f"FORGET={rep['forgetting']:+.4f} ACC(B)={accB2:.4f}", flush=True)

    R["results"] = results
    naive = results["NAIVE"]["forgetting"]
    for cfg in configs:
        if cfg == "NAIVE":
            continue
        results[cfg]["mechanism_effective_vs_naive"] = bool(
            results[cfg]["forgetting"] < naive - 0.02)
    R["verdicts"] = {
        "forgetting_naive": naive,
        "best_mechanism": min(configs, key=lambda c: results[c]["forgetting"]),
        "best_forgetting": min(results[c]["forgetting"] for c in configs),
        "any_mechanism_reduces_forgetting": bool(
            any(results[c]["mechanism_effective_vs_naive"] for c in configs
                if c != "NAIVE")),
        "significant_forgetting_naive": bool(naive > 0.02
                                             and results["NAIVE"]["forgetting_exceeds_2se"]),
    }
    if dev == "cuda":
        R["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
    R["honest_limits"] = [
        "TWO tasks only - the minimum for a forgetting measurement, not a stream",
        "success rates ~0.5 carry binomial SE ~0.06 at n=64; SE is reported",
        "synthetic pixel env; no robot or benchmark claim",
    ]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print("\n[cl] WROTE", out, flush=True)
    print("[cl] VERDICTS:", json.dumps(R["verdicts"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
