"""
System-1 v0.5.1 egress discriminator (training labels on verifier tests only).
=============================================================================
Pre-registered 2026-08-24 (reference system1-structural-egress-cegis.md,
v0.5.1 protocol: disjoint verifier/outcome tests, egress-state calibration,
matched ordering arms, cost + capability + calibration gates).

This module builds multi-test task splits and trains a Brier discriminator
on EXACT online egress candidate states with labels from VERIFIER tests only.
Outcome tests are NEVER consulted during selection or training.

Design decisions (all pre-registered):
- Multi-test splits: each task carries 8 generated tests.
  Partition: first 4 (indices 0..3) = VERIFIER tests (visible to CEGIS and
  discriminator training). Last 4 (indices 4..7) = OUTCOME tests (never seen
  during selection; used only to score the admitted program).
- Deterministic partition via seeded RNG; tests are generated IN ORDER per
  task so the partition is a fixed prefix/suffix split, reproducible and
  leakage-free.
- Discriminator input = the EXACT candidate state family used at inference:
  BOS+tokenize_code(code) -> backbone.encode_tokens -> 8-step core ->
  mean over slots (the same state the v0.4.1 energy head consumes). This is
  NOT raw token embeddings and NOT the signature latent.
- Labels = binary verifier-test outcome (sandbox passes ALL verifier tests).
- Backbone, tokenizer, FSA, grammar, candidate generator: FROZEN. Only the
  new discriminator head is trainable. Frozen-module audit asserts no
  backbone parameter requires grad.

Gates (checked in eval_v051_cegis.py, not here):
  calibration / CEGIS-operational / promotion (see that module).

Split hygiene:
  train_v051 : training labels only (seed 12391, tag train_v051)
  dev2_v051  : sealed BEFORE training, evaluated once (seed 90837, tag dev2_v051)
  smoke split: tag dev51_smoke (seed 44711), NEVER re-used.
  heldout40_v04 : consumed; code-level refusal via digest guard.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import pathlib
import random
import subprocess
import sys
import time

import torch
import torch.nn as nn

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, ID2TOK, System1KernelV04, tokenize_code, detokenize,
    KernelV04Config)
from system1_kernel_v05_ast_skeleton import (  # noqa: E402
    SkeletonGrammar, System1KernelV05, CORE_STEPS)
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, sig_ids, pad_tokens)

HELDOUT_DIGEST = "887d0d6c"      # consumed heldout40_v04 (sha prefix)
N_VERIFIER = 4
N_OUTCOME = 4
N_TESTS = N_VERIFIER + N_OUTCOME


# ---------------------------------------------------------------------------
# 1. Multi-test split builder (disjoint verifier/outcome partition)
# ---------------------------------------------------------------------------

def _expected(fid: int, args_list: list) -> object:
    """Reference implementation for the 13-family DSL (mirrors gen_task)."""
    a = args_list[0]
    if fid == 0:
        return sum(a)
    if fid == 1:
        return max(a)
    if fid == 2:
        return sum(1 for x in a if x > 0)
    if fid == 3:
        return tuple(sorted(set(args_list[0]) & set(args_list[1])))
    if fid == 4:
        return tuple(sorted(set(args_list[0]) | set(args_list[1])))
    if fid == 5:
        return [x + y for x, y in zip(args_list[0], args_list[1])]
    if fid == 6:
        return math.factorial(a)
    if fid == 7:
        return min(a)
    if fid == 8:
        return [abs(x) for x in a]
    if fid == 9:
        return sorted(a)
    if fid == 10:
        return sum(range(len(a)))
    if fid == 11:
        return [x - y for x, y in zip(args_list[0], args_list[1])]
    prod = 1
    for x in a:
        prod *= x
    return prod


def build_split(out_dir: str, n_tasks: int, seed: int, tag: str,
                n_families: int = 7) -> list[dict]:
    """Deterministic multi-test split. Creates tasks with 8 tests each
    (4 verifier + 4 outcome), a random-ish input set, and stores the
    canonical code + fp + tests + partition.

    n_families: how many DSL families to sample from (7 = v0.5.1/0.5.2
    compatibility; 13 = grammar-expansion cycle). Default 7 preserves
    prior split reproducibility exactly."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{tag}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    rng = random.Random(seed)
    tasks = []
    for i in range(n_tasks):
        t = gen_task(rng, fid=rng.randrange(n_families))
        name, fid, nargs = t["name"], t["fid"], t["nargs"]
        # generate 8 fresh inputs per task (4 verifier + 4 outcome),
        # guaranteeing cross-boundary input uniqueness so verifier/outcome
        # test STRINGS are provably disjoint (partition is airtight)
        verifier_args, outcome_args = [], []
        seen_inputs: set = set()
        for _ in range(N_VERIFIER):
            a = _rand_args(rng, fid)
            while _args_key(a) in seen_inputs:
                a = _rand_args(rng, fid)
            seen_inputs.add(_args_key(a))
            verifier_args.append(a)
        for _ in range(N_OUTCOME):
            a = _rand_args(rng, fid)
            while _args_key(a) in seen_inputs:
                a = _rand_args(rng, fid)
            seen_inputs.add(_args_key(a))
            outcome_args.append(a)
        tests = []
        for args_list in verifier_args + outcome_args:
            exp = _expected(fid, args_list)
            if nargs == 1:
                tests.append(f"assert {name}({args_list[0]}) == {exp}")
            else:
                tests.append(
                    f"assert {name}({tuple(args_list[0])}, {tuple(args_list[1])})"
                    f" == {tuple(exp) if isinstance(exp, tuple) else exp}")
        t["tests"] = tests          # 8 tests: [0:4]=verifier, [4:8]=outcome
        t["verifier_tests"] = tests[:N_VERIFIER]
        t["outcome_tests"] = tests[N_VERIFIER:]
        t["verifier_args"] = verifier_args
        t["outcome_args"] = outcome_args
        tasks.append(t)
    with open(p, "w") as f:
        json.dump(tasks, f, indent=1)
    return tasks


def _rand_args(rng: random.Random, fid: int) -> list:
    if fid == 0:
        return [[rng.randint(-10, 10) for _ in range(rng.randint(2, 8))]]
    if fid == 1:
        return [[rng.randint(-10, 10) for _ in range(rng.randint(2, 8))]]
    if fid == 2:
        return [[rng.randint(-10, 10) for _ in range(rng.randint(2, 8))]]
    if fid == 3:
        return [[rng.randint(-5, 5) for _ in range(rng.randint(3, 6))],
                [rng.randint(-5, 5) for _ in range(rng.randint(3, 6))]]
    if fid == 4:
        return [[rng.randint(-5, 5) for _ in range(rng.randint(3, 6))],
                [rng.randint(-5, 5) for _ in range(rng.randint(3, 6))]]
    if fid == 5:
        n = rng.randint(2, 6)
        return [[rng.randint(-10, 10) for _ in range(n)],
                [rng.randint(-10, 10) for _ in range(n)]]
    if fid == 6:
        return [rng.randint(0, 8)]
    if fid == 7:
        return [[rng.randint(-10, 10) for _ in range(rng.randint(2, 8))]]
    if fid == 8:
        return [[rng.randint(-10, 10) for _ in range(rng.randint(2, 8))]]
    if fid == 9:
        return [[rng.randint(-10, 10) for _ in range(rng.randint(2, 8))]]
    if fid == 10:
        return [[rng.randint(-10, 10) for _ in range(rng.randint(2, 8))]]
    if fid == 11:
        n = rng.randint(2, 6)
        return [[rng.randint(-10, 10) for _ in range(n)],
                [rng.randint(-10, 10) for _ in range(n)]]
    return [[rng.randint(1, 6) for _ in range(rng.randint(2, 7))]]


def _args_key(args_list: list) -> tuple:
    """Deterministic key for input uniqueness across the partition."""
    return tuple(tuple(a) if isinstance(a, list) else a
                 for a in args_list)


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 2. Candidate-state encoder (exact online egress family)
# ---------------------------------------------------------------------------

@torch.no_grad()
def candidate_state(v05: System1KernelV05, code: str, dev) -> torch.Tensor:
    """Exact candidate latent: BOS+tokenize -> encode -> 8-step core ->
    mean over slots. Same family as the energy head's training states."""
    ids = [TOK2ID["BOS"]] + tokenize_code(code)
    ids_t = torch.tensor([ids], dtype=torch.long, device=dev)
    z = v05.backbone.encode_tokens(ids_t)
    slow_cache = None
    for t in range(CORE_STEPS):
        z, slow_cache = v05.backbone.core(z, t, slow_cache)
    return z.mean(dim=1)                       # [1, d_slot]


# ---------------------------------------------------------------------------
# 3. Egress discriminator (trainable; backbone frozen)
# ---------------------------------------------------------------------------

class EgressDiscriminator(nn.Module):
    def __init__(self, d_slot: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_slot, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:   # [B, d]
        return self.net(z).squeeze(-1)                    # [B]


def freeze_audit(backbone: System1KernelV04) -> None:
    trainable = [n for n, p in backbone.named_parameters() if p.requires_grad]
    if trainable:
        raise SystemExit(f"FROZEN AUDIT FAILED: {trainable}")


# ---------------------------------------------------------------------------
# 4. Trainer (verifier-test labels only)
# ---------------------------------------------------------------------------

def collect_egress_data(v05: System1KernelV05, tasks: list[dict],
                        dev, budget: int = 64) -> tuple[list, list, list]:
    """Generate candidates per task (uniform), label by VERIFIER tests,
    return (states, labels, meta). States are the exact inference family."""
    v05.eval()
    states, labels, meta = [], [], []
    for t in tasks:
        sig_t = pad_tokens([sig_ids(t)], 16).to(dev)
        z0 = v05.backbone.encode_tokens(sig_t)
        sp = torch.zeros(1, 16, v05.backbone.cfg.d_slot, device=dev)
        cands = v05.generate_skeleton_candidates(
            z0, sp, t, top_k=budget, use_energy=False)
        for c in cands:
            st = candidate_state(v05, c["code"], dev)
            lab = sandbox(c["code"], t["verifier_tests"])
            states.append(st)
            labels.append(lab)
            meta.append({"task": t["name"], "code": c["code"],
                         "rule_id": c["rule_id"]})
    return states, labels, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-n", type=int, default=40)
    ap.add_argument("--train-seed", type=int, default=12391)
    ap.add_argument("--tag", default="train_v051")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--budget", type=int, default=64)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.train_seed)

    # split (train labels only)
    tasks = build_split(args.out, args.train_n, args.train_seed, args.tag)
    split_p = out / f"{args.tag}.json"
    sd = sha256_file(split_p)
    if sd.startswith(HELDOUT_DIGEST):
        raise SystemExit("REFUSED: split matches consumed heldout digest.")

    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05 = System1KernelV05(backbone).to(dev)
    v05.eval()
    freeze_audit(backbone)

    print(f"LOADED {args.ckpt} step={st.get('step')}", flush=True)
    print(f"TRAIN_SPLIT {args.tag} n={args.train_n} seed={args.train_seed} "
          f"sha={sd[:16]}", flush=True)

    # collect exact egress states + verifier labels
    states, labels, meta = collect_egress_data(v05, tasks, dev,
                                               budget=args.budget)
    X = torch.cat(states, dim=0)                     # [N, d]
    y = torch.tensor(labels, dtype=torch.float32, device=dev)
    print(f"EGRESS DATA {X.shape[0]} candidates, "
          f"pos={int(y.sum())} neg={int((1 - y).sum())}", flush=True)
    if int(y.sum()) == 0 or int((1 - y).sum()) == 0:
        raise SystemExit("CALIBRATION GATE: one class absent in training data.")

    disc = EgressDiscriminator(d_slot=backbone.cfg.d_slot).to(dev)
    opt = torch.optim.AdamW(disc.parameters(), lr=args.lr)
    n_trainable = sum(p.numel() for p in disc.parameters())
    print(f"DISCRIMINATOR trainable params: {n_trainable}", flush=True)

    # task-balanced batching
    task_ids = [m["task"] for m in meta]
    uniq = sorted(set(task_ids))
    task_idx = {u: i for i, u in enumerate(uniq)}
    batch_by_task = [[] for _ in uniq]
    for i, tid in enumerate(task_ids):
        batch_by_task[task_idx[tid]].append(i)

    t0 = time.time()
    for ep in range(args.epochs):
        disc.train()
        rng = random.Random(args.train_seed + ep)
        order = list(range(len(uniq)))
        rng.shuffle(order)
        total_loss = 0.0
        n_batches = 0
        for ti in order:
            idx = batch_by_task[ti]
            if not idx:
                continue
            # deterministic task-balanced batch
            for s in range(0, len(idx), 16):
                batch = idx[s:s + 16]
                xb = X[batch].to(dev)
                yb = y[batch]
                logit = disc(xb)
                loss = torch.mean((torch.sigmoid(logit) - yb) ** 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_batches += 1
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep + 1} brier={total_loss / max(n_batches, 1):.4f} "
                  f"t={time.time() - t0:.0f}s", flush=True)

    # save discriminator + frozen audit receipt
    torch.save({"disc": disc.state_dict(),
                "n_trainable": n_trainable,
                "train_split": args.tag,
                "split_sha": sd,
                "ckpt_sha": st.get("ckpt_sha", "11d56121"),
                "epochs": args.epochs,
                "trainable_params": n_trainable}, out / "disc_v051.pt")
    print(f"SAVED disc_v051.pt trainable={n_trainable}", flush=True)


if __name__ == "__main__":
    main()
