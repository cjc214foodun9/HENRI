"""Stage 0 bounded seeding run — the self-play loop end to end.

Protocol: HENRI-ARCH-2026-SELFPLAY-DREAMING-V1 (.md), MILESTONE 2, under the
ratified bounded-scope decision: 10^7-10^8 program EXECUTIONS first, measure
throughput, extrapolate cost, and only then decide on the scaled run.

HONEST BOUNDARY (state this in every report)
--------------------------------------------
This bounded run validates the LOOP PLUMBING and MEASURES THROUGHPUT. It does NOT
establish curriculum validity and does NOT establish ICL emergence:

  * The reward's ability to rank families by LEARNABILITY is NOT established.
    Two gates built this session FAILED, and both failures trace to gate defects
    of the same class: high-loss/novel data dominates any magnitude- or
    memorization-based score.
      - discrimination gate: a high-loss gradient aligns with the general
        loss-reduction direction, so reward alone cannot separate learnable
        structure from noise.
      - validity gate: progress was measured on the SAME samples used for
        training, which measures MEMORIZATION. Noise shows large "progress" by
        being memorized.
    A defect-free validity gate needs HELD-OUT loss. That gate is not built.
  * ICL emergence (reverse string, stack, associative recall) is the .md's claim
    for the 10B-token run. It is NOT testable at this scale.

SCALE-CONFLATION GUARD
----------------------
Three budgets are tracked SEPARATELY and never conflated:
    budget_vm_executions      program runs              (the cheap, approved budget)
    budget_reward_evaluations JVP/gradient per candidate (subsampled)
    budget_learner_tokens     next-byte training tokens  (a separate job at scale)

Usage:
  python stage0_seeding_run.py --n-executions 1000000 --batch-size 256 \
      --out telemetry/stage0_seeding --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import List

import torch

from henri_gradient_alignment_reward import alignment_reward
from stage0_universal_seeder import (
    ALPHABET,
    TIMEOUT_TOKEN,
    VMConfig,
    CircularTapeVM,
    sample_program,
)

VOCAB = 257  # 256 byte values + TIMEOUT
DEPTH = 64


# ======================================================================================
# learner (real AdamW; the M1 kernel supplies the preconditioner form)
# ======================================================================================

class TapeLearner:
    """Embedding + linear head predicting the next output-tape byte."""

    def __init__(self, seed: int = 0, lr: float = 3e-3) -> None:
        gen = torch.Generator().manual_seed(seed)
        self.params = {
            "emb": torch.randn(VOCAB, DEPTH, dtype=torch.float64, generator=gen) * 0.1,
            "head": torch.randn(DEPTH, VOCAB, dtype=torch.float64, generator=gen) * 0.1,
        }
        for p in self.params.values():
            p.requires_grad_(True)
        self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.b1, self.b2, self.eps = 0.9, 0.999, 1e-8
        self.lr = lr
        self.step_count = 0

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.params["emb"][ids]
        logits = x @ self.params["head"]
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
        )

    def step(self, ids: torch.Tensor) -> float:
        loss = self.loss(ids)
        grads = torch.autograd.grad(loss, tuple(self.params.values()))
        self.step_count += 1
        with torch.no_grad():
            for (name, p), g in zip(self.params.items(), grads):
                self.v[name].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                v_hat = self.v[name] / (1 - self.b2 ** self.step_count)
                p.add_(-self.lr * g / (torch.sqrt(v_hat) + self.eps))
        return float(loss.item())

    def v_flat(self) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in self.v.values()])

    def flat(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.params.values()])

    def snapshot(self) -> torch.Tensor:
        return self.flat().clone()


# ======================================================================================
# adaptive generator: uniform -> reward-weighted bank (epsilon-greedy)
# ======================================================================================

@dataclass
class ProgramBank:
    capacity: int = 4096
    programs: List[list] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    def admit(self, program: list, reward: float) -> None:
        self.programs.append(program)
        self.rewards.append(reward)
        if len(self.programs) > self.capacity:
            i = min(range(len(self.rewards)), key=lambda j: self.rewards[j])
            self.programs.pop(i)
            self.rewards.pop(i)

    def sample(self, rng: torch.Generator, n: int) -> List[list]:
        if not self.programs or n <= 0:
            return []
        w = torch.tensor(self.rewards, dtype=torch.float64).clamp_min(0.0)
        if float(w.sum()) <= 0:
            idx = torch.randint(0, len(self.programs), (min(n, len(self.programs)),), generator=rng)
        else:
            idx = torch.multinomial(w / w.sum(), min(n, len(w)), replacement=True, generator=rng)
        return [self.programs[i] for i in idx.tolist()]


# ======================================================================================
# the run
# ======================================================================================

def run_seeding(
    n_executions: int,
    batch_size: int,
    out_dir: str,
    seed: int,
    prog_len: int = 32,
    seq_len: int = 33,
    reward_subsample: int = 64,
) -> dict:
    """Execute the bounded seeding loop. Returns the summary dict."""
    os.makedirs(out_dir, exist_ok=True)
    vm = CircularTapeVM(VMConfig(tape_size=256, max_steps=512, max_output=seq_len - 1))
    learner = TapeLearner(seed=seed)
    rng = torch.Generator().manual_seed(seed)
    bank = ProgramBank()

    n_rounds = max(1, (n_executions + batch_size - 1) // batch_size)
    executed = 0
    reward_evals = 0
    tokens = 0
    timeouts = 0
    reward_sum = 0.0
    reward_n = 0
    # BOUNDED distinct-output accounting: an unbounded set at 10^7 executions
    # would hold ~1.25M tuples (~390 MB). Cap it and record saturation honestly.
    DISTINCT_CAP = 200_000
    distinct_saturated = False
    loss_hist: List[float] = []
    lookback: dict = {}
    t0 = time.perf_counter()

    telemetry_path = os.path.join(out_dir, "telemetry.jsonl")
    with open(telemetry_path, "w", encoding="utf-8") as fh:
        for rnd in range(n_rounds):
            # ---- 1. GENERATE: epsilon-greedy (30% from the reward-weighted bank)
            n_bank = int(0.3 * batch_size) if bank.programs else 0
            n_fresh = batch_size - n_bank
            batch = [sample_program(prog_len, rng) for _ in range(n_fresh)]
            batch += bank.sample(rng, n_bank)

            # ---- 2. EXECUTE (the cheap approved budget; total execution)
            results = vm.execute_batch(batch)
            executed += len(batch)
            timeouts += sum(1 for r in results if r.timed_out)

            # ---- 3. build the learner batch
            rows = []
            for r in results:
                seq = list(r.output[: seq_len - 1])
                if r.timed_out:
                    seq.append(TIMEOUT_TOKEN)
                seq = seq + [0] * (seq_len - len(seq))
                rows.append([min(x, VOCAB - 1) for x in seq[:seq_len]])
            ids = torch.tensor(rows, dtype=torch.long)

            # ---- 4. REWARD on a SUBSAMPLE (never one eval per execution)
            k = min(reward_subsample, ids.shape[0])
            sel = torch.randperm(ids.shape[0], generator=rng)[:k]
            sub = ids[sel]
            loss = learner.loss(sub)
            grads = torch.autograd.grad(loss, tuple(learner.params.values()))
            g = torch.cat([gr.reshape(-1) for gr in grads])
            p_e = learner.step_count // 2
            theta_lb = lookback.get(p_e, learner.snapshot())
            d_theta = theta_lb - learner.flat()
            r_mean = float(alignment_reward(g, d_theta, learner.v_flat()))
            reward_evals += k
            reward_sum += r_mean
            reward_n += 1

            # ---- 5. ADMIT the best-scoring programs to the bank
            for prog_item, res in zip(batch[: max(1, batch_size // 8)],
                                      results[: max(1, batch_size // 8)]):
                if not res.timed_out and res.output:
                    bank.admit(prog_item, r_mean)
                    if len(distinct) < DISTINCT_CAP:
                        distinct.add(tuple(res.output))
                    else:
                        distinct_saturated = True

            # ---- 6. LEARNER UPDATE (the separate token budget)
            loss_val = learner.step(ids)
            tokens += int(ids.numel())
            loss_hist.append(loss_val)
            lookback[learner.step_count] = learner.snapshot()
            if len(lookback) > 4:
                del lookback[min(lookback)]

            if rnd % 10 == 0 or rnd == n_rounds - 1:
                dt = time.perf_counter() - t0
                fh.write(json.dumps({
                    "round": rnd,
                    "vm_executions": executed,
                    "exec_per_sec": round(executed / dt, 1),
                    "reward_evals": reward_evals,
                    "learner_tokens": tokens,
                    "loss": round(loss_val, 6),
                    "reward_mean": round(reward_sum / max(reward_n, 1), 10),
                    "timeout_rate": round(timeouts / max(executed, 1), 4),
                    "bank_size": len(bank.programs),
                    "distinct_outputs": len(distinct),
                }) + "\n")
                fh.flush()

    dt = time.perf_counter() - t0
    summary = {
        "gate": "STAGE0_BOUNDED_SEEDING",
        "purpose": "loop plumbing + throughput measurement (NOT curriculum, NOT ICL)",
        "seed": seed,
        "n_executions_requested": n_executions,
        "budget_vm_executions": executed,
        "budget_reward_evaluations": reward_evals,
        "budget_learner_tokens": tokens,
        "wall_seconds": round(dt, 2),
        "exec_per_sec": round(executed / dt, 1),
        "reward_evals_per_sec": round(reward_evals / dt, 1),
        "learner_tokens_per_sec": round(tokens / dt, 1),
        "final_loss": round(loss_hist[-1], 6) if loss_hist else None,
        "first_loss": round(loss_hist[0], 6) if loss_hist else None,
        "reward_mean": round(reward_sum / max(reward_n, 1), 10),
        "timeout_rate": round(timeouts / max(executed, 1), 4),
        "distinct_outputs": len(distinct),
        "bank_size": len(bank.programs),
        "alphabet_size": len(ALPHABET),
        "honest_boundary": (
            "Validates loop plumbing and measures throughput only. Reward curriculum "
            "validity is UNRESOLVED (two gates failed on gate defects). ICL emergence "
            "is NOT testable at this scale."
        ),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-executions", type=int, default=1_000_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="telemetry/stage0_seeding")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-subsample", type=int, default=64)
    a = ap.parse_args()
    s = run_seeding(a.n_executions, a.batch_size, a.out, a.seed,
                    reward_subsample=a.reward_subsample)
    print(json.dumps(s, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
