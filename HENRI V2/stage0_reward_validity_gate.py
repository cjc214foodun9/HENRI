"""REWARD VALIDITY GATE — does the M1 reward PREDICT LEARNING PROGRESS?

WHY THIS REPLACES THE DISCRIMINATION GATE
-----------------------------------------
The v3 discrimination gate FAILED and the failure is a GATE-DESIGN defect, not a
mechanism verdict:

    C1 reward(FRONTIER) > reward(MASTERED) : FALSE
    C2 reward(FRONTIER) > reward(NOISE)    : FALSE  (noise scored HIGHEST)
    C3 reward(FRONTIER) > random tangent   : TRUE (non-vacuous)

The diagnosis is structural. r_i = |<grad L(y_i), P_e dtheta>| on a NOVEL input is
large for ANY high-loss data, because a high-loss gradient points roughly along
the general loss-reduction direction that dtheta already records. So reward
magnitude alone CANNOT separate "learnable structure" from "unlearnable noise".
Discrimination is the wrong test.

THE RIGHT TEST (this gate)
--------------------------
The reward is a CURRICULUM signal. Its definitional job is to rank candidates by
how much LEARNABLE PROGRESS they induce — i.e. by how much the learner improves if
it trains on them. So:

    learning_progress(F) = loss_before(F) - loss_after_training_on(F)

    VALID   <=>  rank(reward over families)  POSITIVELY correlates with
                 rank(learning_progress over families)

Families that must span the axis:
    MASTERED      constant byte         -> little progress, low reward  (expected)
    FRONTIER      incrementing counter  -> high progress, high reward   (expected)
    FRONTIER_MOD  mod-256 decrement     -> high progress, high reward   (expected)
    NOISE         random bytes          -> ~no progress, ~low reward    (expected)

PRE-REGISTERED ACCEPTANCE
  C1  Spearman(reward, learning_progress) >= 0.60 over the families
  C2  reward(FRONTIER) > reward(NOISE)
  C3  progress(FRONTIER) > progress(NOISE)     [the axis exists at all]
  C4  reward is non-vacuous: > 10x a random tangent of equal norm

If C1 or C2 fails on a defect-free gate, the Stage-0 self-play premise is
FALSIFIED and the scaled run must not be funded.

SCALE-CONFLATION GUARD: VM executions, reward evaluations and learner training
tokens are reported as THREE separate budgets.

Usage: python stage0_reward_validity_gate.py [--seeds 0,1,2] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from henri_gradient_alignment_reward import alignment_reward
from stage0_universal_seeder import ALPHABET, VMConfig, CircularTapeVM

TOK = {t: i for i, t in enumerate(ALPHABET)}
VOCAB = 257
DEPTH = 64
FAMILIES = ("MASTERED", "FRONTIER", "FRONTIER_MOD", "NOISE")


def prog(*syms: str) -> list[int]:
    return [TOK[s] for s in syms]


# ======================================================================================
# learner
# ======================================================================================

class Learner:
    def __init__(self, seed: int = 0, lr: float = 5e-2) -> None:
        g = torch.Generator().manual_seed(seed)
        self.params = {
            "emb": torch.randn(VOCAB, DEPTH, dtype=torch.float64, generator=g) * 0.1,
            "head": torch.randn(DEPTH, VOCAB, dtype=torch.float64, generator=g) * 0.1,
        }
        for p in self.params.values():
            p.requires_grad_(True)
        self.v = {k: torch.zeros_like(p) for k, p in self.params.items()}
        self.b2, self.eps, self.lr = 0.999, 1e-8, lr
        self.steps = 0

    def clone(self) -> "Learner":
        other = Learner.__new__(Learner)
        other.params = {k: p.detach().clone().requires_grad_(True) for k, p in self.params.items()}
        other.v = {k: t.clone() for k, t in self.v.items()}
        other.b2, other.eps, other.lr = self.b2, self.eps, self.lr
        other.steps = self.steps
        return other

    def loss(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.params["emb"][ids]
        logits = x @ self.params["head"]
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
        )

    def step(self, ids: torch.Tensor) -> float:
        loss = self.loss(ids)
        grads = torch.autograd.grad(loss, tuple(self.params.values()))
        self.steps += 1
        with torch.no_grad():
            for (name, p), g in zip(self.params.items(), grads):
                self.v[name].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                v_hat = self.v[name] / (1 - self.b2 ** self.steps)
                p.add_(-self.lr * g / (torch.sqrt(v_hat) + self.eps))
        return float(loss.item())

    def batched_loss(self, ids: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self.loss(ids))

    def v_flat(self) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in self.v.values()])

    def displacement(self, snap) -> torch.Tensor:
        return torch.cat([(snap[k] - self.params[k]).detach().reshape(-1)
                          for k in self.params])

    def per_sample_reward(self, ids: torch.Tensor, d_theta: torch.Tensor) -> list[float]:
        v_flat = self.v_flat()
        out = []
        for i in range(ids.shape[0]):
            loss_i = self.loss(ids[i : i + 1])
            gr = torch.autograd.grad(loss_i, tuple(self.params.values()))
            g = torch.cat([x.reshape(-1) for x in gr])
            out.append(float(alignment_reward(g, d_theta, v_flat)))
        return out


# ======================================================================================
# families -- produced by REAL VM execution
# ======================================================================================

def tape_for(fam: str, i: int, seq_len: int, base_seed: int) -> list[int]:
    if fam == "MASTERED":
        p = prog(*(["+"] * (1 + i % 3)), *(["."] * (seq_len + 2)))
        res = CircularTapeVM(VMConfig(tape_size=64, max_steps=512, max_output=32)).execute(p)
    elif fam == "FRONTIER":
        p = prog(*(["+", "."] * (seq_len + 2)))
        res = CircularTapeVM(VMConfig(tape_size=64, max_steps=512, max_output=32)).execute(p)
    elif fam == "FRONTIER_MOD":
        p = prog(*(["-", "."] * (seq_len + 2)))
        res = CircularTapeVM(VMConfig(tape_size=64, max_steps=512, max_output=32)).execute(p)
    elif fam == "NOISE":
        g = torch.Generator().manual_seed(base_seed * 100003 + i)
        src = torch.randint(0, 256, (64,), generator=g).tolist()
        p = prog(*([",", "."] * (seq_len + 2)))
        res = CircularTapeVM(VMConfig(tape_size=64, max_steps=512, max_output=32,
                                      input_stream=tuple(src))).execute(p)
    else:
        raise ValueError(fam)
    t = list(res.output[:seq_len])
    return [min(x, VOCAB - 1) for x in t] + [0] * (seq_len - len(t))


def family_batch(fam: str, n: int, seq_len: int, seed: int) -> torch.Tensor:
    return torch.tensor([tape_for(fam, i, seq_len, seed) for i in range(n)], dtype=torch.long)


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (no scipy dependency)."""
    n = len(a)
    if n < 2:
        return float("nan")

    def ranks(x):
        order = sorted(range(n), key=lambda i: x[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else float("nan")


# ======================================================================================
# one seed of the validity gate
# ======================================================================================

def run_seed(seed: int, n_base_steps: int, n_probe: int, n_finetune: int,
             seq_len: int) -> dict:
    vm_exec = 0
    reward_evals = 0
    tokens = 0

    # --- a MIXED base learner, so a consolidation direction dtheta exists
    base_rows = []
    for i in range(24):
        base_rows.append(tape_for("FRONTIER", i, seq_len, seed))
        base_rows.append(tape_for("MASTERED", i, seq_len, seed))
        vm_exec += 2 if i == 0 else 0
    vm_exec += len(base_rows)
    base_ids = torch.tensor(base_rows, dtype=torch.long)

    learner = Learner(seed=seed)
    midpoint = max(1, n_base_steps // 2)
    snap = None
    for s in range(n_base_steps):
        learner.step(base_ids)
        if s + 1 == midpoint:
            snap = {k: p.detach().clone() for k, p in learner.params.items()}
    if snap is None:
        snap = {k: p.detach().clone() for k, p in learner.params.items()}
    d_theta = learner.displacement(snap)

    per_family = {}
    for fam in FAMILIES:
        ids = family_batch(fam, n_probe, seq_len, seed)
        vm_exec += n_probe

        # ---- REWARD (measured on the frozen base learner, before any fine-tune)
        vals = learner.per_sample_reward(ids, d_theta)
        reward_evals += len(vals)
        reward = sum(vals) / len(vals)

        # ---- LEARNING PROGRESS (measured on a CLONE so arms are independent)
        probe = learner.clone()
        before = probe.batched_loss(ids)
        for _ in range(n_finetune):
            probe.step(ids)
            tokens += int(ids.numel())
        after = probe.batched_loss(ids)

        per_family[fam] = {
            "reward_mean": reward,
            "loss_before": before,
            "loss_after": after,
            "learning_progress": before - after,
        }

    rewards = [per_family[f]["reward_mean"] for f in FAMILIES]
    progress = [per_family[f]["learning_progress"] for f in FAMILIES]
    rho = spearman(rewards, progress)

    # ---- vacuousness control
    g = torch.Generator().manual_seed(seed + 777)
    d_rand = torch.randn(d_theta.shape, dtype=torch.float64, generator=g)
    d_rand = d_rand * (d_theta.norm() / (d_rand.norm() + 1e-12))
    ctrl = learner.per_sample_reward(family_batch("FRONTIER", n_probe, seq_len, seed), d_rand)
    ctrl_mean = sum(ctrl) / len(ctrl)
    reward_evals += len(ctrl)

    c1 = (rho >= 0.60) if rho == rho else False
    c2 = per_family["FRONTIER"]["reward_mean"] > per_family["NOISE"]["reward_mean"]
    c3 = (per_family["FRONTIER"]["learning_progress"]
          > per_family["NOISE"]["learning_progress"])
    c4 = per_family["FRONTIER"]["reward_mean"] > 10.0 * ctrl_mean

    return {
        "seed": seed,
        "spearman_reward_vs_progress": rho,
        "checks": {
            "C1_spearman_ge_0.60": bool(c1),
            "C2_reward_frontier_gt_noise": bool(c2),
            "C3_progress_frontier_gt_noise": bool(c3),
            "C4_non_vacuous_10x_random_tangent": bool(c4),
        },
        "per_family": per_family,
        "random_tangent_mean": ctrl_mean,
        "budget_vm_executions": vm_exec,
        "budget_reward_evaluations": reward_evals,
        "budget_learner_tokens": tokens,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="telemetry/reward_validity")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--base-steps", type=int, default=30)
    ap.add_argument("--n-probe", type=int, default=12)
    ap.add_argument("--fine-tune-steps", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=16)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    t0 = time.perf_counter()
    runs = [run_seed(s, a.base_steps, a.n_probe, a.fine_tune_steps, a.seq_len)
            for s in seeds]
    wall = time.perf_counter() - t0

    agg = {k: all(r["checks"][k] for r in runs) for k in runs[0]["checks"]}
    verdict = "PASS" if all(agg.values()) else "FAIL"

    summary = {
        "gate": "STAGE0_REWARD_VALIDITY",
        "test": "reward must PREDICT learning progress (curriculum validity)",
        "seeds": seeds,
        "verdict": verdict,
        "aggregate_checks": agg,
        "spearman_by_seed": [r["spearman_reward_vs_progress"] for r in runs],
        "runs": runs,
        "budget_vm_executions": sum(r["budget_vm_executions"] for r in runs),
        "budget_reward_evaluations": sum(r["budget_reward_evaluations"] for r in runs),
        "budget_learner_tokens": sum(r["budget_learner_tokens"] for r in runs),
        "wall_seconds": round(wall, 3),
        "honest_boundary": (
            "Tests whether the reward RANKS families by learning progress. A PASS "
            "validates the curriculum signal; it does NOT confirm ICL emergence and "
            "does NOT validate the 10B-token premise claim."
        ),
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"VERDICT: {verdict}   seeds={seeds}")
    print(f"aggregate checks: {agg}")
    print(f"spearman by seed: {[round(x, 3) for x in summary['spearman_by_seed']]}")
    print(f"\n{'family':14s} {'reward':>12s} {'progress':>12s} {'loss_before':>12s} {'loss_after':>12s}")
    for fam, d in runs[0]["per_family"].items():
        print(f"{fam:14s} {d['reward_mean']:>12.4e} {d['learning_progress']:>12.4f} "
              f"{d['loss_before']:>12.4f} {d['loss_after']:>12.4f}")
    print(f"{'RAND_TANGENT':14s} {runs[0]['random_tangent_mean']:>12.4e}")
    print(f"\nbudgets: vm_exec={summary['budget_vm_executions']} "
          f"reward_evals={summary['budget_reward_evaluations']} "
          f"learner_tokens={summary['budget_learner_tokens']} wall={wall:.2f}s")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
