"""CHEAPEST-KILL GATE v3: does the M1 reward DISCRIMINATE real program families?

v1 FAILED -> traced to gate defects. v2 FAILED -> traced to ONE REMAINING DEFECT.
v3 fixes it. History is kept because the failure chain IS the evidence.

DEFECT CHAIN (each was a GATE defect, not a mechanism verdict)
--------------------------------------------------------------
D1 (v1) "NOISE" was not noise: VMConfig defaulted to input_stream=(0,) so a
        program that reads the tape emitted a deterministic zero-driven sequence.
        A control with no randomness cannot discriminate anything.
D2 (v1) The reward was applied PER BATCH. The formulation is per-sample:
        r_i = |<grad L(y_i), P_e dtheta>|. A batch gradient is an average and
        destroys per-sample discrimination.
D3 (v1) No vacuousness control: if reward(real tangent) ~= reward(random tangent)
        the "alignment" claim is not being measured at all.
D4 (v2) The lookback displacement had no consolidated checkpoint.
D5 (v2, THE REMAINING DEFECT) The noise fix was INCOMPLETE. `CircularTapeVM.
        execute()` resets `in_idx = 0` on EVERY call, and the input_stream is
        fixed at construction. So even with a random input_stream, every NOISE
        program emitted the SAME 16 bytes -- a CONSTANT stream, i.e. a mastered
        stream, i.e. the OPPOSITE of noise. v3 gives each noise program its OWN
        random input stream, so the emitted tapes genuinely differ.

THE GATE (pre-registered, STRICT)
---------------------------------
  C1  reward(FRONTIER) > reward(MASTERED)
  C2  reward(FRONTIER) > reward(NOISE)
  C3  reward(FRONTIER) > reward(RANDOM TANGENT)      [non-vacuous]

If C2 or C3 fails on a DEFECT-FREE gate, the Stage-0 self-play premise is
FALSIFIED and the 10B-token budget must not be spent.

SCALE-CONFLATION GUARD: VM executions, reward evaluations and learner training
tokens are THREE separate budgets and are reported separately.

Usage: python stage0_reward_discrimination_gate.py [--seeds 0,1,2] [--out DIR]
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

    def snapshot(self):
        return {k: p.detach().clone() for k, p in self.params.items()}

    def v_flat(self) -> torch.Tensor:
        return torch.cat([t.reshape(-1) for t in self.v.values()])

    def displacement(self, snap) -> torch.Tensor:
        return torch.cat([(snap[k] - self.params[k]).detach().reshape(-1)
                          for k in self.params])

    def batched_loss(self, ids: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self.loss(ids))

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
# families -- EVERY stream is produced by real VM execution
# ======================================================================================

def plain_vm() -> CircularTapeVM:
    return CircularTapeVM(VMConfig(tape_size=64, max_steps=512, max_output=32))


def noise_vm_for(i: int, base_seed: int) -> CircularTapeVM:
    """D5 FIX: each noise program gets its OWN random input stream."""
    g = torch.Generator().manual_seed(base_seed * 100003 + i)
    src = torch.randint(0, 256, (64,), generator=g).tolist()
    return CircularTapeVM(VMConfig(tape_size=64, max_steps=512, max_output=32,
                                   input_stream=tuple(src)))


def tape_for(fam: str, i: int, seq_len: int, base_seed: int) -> list[int]:
    if fam == "MASTERED":
        k = 1 + (i % 3)
        p = prog(*(["+"] * k), *(["."] * (seq_len + 2)))          # constant byte
        res = plain_vm().execute(p)
    elif fam == "FRONTIER":
        p = prog(*(["+", "."] * (seq_len + 2)))                    # incrementing counter
        res = plain_vm().execute(p)
    elif fam == "FRONTIER_MOD":
        p = prog(*(["-", "."] * (seq_len + 2)))                    # mod-256 decrement
        res = plain_vm().execute(p)
    elif fam == "NOISE":
        p = prog(*([",", "."] * (seq_len + 2)))                    # own random source
        res = noise_vm_for(i, base_seed).execute(p)
    else:
        raise ValueError(fam)
    t = list(res.output[:seq_len])
    return [min(x, VOCAB - 1) for x in t] + [0] * (seq_len - len(t))


# ======================================================================================
# one seed of the gate
# ======================================================================================

def run_seed(seed: int, n_train_steps: int, n_eval: int, seq_len: int) -> dict:
    vm_exec = 0
    # training corpus: MIXED so a consolidation direction exists (mastered + frontier)
    rows = []
    for i in range(24):
        rows.append(tape_for("FRONTIER", i, seq_len, seed))
        rows.append(tape_for("MASTERED", i, seq_len, seed))
    vm_exec += len(rows)
    train_ids = torch.tensor(rows, dtype=torch.long)

    learner = Learner(seed=seed)
    midpoint = max(1, n_train_steps // 2)
    snap = None
    for s in range(n_train_steps):
        learner.step(train_ids)
        if s + 1 == midpoint:
            snap = learner.snapshot()
    if snap is None:
        snap = learner.snapshot()

    d_theta = learner.displacement(snap)

    per_family = {}
    reward_evals = 0
    for fam in ("MASTERED", "FRONTIER", "FRONTIER_MOD", "NOISE"):
        rows_f, losses = [], []
        for i in range(n_eval):
            t = tape_for(fam, i, seq_len, seed)
            vm_exec += 1
            ids = torch.tensor([t], dtype=torch.long)
            rows_f.append(ids)
            losses.append(learner.batched_loss(ids))
        ids_all = torch.cat(rows_f, dim=0)
        vals = learner.per_sample_reward(ids_all, d_theta)
        reward_evals += len(vals)
        per_family[fam] = {
            "reward_mean": sum(vals) / len(vals),
            "reward_max": max(vals),
            "loss_mean": sum(losses) / len(losses),
            "n": len(vals),
        }

    # vacuousness control: a random tangent of the SAME norm
    g = torch.Generator().manual_seed(seed + 777)
    d_rand = torch.randn(d_theta.shape, dtype=torch.float64, generator=g)
    d_rand = d_rand * (d_theta.norm() / (d_rand.norm() + 1e-12))
    ctrl = []
    for i in range(n_eval):
        t = tape_for("FRONTIER", i, seq_len, seed)
        ids = torch.tensor([t], dtype=torch.long)
        ctrl.extend(learner.per_sample_reward(ids, d_rand))
    ctrl_mean = sum(ctrl) / len(ctrl)

    frontier = max(per_family["FRONTIER"]["reward_mean"],
                   per_family["FRONTIER_MOD"]["reward_mean"])
    mastered = per_family["MASTERED"]["reward_mean"]
    noise = per_family["NOISE"]["reward_mean"]

    return {
        "seed": seed,
        "checks": {
            "C1_frontier_gt_mastered": bool(frontier > mastered),
            "C2_frontier_gt_noise": bool(frontier > noise),
            "C3_non_vacuous_vs_random_tangent": bool(frontier > ctrl_mean),
        },
        "frontier_mean": frontier, "mastered_mean": mastered, "noise_mean": noise,
        "random_tangent_mean": ctrl_mean,
        "ratios": {
            "frontier_over_mastered": frontier / mastered if mastered else None,
            "frontier_over_noise": frontier / noise if noise else None,
            "frontier_over_control": frontier / ctrl_mean if ctrl_mean else None,
        },
        "per_family": per_family,
        "budget_vm_executions": vm_exec,
        "budget_reward_evaluations": reward_evals,
        "budget_learner_train_steps": n_train_steps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="telemetry/discrim")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-train-steps", type=int, default=30)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--seq-len", type=int, default=16)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    t0 = time.perf_counter()
    runs = [run_seed(s, a.n_train_steps, a.n_eval, a.seq_len) for s in seeds]
    wall = time.perf_counter() - t0

    # a check passes only if it passes on EVERY seed
    agg_checks = {
        k: all(r["checks"][k] for r in runs) for k in runs[0]["checks"]
    }
    verdict = "PASS" if all(agg_checks.values()) else "FAIL"

    summary = {
        "gate": "STAGE0_REWARD_DISCRIMINATION_V3",
        "defect_fixed": "D5 noise family had an identical fixed input stream on every program",
        "seeds": seeds,
        "n_train_steps": a.n_train_steps,
        "n_eval": a.n_eval,
        "seq_len": a.seq_len,
        "verdict": verdict,
        "aggregate_checks": agg_checks,
        "runs": runs,
        "budget_vm_executions": sum(r["budget_vm_executions"] for r in runs),
        "budget_reward_evaluations": sum(r["budget_reward_evaluations"] for r in runs),
        "budget_learner_train_steps": sum(r["budget_learner_train_steps"] for r in runs),
        "wall_seconds": round(wall, 3),
        "honest_boundary": (
            "Tests reward DISCRIMINATION on real VM streams with a vacuousness control. "
            "It does NOT test ICL emergence and does NOT validate the 10B-token premise."
        ),
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary_v3.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"VERDICT: {verdict}   seeds={seeds}")
    print(f"aggregate checks: {agg_checks}")
    print(f"\n{'seed':>4} {'frontier':>10} {'mastered':>10} {'noise':>10} {'control':>10}")
    for r in runs:
        print(f"{r['seed']:>4} {r['frontier_mean']:>10.4e} {r['mastered_mean']:>10.4e} "
              f"{r['noise_mean']:>10.4e} {r['random_tangent_mean']:>10.4e}")
    print("\nper-family detail (seed 0):")
    for fam, d in runs[0]["per_family"].items():
        print(f"  {fam:14s} reward={d['reward_mean']:.4e}  loss={d['loss_mean']:.4f}")
    print(f"\nbudgets: vm_exec={summary['budget_vm_executions']} "
          f"reward_evals={summary['budget_reward_evaluations']} "
          f"train_steps={summary['budget_learner_train_steps']}  wall={wall:.2f}s")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
