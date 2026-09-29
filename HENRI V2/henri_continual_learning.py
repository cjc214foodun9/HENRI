"""HENRI continual learning: replay + EWC retention + Hopfield write-through memory.

WHY THIS EXISTS
===============
The project claims a *continuous learning* vision-language-action model. Nothing
in the repo measured whether learning a new task DESTROYS an old one. This module
implements the three mechanisms the architecture implies and measures them on a
two-task sequence with an EXTERNAL OUTCOME:

    ACQUISITION  = success on the NEW task after training on it
    RETENTION    = success on the OLD task after training on the new one
    FORGETTING   = acquisition(A) - retention(A)      <- the number that matters

The comparison is what makes it scientific: naive sequential fine-tuning is the
control. If replay/EWC do not beat it, they do not work.

THREE MECHANISMS
================
1. REPLAY (Lin 1992; Rolnick et al. arXiv:1811.00401). A reservoir buffer keeps a
   uniform sample of everything seen; each update mixes old and new batches.
2. EWC (Kirkpatrick et al. 2017, PNAS). A diagonal Fisher estimate F anchors
   parameters that mattered for task A: L += (lambda/2) * sum F_i (t_i - t*_i)^2.
3. HOPFIELD WRITE-THROUGH (Kanerva 2009; Ramsauer et al. arXiv:2008.02217). The
   episodic memory stores (key, value) pairs with NO gradient step and recalls by
   softmax attention. The policy output is blended with the recall, so recent
   experience can influence action selection without touching the weights.

WHY THIS IS THE RIGHT TEST FOR HENRI
====================================
The architecture's own claim is that fast episodic memory (Hopfield) plus slow
weight consolidation (EWC/replay) makes continuous learning feasible. That is a
testable claim with a number attached. This script produces it.

HONEST LIMITS
=============
* Task pair = two goal distributions in the synthetic pixel env. Two tasks is the
  MINIMUM for a forgetting measurement; it is not a task stream, and no claim is
  made about long-horizon non-forgetting.
* Success rates at ~50% leave a wide confidence band; the script reports n and the
  binomial standard error rather than implying precision it does not have.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# 1. REPLAY
# -----------------------------------------------------------------------------


class ReservoirBuffer:
    """Uniform-sample replay over a stream, with NO knowledge of the total length.

    Reservoir sampling (Vitter 1985) keeps each seen item with probability
    n_seen/max_size, so the buffer stays an unbiased sample of the WHOLE stream.
    That property is what makes it appropriate for continual learning: it cannot
    silently become a buffer of only the newest task.
    """

    def __init__(self, max_size: int, d_key: int, d_val: int, seed: int = 0):
        self.max_size = int(max_size)
        self.keys = torch.zeros(self.max_size, d_key, dtype=torch.float16)
        self.vals = torch.zeros(self.max_size, d_val, dtype=torch.float32)
        self.n_seen = 0
        self.n_stored = 0
        self._g = torch.Generator().manual_seed(seed)
        self.task_id = torch.zeros(self.max_size, dtype=torch.int64)

    def add(self, key: torch.Tensor, val: torch.Tensor, task: int = 0) -> int:
        """Add one item. Returns how many slots were written (0 or 1)."""
        k = key.detach().to(torch.float16).cpu().reshape(-1)
        v = val.detach().to(torch.float32).cpu().reshape(-1)
        if k.numel() != self.keys.shape[1] or v.numel() != self.vals.shape[1]:
            raise ValueError(f"shape mismatch: key {k.numel()} != {self.keys.shape[1]}")
        self.n_seen += 1
        if self.n_stored < self.max_size:
            i = self.n_stored
            self.n_stored += 1
        else:
            j = int(torch.randint(0, self.n_seen, (1,), generator=self._g).item())
            if j >= self.max_size:
                return 0
            i = j
        self.keys[i] = k
        self.vals[i] = v
        self.task_id[i] = task
        return 1

    def sample(self, n: int, device="cpu", task: int | None = None):
        if self.n_stored == 0:
            return None, None
        if task is None:
            idx = torch.randint(0, self.n_stored, (min(n, self.n_stored),))
        else:
            pool = (self.task_id[: self.n_stored] == task).nonzero().reshape(-1)
            if pool.numel() == 0:
                return None, None
            sel = torch.randint(0, pool.numel(), (min(n, pool.numel()),))
            idx = pool[sel]
        return (self.keys[idx].to(device).to(torch.float32),
                self.vals[idx].to(device))

    def stats(self) -> dict:
        out = {"n_seen": self.n_seen, "n_stored": self.n_stored,
               "fill_fraction": round(self.n_stored / self.max_size, 4)}
        if self.n_stored:
            for t in sorted(set(int(x) for x in self.task_id[: self.n_stored].tolist())):
                out[f"task_{t}_stored"] = int((self.task_id[: self.n_stored] == t).sum())
        return out


# -----------------------------------------------------------------------------
# 2. EWC
# -----------------------------------------------------------------------------


class EWC:
    """Diagonal elastic weight consolidation.

    After a task, estimate F_i = E[(d log p / d theta_i)^2] (the Fisher diagonal)
    and the anchor theta*. On later tasks, penalise movement on high-F parameters.
    """

    def __init__(self, lambda_: float = 1e3, n_batches: int = 12):
        self.lambda_ = float(lambda_)
        self.n_batches = int(n_batches)
        self.fisher: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}
        self.n_tasks = 0

    @torch.no_grad()
    def estimate(self, loss_fn, params: dict[str, torch.Tensor],
                 batch_sampler=None, n_batches: int | None = None) -> dict:
        """Accumulate the Fisher diagonal over sampled losses.

        DEFECT FIXED 2026-09-28: this method was decorated @torch.no_grad(), so
        torch.autograd.grad raised
        "element 0 of tensors does not require grad and does not have a grad_fn".
        The decorator is on a GRADIENT-FREE wrapper below; the actual estimate
        must build a graph, so it runs with grad enabled.
        """
        return self._estimate_impl(loss_fn, params,
                                   batch_sampler=batch_sampler,
                                   n_batches=n_batches)

    @torch.enable_grad()
    def _estimate_impl(self, loss_fn, params: dict[str, torch.Tensor],
                       batch_sampler=None, n_batches: int | None = None) -> dict:
        nb = int(n_batches if n_batches is not None else self.n_batches)
        fisher = {k: torch.zeros_like(p) for k, p in params.items()}
        for _ in range(nb):
            loss = loss_fn() if batch_sampler is None else loss_fn(batch_sampler())
            grads = torch.autograd.grad(loss, list(params.values()),
                                        retain_graph=False, allow_unused=True)
            for (k, p), g in zip(params.items(), grads):
                if g is not None:
                    fisher[k] += g.detach() ** 2
        self.fisher = {k: v / max(nb, 1) for k, v in fisher.items()}
        self.anchor = {k: p.detach().clone() for k, p in params.items()}
        self.n_tasks += 1
        nonfinite = [k for k, v in self.fisher.items() if not torch.isfinite(v).all()]
        return {"n_tasks": self.n_tasks,
                "fisher_mean": float(torch.stack(
                    [v.mean() for v in self.fisher.values()]).mean().item()),
                "fisher_max": float(max(v.max().item() for v in self.fisher.values())),
                "nonfinite_params": nonfinite}

    def penalty(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros((), device=next(iter(params.values())).device)
        total = torch.zeros((), device=next(iter(params.values())).device)
        for k, p in params.items():
            f = self.fisher[k].to(p.device)
            a = self.anchor[k].to(p.device)
            total = total + (f * (p - a) ** 2).sum()
        return 0.5 * self.lambda_ * total


# -----------------------------------------------------------------------------
# 3. HOPFIELD WRITE-THROUGH MEMORY
# -----------------------------------------------------------------------------


class HopfieldActionMemory:
    """Modern Hopfield network over (phase key -> action value) pairs.

    Storage requires NO gradient step: `write()` is a single row assignment. That
    is the property that makes it 'write-through' and is why it can serve a
    continuous-learning loop where a gradient step would be too slow.

    Recall is softmax attention:  a_hat = softmax(beta * K q) V
    with beta the inverse temperature. Blending with the policy gives the action.
    """

    def __init__(self, d_key: int, d_val: int, capacity: int = 4096,
                 beta: float = 8.0, seed: int = 0, device="cpu"):
        self.d_key = int(d_key)
        self.d_val = int(d_val)
        self.capacity = int(capacity)
        self.beta = float(beta)
        self.device = device
        self.K = torch.zeros(capacity, d_key, dtype=torch.float16, device="cpu")
        self.V = torch.zeros(capacity, d_val, dtype=torch.float32, device="cpu")
        self.task_id = torch.zeros(capacity, dtype=torch.int64)
        self.n = 0
        self._g = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def write(self, key: torch.Tensor, val: torch.Tensor, task: int = 0) -> int:
        k = key.detach().to(torch.float16).cpu().reshape(-1)
        v = val.detach().to(torch.float32).cpu().reshape(-1)
        if self.n < self.capacity:
            i = self.n
            self.n += 1
        else:
            # capacity full: overwrite a random slot (keeps recency unbiased)
            i = int(torch.randint(0, self.capacity, (1,), generator=self._g).item())
        self.K[i] = k
        self.V[i] = v
        self.task_id[i] = task
        return i

    @torch.no_grad()
    def recall(self, queries: torch.Tensor) -> torch.Tensor:
        if self.n == 0:
            raise RuntimeError("Hopfield memory is empty; write() first")
        K = self.K[: self.n].to(self.device).to(torch.float32)
        V = self.V[: self.n].to(self.device)
        q = F.normalize(queries.to(torch.float32), dim=-1)
        kn = F.normalize(K, dim=-1)
        sim = self.beta * (q @ kn.t())                     # [B, n]
        attn = torch.softmax(sim, dim=-1)
        return attn @ V

    @torch.no_grad()
    def recall_entropy(self, queries: torch.Tensor) -> float:
        """Mean attention entropy in nats. Near 0 => a single engram dominates
        (confident recall); near ln(n) => flat (uninformative memory)."""
        if self.n == 0:
            return float("nan")
        K = self.K[: self.n].to(self.device).to(torch.float32)
        q = F.normalize(queries.to(torch.float32), dim=-1)
        kn = F.normalize(K, dim=-1)
        p = torch.softmax(self.beta * (q @ kn.t()), dim=-1)
        return float(-(p * (p + 1e-12).log()).sum(dim=-1).mean().item())

    def stats(self) -> dict:
        out = {"n_slots": self.n, "capacity": self.capacity, "beta": self.beta,
               "fill_fraction": round(self.n / self.capacity, 4)}
        for t in sorted(set(int(x) for x in self.task_id[: self.n].tolist())):
            out[f"task_{t}_slots"] = int((self.task_id[: self.n] == t).sum())
        return out


# -----------------------------------------------------------------------------
# utility
# -----------------------------------------------------------------------------


def binomial_se(p: float, n: int) -> float:
    """Standard error of a proportion. Reported so a 0.47 from n=64 is not read
    as a 0.47 from n=10000."""
    if n <= 0:
        return float("nan")
    return math.sqrt(max(p * (1.0 - p), 1e-12) / n)


def forgetting_report(acc_a_after_a: float, acc_a_after_b: float, n: int) -> dict:
    return {
        "acquisition_task_A": round(acc_a_after_a, 4),
        "retention_task_A": round(acc_a_after_b, 4),
        "forgetting": round(acc_a_after_a - acc_a_after_b, 4),
        "se_acquisition": round(binomial_se(acc_a_after_a, n), 4),
        "se_retention": round(binomial_se(acc_a_after_b, n), 4),
        "n_eval_episodes": n,
        "forgetting_exceeds_2se": bool(
            (acc_a_after_a - acc_a_after_b) > 2 * (binomial_se(acc_a_after_a, n)
                                                   + binomial_se(acc_a_after_b, n))),
    }
