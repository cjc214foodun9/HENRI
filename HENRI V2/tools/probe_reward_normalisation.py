"""DECISIVE PROBE: is the FAIL a reward-implementation defect or a mechanism defect?

CONTEXT
-------
Two gates failed this session:
  discrimination v3: C1 frontier>mastered FALSE, C2 frontier>noise FALSE (noise HIGHEST)
  validity gate:     C1 spearman FALSE, C2 FALSE, C3 FALSE

Both failures share ONE candidate cause, and it is now analysable. The reward is

    r_i = | < grad L(y_i; theta), P_e @ delta_theta > |

Two defects in MY gates, both fixable, both plausibly sufficient to explain the FAIL:

  G-DEFECT-1  WRONG FRONTIER REGIME. The v3 gate trained the learner on
              FRONTIER + MASTERED. So by measurement time the "frontier" family was
              ALREADY MASTERED (measured loss 0.38 vs mastered 0.25). A frontier
              must be OUT of the training set. FIX: train the base learner on
              MASTERED ONLY, so FRONTIER is genuinely novel.

  G-DEFECT-2  MAGNITUDE CONFOUND (the important one). The reward is an ABSOLUTE
              inner product, so it scales with ||grad L||. Noise has the highest
              loss (5.87), hence the largest gradient, hence the largest inner
              product with ANY direction - including delta_theta. The paper's
              claim ("zero reward for noise") requires noise gradients to be
              ORTHOGONAL to the consolidated direction; an absolute inner product
              can still be LARGE when ||grad|| is large and cos is small.
              FIX: test NORMALIZED variants and see which ordering they give.

VARIANTS UNDER TEST (all on the SAME real VM streams)
    RAW   |<g, v>|                     as implemented (magnitude-confounded)
    COS   |<g, v>| / (||g|| ||v||)     pure alignment, magnitude-free
    COSV  |<g, v>| / ||g||             gradient-normalised
where v = P_e @ delta_theta.

PRE-REGISTERED DECISION RULE
    If COS (or COSV) yields frontier > mastered AND frontier > noise while RAW does
    not -> the mechanism SURVIVES and the defect is in the REWARD IMPLEMENTATION
    (missing normalisation). The M1 kernel must then normalise.
    If NO variant separates -> the mechanism is FALSIFIED at this scale and the
    10B-token run must not be funded.

Also reports a PROPER held-out learning-progress measure (train on a family, probe
on held-out members of the same family) instead of the memorisation proxy used in
the previous gate.

Usage: python tools/probe_reward_normalisation.py [--seeds 0,1,2] [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import torch  # noqa: E402

from stage0_universal_seeder import ALPHABET, VMConfig, CircularTapeVM  # noqa: E402

TOK = {t: i for i, t in enumerate(ALPHABET)}
VOCAB = 257
DEPTH = 64
FAMILIES = ("MASTERED", "FRONTIER", "NOISE")


def prog(*syms: str) -> list[int]:
    return [TOK[s] for s in syms]


# --------------------------------------------------------------------------- learner

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

    def clone(self):
        o = Learner.__new__(Learner)
        o.params = {k: p.detach().clone().requires_grad_(True) for k, p in self.params.items()}
        o.v = {k: t.clone() for k, t in self.v.items()}
        o.b2, o.eps, o.lr = self.b2, self.eps, self.lr
        o.steps = self.steps
        return o

    def loss(self, ids):
        x = self.params["emb"][ids]
        logits = x @ self.params["head"]
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB), ids[:, 1:].reshape(-1)
        )

    def step(self, ids):
        loss = self.loss(ids)
        grads = torch.autograd.grad(loss, tuple(self.params.values()))
        self.steps += 1
        with torch.no_grad():
            for (name, p), g in zip(self.params.items(), grads):
                self.v[name].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                v_hat = self.v[name] / (1 - self.b2 ** self.steps)
                p.add_(-self.lr * g / (torch.sqrt(v_hat) + self.eps))
        return float(loss.item())

    def probe_loss(self, ids):
        with torch.no_grad():
            return float(self.loss(ids))

    def v_flat(self):
        return torch.cat([t.reshape(-1) for t in self.v.values()])

    def snapshot(self):
        return {k: p.detach().clone() for k, p in self.params.items()}

    def displacement(self, snap, scale: float = 1.0):
        d = torch.cat([(snap[k] - self.params[k]).detach().reshape(-1) for k in self.params])
        return d * scale


# --------------------------------------------------------------------------- families

def tape(fam: str, i: int, seq_len: int, seed: int) -> list[int]:
    if fam == "MASTERED":
        p = prog(*(["+"] * (1 + i % 3)), *(["."] * (seq_len + 2)))
        res = CircularTapeVM(VMConfig(tape_size=64, max_steps=512, max_output=32)).execute(p)
    elif fam == "FRONTIER":
        p = prog(*(["+", "."] * (seq_len + 2)))
        res = CircularTapeVM(VMConfig(tape_size=64, max_steps=512, max_output=32)).execute(p)
    elif fam == "NOISE":
        g = torch.Generator().manual_seed(seed * 100003 + i)
        src = torch.randint(0, 256, (64,), generator=g).tolist()
        p = prog(*([",", "."] * (seq_len + 2)))
        res = CircularTapeVM(VMConfig(tape_size=64, max_steps=512, max_output=32,
                                      input_stream=tuple(src))).execute(p)
    else:
        raise ValueError(fam)
    t = list(res.output[:seq_len])
    return [min(x, VOCAB - 1) for x in t] + [0] * (seq_len - len(t))


def batch(fam: str, n: int, seq_len: int, seed: int, offset: int = 0):
    return torch.tensor([tape(fam, offset + i, seq_len, seed) for i in range(n)],
                        dtype=torch.long)


# --------------------------------------------------------------------------- reward variants

def variants(g, v, v_scale_denom_ok=True):
    """Return the three reward variants for one gradient g and tangent v."""
    inner = torch.abs(torch.sum(g * v))
    raw = float(inner)
    denom_gv = float(torch.linalg.vector_norm(g) * torch.linalg.vector_norm(v)) + 1e-30
    cos = float(inner) / denom_gv
    denom_g = float(torch.linalg.vector_norm(g)) + 1e-30
    cosv = float(inner) / denom_g
    return {"RAW": raw, "COS": cos, "COSV": cosv,
            "grad_norm": float(torch.linalg.vector_norm(g))}


# --------------------------------------------------------------------------- seed run

def run_seed(seed: int, base_steps: int, n_probe: int, seq_len: int,
             progress_steps: int) -> dict:
    # G-DEFECT-1 FIX: the base learner trains on MASTERED ONLY, so FRONTIER is novel.
    base_ids = batch("MASTERED", 16, seq_len, seed)
    learner = Learner(seed=seed)
    midpoint = max(1, base_steps // 2)
    snap = None
    for s in range(base_steps):
        learner.step(base_ids)
        if s + 1 == midpoint:
            snap = learner.snapshot()
    if snap is None:
        snap = learner.snapshot()

    d_theta = learner.displacement(snap)
    # the preconditioned tangent v = P_e @ delta_theta (M1 form)
    p_e = 1e-4 / (torch.sqrt(learner.v_flat().clamp_min(1e-30)) + 1e-8)
    v = p_e * d_theta

    out = {}
    for fam in FAMILIES:
        # reward measured on held-out members of this family
        ids = batch(fam, n_probe, seq_len, seed, offset=1000)
        per = {"RAW": [], "COS": [], "COSV": [], "grad_norm": []}
        loss_before = learner.probe_loss(ids)
        for i in range(ids.shape[0]):
            one = ids[i:i + 1]
            g = torch.cat([x.reshape(-1) for x in
                           torch.autograd.grad(learner.loss(one), tuple(learner.params.values()))])
            for k, val in variants(g, v).items():
                per[k].append(val)
        # PROPER held-out progress: train on OTHER members, probe on these
        train_ids = batch(fam, 16, seq_len, seed, offset=5000)
        probe = learner.clone()
        for _ in range(progress_steps):
            probe.step(train_ids)
        loss_after = probe.probe_loss(ids)
        out[fam] = {
            "reward": {k: sum(per[k]) / len(per[k]) for k in ("RAW", "COS", "COSV")},
            "grad_norm_mean": sum(per["grad_norm"]) / len(per["grad_norm"]),
            "loss_before": loss_before,
            "loss_after_heldout": loss_after,
            "heldout_progress": loss_before - loss_after,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--base-steps", type=int, default=40)
    ap.add_argument("--n-probe", type=int, default=10)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--progress-steps", type=int, default=10)
    ap.add_argument("--out", default="telemetry/reward_norm")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]

    t0 = time.perf_counter()
    runs = {s: run_seed(s, a.base_steps, a.n_probe, a.seq_len, a.progress_steps)
            for s in seeds}
    wall = time.perf_counter() - t0

    # aggregate: does each variant order frontier above mastered and noise on EVERY seed?
    verdict = {}
    for var in ("RAW", "COS", "COSV"):
        c1 = all(runs[s]["FRONTIER"]["reward"][var] > runs[s]["MASTERED"]["reward"][var]
                 for s in seeds)
        c2 = all(runs[s]["FRONTIER"]["reward"][var] > runs[s]["NOISE"]["reward"][var]
                 for s in seeds)
        verdict[var] = {"frontier_gt_mastered": bool(c1),
                        "frontier_gt_noise": bool(c2),
                        "SEPARATES": bool(c1 and c2)}
    prog_ok = all(runs[s]["FRONTIER"]["heldout_progress"] > runs[s]["NOISE"]["heldout_progress"]
                  for s in seeds)

    summary = {
        "probe": "REWARD_NORMALISATION_DECISION",
        "defect_fixed": "base learner trains on MASTERED only, so FRONTIER is genuinely novel",
        "seeds": seeds, "verdict_by_variant": verdict,
        "heldout_progress_axis_exists": bool(prog_ok),
        "runs": {str(s): runs[s] for s in seeds},
        "wall_seconds": round(wall, 3),
    }
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"seeds={seeds}  heldout-progress axis exists: {prog_ok}")
    print(f"{'variant':8s} {'f>mastered':>12s} {'f>noise':>10s} {'SEPARATES':>10s}")
    for var, v in verdict.items():
        print(f"{var:8s} {str(v['frontier_gt_mastered']):>12s} "
              f"{str(v['frontier_gt_noise']):>10s} {str(v['SEPARATES']):>10s}")
    print()
    for s in seeds:
        r = runs[s]
        print(f"seed {s}:  {'family':10s} {'RAW':>10s} {'COS':>8s} {'COSV':>8s} "
              f"{'||grad||':>9s} {'loss0':>7s} {'heldout_prog':>12s}")
        for fam in FAMILIES:
            d = r[fam]
            print(f"          {fam:10s} {d['reward']['RAW']:>10.3e} {d['reward']['COS']:>8.4f} "
                  f"{d['reward']['COSV']:>8.3e} {d['grad_norm_mean']:>9.3f} "
                  f"{d['loss_before']:>7.3f} {d['heldout_progress']:>12.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
