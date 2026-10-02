#!/usr/bin/env python
"""H2 kill test: does the UNIVERSAL WEIGHT SUBSPACE hypothesis hold here?

PRE-REGISTERED (SPEC-2026-10-02-ZONE-A.md section 6):

    H2  Per-instance adapters confined to a universal basis reach the same
        held-out correlation as free adapters, at <= 1% of the parameters.
    KILL: subspace < free by the pre-registered margin.

SOURCE (arXiv:2512.05117, The Universal Weight Subspace Hypothesis)
    The claim is about LEARNED WEIGHT MATRICES: networks trained on diverse
    tasks converge to shared low-dimensional parametric subspaces, with a few
    principal directions capturing majority variance.

    An earlier revision of this harness tested the wrong object -- it asked
    whether the TARGET OPERATORS of different tasks share a subspace.  That is a
    different question and it answered "no" (held-out target captured only 7% of
    its energy in a 16-dim basis fit on 24 other tasks).  This file tests the
    actual claim in two parts.

PART A -- DIAGNOSTIC: do LEARNED parameter vectors share a low-dim subspace?
    Train K independent adapters (diagonal m in C^D) on K distinct tasks from the
    same initialisation.  Stack the learned solutions into a [K, D] matrix and
    SVD it.  M2 predicts the top-q directions capture most of the variance.
    CONTROL: the same stack built from random task-independent vectors must NOT
    concentrate -- otherwise "shared subspace" is an artefact of the geometry.

PART B -- OPERATIONAL CONSEQUENCE: does a subspace adapter match a free one?
    Fit the basis from K-1 learned adapters.  On a HELD-OUT task, train
        free      m in C^D learned directly           (2D real parameters)
        subspace  m = U c, c in C^q learned           (2q real parameters)
        shuffled  free class, permuted targets        (control)
    plus a DIRECT span diagnostic: project the held-out task's exact solution
    into the basis and report the residual.  This separates "the basis cannot
    hold the solution" from "the optimiser failed".

HONEST HYPOTHESIS (declared before the run)
    For an ELEMENTWISE task family the per-task solutions are near-orthogonal
    directions, so cross-task subspace sharing is not expected to transfer.
    If that is what the data show, the design consequence is explicit: do NOT
    rely on a shared basis for elementwise transforms -- carry a per-task
    low-PARAMETER full-rank diagonal adapter instead.  A negative here is a
    governance-relevant result, not a failure of the harness.

HONEST LIMITS
    CPU only.  Reduced dimension.  Diagonal hypothesis class, not a trained
    foundation backbone.  No ARC score.  No latency.  No GPU.

Run:  python experiments/exploratory/phase1_h2_subspace_adapter.py --out <json>
Exit: 0 if every gate passes; 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_V2 = _HERE.parents[2]
sys.path.insert(0, str(_V2))

from henri.ingress.spatial_tokenizer import SpatialCliffordTokenizer  # noqa: E402
from henri.determinism import RunManifest                            # noqa: E402

MODULUS = 8
NUM_BLOCKS = int(os.environ.get("HENRI_ZA_NUM_BLOCKS", "256"))
BLOCK_SLOTS = 8
DIM = NUM_BLOCKS * BLOCK_SLOTS      # 2048 at default; 65536 at NUM_BLOCKS=8192
Q = 16                              # subspace dimension under test
N_TASKS = 24
N_FIT = 96
N_TEST = 32
STEPS = 600
LR = 5e-2
RUN_SEED = 20261002
MARGIN = 0.05

TASK_TRIPLES = [
    (1, 2, 3), (0, 1, 1), (2, 0, 5), (1, 1, 2), (3, 2, 1),
    (0, 2, 4), (2, 1, 6), (1, 0, 7), (3, 0, 2), (0, 3, 6),
    (2, 3, 1), (1, 3, 4), (3, 1, 5), (0, 0, 3), (2, 2, 7),
    (3, 3, 2), (1, 2, 6), (0, 1, 5), (2, 0, 1), (3, 2, 4),
    (1, 1, 7), (0, 2, 2), (2, 3, 5), (3, 0, 6),
]
TEST_TASK = (2, 2, 3)   # NOT in TASK_TRIPLES


# ------------------------------------------------------------------ utilities
def corr(a, b) -> torch.Tensor:
    """Mean |cosine| over the batch. TENSOR, so it stays in the autograd graph."""
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return (torch.conj(a) * b).sum(dim=-1).abs().mean()


def corr_value(a, b) -> float:
    with torch.no_grad():
        return float(corr(a, b))


def random_canvas(gen):
    return torch.randint(0, MODULUS, (MODULUS, MODULUS), generator=gen)


def make_dataset(tok, gen, dw, dh, k, n):
    xs, ys = [], []
    for _ in range(n):
        g = random_canvas(gen)
        t = tok.roll_canvas(((g + k) % MODULUS).tolist(), dw, dh)
        xs.append(tok.encode_canvas(g))
        ys.append(tok.encode_canvas(t))
    return torch.stack(xs), torch.stack(ys)


def exact_diagonal(tok, gen, dw, dh, k, n=64):
    """Least-squares recovery of the task's exact elementwise multiplier."""
    xs, ys = make_dataset(tok, gen, dw, dh, k, n)
    num = (torch.conj(xs) * ys).sum(dim=0)
    den = (xs.abs() ** 2).sum(dim=0) + 1e-9
    return num / den


def topq_energy(mat: torch.Tensor, q: int) -> float:
    """Fraction of spectral energy captured by the top-q directions."""
    s = torch.linalg.svdvals(mat)
    denom = float((s ** 2).sum()) + 1e-12
    return float((s[:q] ** 2).sum()) / denom


def train_diagonal(xs, ys, seed, basis=None, steps=STEPS, lr=LR, init_scale=1e-2):
    """Train a diagonal adapter; optionally constrained to a basis span.

    Both variants initialise at the SAME small-random magnitude.  An exact-zero
    initialisation traps the constrained arm: m = 0 gives a zero output and
    |cosine| has a zero subgradient at the origin, so it never moves.  That
    asymmetry produced a spurious FALSIFIED in an earlier revision.
    """
    torch.manual_seed(seed)
    if basis is None:
        p_re = torch.nn.Parameter(init_scale * torch.randn(DIM))
        p_im = torch.nn.Parameter(init_scale * torch.randn(DIM))
        params = [p_re, p_im]

        def forward(x):
            return torch.complex(p_re, p_im).unsqueeze(0) * x
    else:
        c_re = torch.nn.Parameter(init_scale * torch.randn(Q))
        c_im = torch.nn.Parameter(init_scale * torch.randn(Q))
        params = [c_re, c_im]

        def forward(x):
            m = basis @ torch.complex(c_re, c_im)
            return m.unsqueeze(0) * x
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = 1.0 - corr(forward(xs), ys)
        loss.backward()
        opt.step()
    return forward


def learned_solution(xs, ys, seed, steps=STEPS, lr=LR):
    """Train an unconstrained diagonal adapter and return its learned m in C^D."""
    torch.manual_seed(seed)
    p_re = torch.nn.Parameter(1e-2 * torch.randn(DIM))
    p_im = torch.nn.Parameter(1e-2 * torch.randn(DIM))
    opt = torch.optim.Adam([p_re, p_im], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        m = torch.complex(p_re, p_im)
        loss = 1.0 - corr(m.unsqueeze(0) * xs, ys)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return torch.complex(p_re, p_im).detach()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="H2 universal weight subspace kill test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=RUN_SEED,
                        help="RunManifest seed_seq; default is the sealed D=2048 seed")
    args = parser.parse_args(argv)

    manifest = RunManifest(seed_seq=args.seed)
    manifest.apply("h2_subspace_adapter")
    gen = torch.Generator().manual_seed(manifest.seed_for("h2_subspace_adapter"))

    tok = SpatialCliffordTokenizer(
        num_blocks=NUM_BLOCKS, block_slots=BLOCK_SLOTS, modulus=MODULUS
    )

    # ---------------- PART A: do LEARNED solutions share a subspace? --------
    train_triples = [t for t in TASK_TRIPLES if t != TEST_TASK]
    learned = []
    per_task_fit = []
    for i, (dw, dh, k) in enumerate(train_triples):
        xs, ys = make_dataset(tok, gen, dw, dh, k, N_FIT)
        m = learned_solution(xs, ys, seed=1000 + i)
        learned.append(m)
        per_task_fit.append(corr_value(m.unsqueeze(0) * xs, ys))
    M_learned = torch.stack(learned)                       # [K, D] complex

    g2 = torch.Generator().manual_seed(4242)
    M_random = torch.complex(
        torch.randn(*M_learned.shape, generator=g2),
        torch.randn(*M_learned.shape, generator=g2),
    )

    energy_learned = topq_energy(M_learned, Q)
    energy_random = topq_energy(M_random, Q)
    chance = Q / len(train_triples)

    # Basis for Part B.  M_learned is [K, D]; for a data matrix of that shape the
    # D-dimensional row space is spanned by the RIGHT singular vectors (rows of
    # vh), NOT by u -- u is [K, K] here.  Using u[:, :Q] would build a [24, 16]
    # "basis" and every projection would fail on shape.
    u, _s, vh = torch.linalg.svd(M_learned, full_matrices=False)
    basis = vh[:Q].conj().transpose(0, 1).contiguous()      # [D, Q]
    assert basis.shape == (DIM, Q), f"basis shape {tuple(basis.shape)} != ({DIM}, {Q})"

    # ---------------- PART B: operational consequence on a held-out task ----
    dw, dh, k = TEST_TASK
    xs, ys = make_dataset(tok, gen, dw, dh, k, N_FIT + N_TEST)
    xtr, ytr, xte, yte = xs[:N_FIT], ys[:N_FIT], xs[N_FIT:], ys[N_FIT:]

    d_test = exact_diagonal(tok, gen, dw, dh, k)
    proj = basis @ (basis.conj().transpose(0, 1) @ d_test)
    span_residual = float((d_test - proj).norm() / (d_test.norm() + 1e-12))
    span_energy = 1.0 - span_residual ** 2

    free_f = train_diagonal(xtr, ytr, seed=1)
    sub_f = train_diagonal(xtr, ytr, seed=1, basis=basis)
    perm = torch.randperm(N_FIT, generator=gen)
    shuf_f = train_diagonal(xtr, ytr[perm], seed=1)

    res = {
        "free_train": corr_value(free_f(xtr), ytr),
        "free_test": corr_value(free_f(xte), yte),
        "subspace_train": corr_value(sub_f(xtr), ytr),
        "subspace_test": corr_value(sub_f(xte), yte),
        "shuffled_test": corr_value(shuf_f(xte), yte),
        "mean_per_task_fit_learned": sum(per_task_fit) / len(per_task_fit),
    }

    free_params, sub_params = 2 * DIM, 2 * Q
    param_frac = sub_params / free_params

    gates = {
        # CORRECTED BASELINE.  An earlier revision compared against chance*4 with
        # chance = Q/K = 16/24 = 0.67, which is unreachable for an energy fraction
        # bounded by 1.  The meaningful baseline is the RANDOM control matrix of
        # identical shape, which shares the same spectral floor.
        "H2A_LEARNED_CONCENTRATES_VS_RANDOM": (energy_learned - energy_random) >= 0.10,
        "H2A_RANDOM_CONTROL_NEAR_CHANCE": abs(energy_random - chance) <= 0.15,
        "H2_SUBSPACE_MATCHES_FREE": res["subspace_test"] >= res["free_test"] - MARGIN,
        "H2_PARAMS_LE_1PCT": param_frac <= 0.01,
        "H2_SUBSPACE_BEATS_SHUFFLED": (res["subspace_test"] - res["shuffled_test"]) >= MARGIN,
        "H2_FREE_BEATS_SHUFFLED": (res["free_test"] - res["shuffled_test"]) >= MARGIN,
    }

    part_a_ok = gates["H2A_LEARNED_CONCENTRATES_VS_RANDOM"]
    if all(gates.values()):
        verdict = "H2_CONFIRMED"
    elif not part_a_ok:
        verdict = "H2A_NO_SHARED_SUBSPACE"
    else:
        verdict = "H2B_SUBSPACE_DOES_NOT_TRANSFER"

    report = {
        "experiment": "H2_universal_weight_subspace",
        "preregistered_kill": "subspace < free by margin",
        "config": {
            "dim": DIM, "q": Q, "n_tasks": len(train_triples),
            "test_task": list(TEST_TASK), "n_fit": N_FIT, "n_test": N_TEST,
            "margin": MARGIN, "run_seed": args.seed,
            "num_blocks_env": os.environ.get("HENRI_ZA_NUM_BLOCKS", "(default)"),
        },
        "part_a_learned_subspace": {
            "top_q_energy_learned": round(energy_learned, 6),
            "top_q_energy_random_control": round(energy_random, 6),
            "chance_level_q_over_k": round(chance, 6),
        },
        "part_b_transfer": {
            "results": {k2: round(v, 6) for k2, v in res.items()},
            "held_out_solution_span_energy": round(span_energy, 6),
            "held_out_solution_span_residual": round(span_residual, 6),
        },
        "params": {
            "free_real": free_params, "subspace_real": sub_params,
            "subspace_fraction": round(param_frac, 6),
        },
        "gates": gates,
        "verdict": verdict,
        "limits": [
            "CPU only; reduced dimension; no GPU; no latency claim.",
            "Diagonal hypothesis class, not a trained foundation backbone.",
            "M2 is a claim about learned weight matrices; this instantiates the",
            "closest testable analogue in HENRI's elementwise task family.",
            "No ARC score, no SOTA claim.",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
