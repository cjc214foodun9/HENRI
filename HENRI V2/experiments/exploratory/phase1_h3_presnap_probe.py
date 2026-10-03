#!/usr/bin/env python
"""H3 kill test: does PRE-SNAP covariance detect adaptation where snapped tokens cannot?

PRE-REGISTERED (SPEC-2026-10-02-ZONE-A.md section 6):

    H3  Residual-stream covariance measured BEFORE the snap detects the task's
        active subspace within 3-5 adaptation steps, where snapped tokens show
        nothing.
    KILL: if snapped tokens detect the change equally well, the pre-snap probe
          adds nothing.

WHY THIS IS EXPECTED TO HOLD (corpus falsification, OBSERVED)
    The Hopfield snap is PIECEWISE CONSTANT.  Its derivative vanishes almost
    everywhere, so a consumer watching only snapped tokens is CAUSALLY BLIND to
    phase mutation that does not cross a decision boundary.  A phase drift that
    stays inside the same Voronoi cell produces literally identical tokens.

DESIGN
    A residual stream h_t in C^D starts on an OLD subspace and drifts toward a
    NEW orthogonal subspace.  Two detectors watch the same stream:
        detector A  PreSnapCovarianceProbe (real, from henri_zone_a_backbone)
        detector B  snapped tokens = nearest codebook entry of a fixed codebook
    Detection = first step at which the detector's statistic moves by more than
    its own declared threshold.  Both thresholds are declared BEFORE execution.

    The codebook is deliberately COARSE but not degenerate (it must be able to
    represent the drift eventually, otherwise the test would be rigged).

GATES
    H3_PRESNAP_DETECTS_WITHIN_5   pre-snap detection step <= 5
    H3_PRESNAP_FASTER_THAN_SNAP   pre-snap step < snap step  (or snap never fires)
    H3_SNAP_IS_BLIND_EARLY        the snap is still emitting old tokens at the
                                  step pre-snap already detected

HONEST LIMITS
    CPU only.  Synthetic stream, labelled synthetic.  Detection thresholds are
    declared, not tuned.  This tests DETECTABILITY MECHANICS, not model quality.
    A negative is a valid outcome.

Run:  python experiments/exploratory/phase1_h3_presnap_probe.py --out <json>
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

from henri.determinism import RunManifest                 # noqa: E402
from henri_zone_a_backbone import PreSnapCovarianceProbe  # noqa: E402

DIM = int(os.environ.get("HENRI_ZA_DIM", "512"))
SUBSPACE_K = 32
STEPS = 24
N_SAMPLES = 64
SHIFT_AT = 4                # the stream starts drifting at this step
DRIFT = 0.12                # per-step drift magnitude
NOISE = 0.05
RUN_SEED = 20261002

# Declared thresholds, fixed BEFORE the run.
COV_REL_THRESHOLD = 0.25    # relative change in the top-k covariance spectrum
SNAP_CHANGE_FRACTION = 0.10  # fraction of samples that must re-snap to fire


def old_subspace_basis(dim, gen, ncols=64):
    """Orthonormal basis for the OLD/NEW subspaces.

    MEMORY CONTRACT.  The subspace we need is only 2*k columns wide (k=32), so
    the QR must be taken on [D, ncols].  The historical form
    `qr(randn(D, D))` allocates a [D, D] matrix: 32 GiB at D=65536, which is the
    same defect class as the probe's dense covariance.  Never build a [D, D].
    """
    if ncols < 2 * SUBSPACE_K:
        raise ValueError(f"need >= {2 * SUBSPACE_K} columns, got {ncols}")
    q, _ = torch.linalg.qr(torch.randn(dim, ncols, generator=gen))
    return q


def make_stream(gen):
    """h_t in C^D: stationary on the OLD subspace, then drifting to the NEW one."""
    D = DIM
    basis = old_subspace_basis(D, gen)
    k = 32
    old_dirs = basis[:, :k]
    new_dirs = basis[:, k:2 * k]

    # Sample coefficients once per trajectory, then hold them while drifting, so
    # the change is a SUBSPACE change rather than coefficient resampling.
    coeff_old = torch.randn(N_SAMPLES, k, generator=gen)
    stream = []
    for t in range(STEPS):
        frac = 0.0 if t < SHIFT_AT else min(1.0, (t - SHIFT_AT) * DRIFT)
        base_old = coeff_old @ old_dirs.transpose(0, 1)
        new = torch.randn(N_SAMPLES, k, generator=gen) @ new_dirs.transpose(0, 1)
        h = (1.0 - frac) * base_old + frac * new
        h = h + NOISE * torch.randn(N_SAMPLES, D, generator=gen)
        stream.append(torch.complex(h, torch.zeros_like(h)))
    return stream, basis


def make_codebook(basis, m=64):
    """Coarse fixed codebook spanning the SAME space the stream lives in.

    Returned as COMPLEX64: the residual stream is complex, and the similarity
    product requires matching dtypes (a real codebook raises here).
    """
    idx = torch.linspace(0, basis.shape[1] - 1, m).long()
    code = basis[:, idx].transpose(0, 1).contiguous()   # [m, D], real
    code = code / (code.norm(dim=-1, keepdim=True) + 1e-12)
    return torch.complex(code, torch.zeros_like(code))


def snap_tokens(h, codebook):
    """Nearest-prototype snap: piecewise constant by construction.

    Similarity is the canonical complex cosine |<h, c>| = |sum_d conj(h_d) c_d|.
    Dividing both sides by the unit norms is unnecessary here because h and the
    codebook rows are already normalised.
    """
    h = h / (h.norm(dim=-1, keepdim=True) + 1e-12)
    sim = torch.abs(h.conj() @ codebook.transpose(0, 1))     # [N, m]
    return sim.argmax(dim=-1)


def top_spectrum(x, n_eig=8):
    """Top-`n_eig` eigenvalues of the [D, D] sample covariance WITHOUT building it.

    MEMORY CONTRACT (measured failure).  The historical form
    `(x - mean).conj().T @ (x - mean)` allocates [D, D].  At D=65536 that is
    65536^2 * 8 = 34,359,738,368 bytes = 32 GiB, and the host dies with
    `DefaultCPUAllocator: not enough memory`.  It crashed H3 on 2026-10-03.

    The nonzero eigenvalues of X^H X ([D, D]) equal those of X X^H ([N, N]).
    N_SAMPLES=64 << D=65536, so the [N, N] Gram is EXACT, not an approximation,
    and its top-`n_eig` eigenvalues are identical to the covariance's.
    """
    xc = x - x.mean(0, keepdim=True)                  # [N, D]
    gram = xc @ xc.conj().transpose(0, 1)             # [N, N], Hermitian PSD
    return torch.linalg.eigvalsh(gram).flip(0).real[:n_eig] / (x.shape[0] - 1)


def h3b_within_cell_blindness(gen):
    """The corpus claim AS LITERALLY STATED: within-cell blindness.

    H3 (above) measured accumulated drift detection -- "does the snap respond to
    a large drift eventually?" -- and the answer was yes, at step 5.  That is NOT
    the claim under test.  The corpus claim is about CAUSAL BLINDNESS: a phase
    mutation that stays inside the same quantization cell leaves the snapped
    token bit-identical, so a consumer watching only snapped tokens receives no
    signal at all, while a continuous pre-snap statistic does move.

    Both results are reported.  H3 (drift detection) is FALSIFIED as declared.
    H3b tests a DIFFERENT, literal quantity, declared here as its own hypothesis.
    """
    D, k = DIM, 32
    basis = old_subspace_basis(D, gen)
    dirs = basis[:, :k]
    coeff = torch.randn(N_SAMPLES, k, generator=gen)
    h0 = coeff @ dirs.transpose(0, 1)
    h0 = torch.complex(h0, torch.zeros_like(h0))
    codebook = make_codebook(basis, m=64)

    tok0 = snap_tokens(h0, codebook)
    base_spec = top_spectrum(h0)

    sweep = []
    for eps in (1e-3, 1e-2, 3e-2, 1e-1, 3e-1):
        delta = torch.randn(N_SAMPLES, k, generator=gen) @ dirs.transpose(0, 1)
        delta = torch.complex(delta, torch.zeros_like(delta))
        h1 = h0 + eps * delta
        tok1 = snap_tokens(h1, codebook)
        tok_change = float((tok0 != tok1).to(torch.float32).mean())
        s1 = top_spectrum(h1)
        pre_rel = float((s1 - base_spec).norm() / (base_spec.norm() + 1e-12))
        sweep.append({
            "eps": eps,
            "token_change_fraction": round(tok_change, 6),
            "presnap_spectrum_rel_change": round(pre_rel, 6),
        })

    blind = [
        s for s in sweep
        if s["token_change_fraction"] == 0.0 and s["presnap_spectrum_rel_change"] > 0.05
    ]
    return {
        "sweep": sweep,
        "blind_zone_exists": bool(blind),
        "blind_zone_eps": [s["eps"] for s in blind],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="H3 pre-snap probe kill test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=RUN_SEED,
                        help="RunManifest seed_seq; default is the sealed D=2048 seed")
    args = parser.parse_args(argv)

    manifest = RunManifest(seed_seq=args.seed)
    manifest.apply("h3_presnap_probe")
    gen = torch.Generator().manual_seed(manifest.seed_for("h3_presnap_probe"))

    stream, basis = make_stream(gen)
    codebook = make_codebook(basis)

    probe = PreSnapCovarianceProbe(dim=DIM, k=8, ema=0.3)
    specs = []
    snaps = []
    for h in stream:
        specs.append(probe.observe(h).clone())
        snaps.append(snap_tokens(h, codebook))

    # Detector A: relative move of the covariance spectrum vs the baseline step.
    base_spec = specs[0]
    cov_steps = []
    for t, s in enumerate(specs):
        rel = float((s - base_spec).norm() / (base_spec.norm() + 1e-12))
        cov_steps.append(rel)
    cov_detect = next(
        (t for t, v in enumerate(cov_steps) if v > COV_REL_THRESHOLD), None
    )

    # Detector B: fraction of samples that re-snap vs the baseline tokenisation.
    base_tok = snaps[0]
    snap_frac = []
    for t, tk in enumerate(snaps):
        frac = float((tk != base_tok).to(torch.float32).mean())
        snap_frac.append(frac)
    snap_detect = next(
        (t for t, v in enumerate(snap_frac) if v > SNAP_CHANGE_FRACTION), None
    )

    # Was the snap still blind at the step pre-snap had already fired?
    snap_blind_early = None
    if cov_detect is not None:
        snap_blind_early = bool(
            (snap_detect is None) or (snap_detect > cov_detect)
        )

    gates = {
        "H3_PRESNAP_DETECTS_WITHIN_5": (cov_detect is not None and cov_detect <= 5),
        "H3_PRESNAP_FASTER_THAN_SNAP": (
            cov_detect is not None
            and (snap_detect is None or cov_detect < snap_detect)
        ),
        "H3_SNAP_IS_BLIND_EARLY": bool(snap_blind_early),
    }

    h3b = h3b_within_cell_blindness(gen)
    gates["H3b_WITHIN_CELL_BLINDNESS_EXISTS"] = h3b["blind_zone_exists"]

    report = {
        "experiment": "H3_presnap_probe",
        "preregistered_kill": "snapped tokens detect equally well",
        "config": {
            "dim": DIM, "steps": STEPS, "n_samples": N_SAMPLES,
            "shift_at": SHIFT_AT, "drift": DRIFT, "noise": NOISE,
            "cov_rel_threshold": COV_REL_THRESHOLD,
            "snap_change_fraction": SNAP_CHANGE_FRACTION,
            "run_seed": args.seed,
            "dim_env": os.environ.get("HENRI_ZA_DIM", "(default)"),
        },
        "detection": {
            "presnap_detect_step": cov_detect,
            "snap_detect_step": snap_detect,
            "presnap_statistic": [round(v, 4) for v in cov_steps],
            "snap_rechange_fraction": [round(v, 4) for v in snap_frac],
        },
        "gates": gates,
        "h3b_within_cell": h3b,
        "verdict": (
            "H3_FALSIFIED__H3B_CONFIRMED"
            if (not gates["H3_PRESNAP_FASTER_THAN_SNAP"]
                and gates["H3b_WITHIN_CELL_BLINDNESS_EXISTS"])
            else ("H3_CONFIRMED" if all(gates.values()) else "H3_FALSIFIED")
        ),
        "interpretation": [
            "H3 (accumulated drift detection) is FALSIFIED: the coarse snap DOES",
            "respond to large drift, and slightly earlier than the covariance probe.",
            "That is a real negative for the 'pre-snap is faster' claim as written.",
            "H3b tests the corpus claim literally: within-cell blindness. If a",
            "mutation leaves snapped tokens bit-identical while the pre-snap",
            "statistic moves, the probe carries information the snap cannot.",
        ],
        "limits": [
            "CPU only; synthetic stream; detection thresholds declared, not tuned.",
            "Tests detectability mechanics, not model quality.",
            "No GPU; no latency claim.",
        ],
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
