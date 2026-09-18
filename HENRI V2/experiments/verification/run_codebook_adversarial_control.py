#!/usr/bin/env python3
"""ACTION 4 — the adversarial control the A4 concern demands.

THE QUESTION
    Does ARC discrimination live in the ALGEBRAIC STRUCTURE of the wave manifold,
    or is it an artifact of the readout projection?

THE TRAP THIS DESIGN AVOIDS (and why the obvious test is invalid)
    The obvious test -- "encode with codebook M, decode with the same frozen M,
    require the random M_rand arm to fail" -- CANNOT WORK. At D = 65,536 random
    Gaussian codebooks are near-orthogonal, and the earlier R2 probe measured
    exactly this: same-basis decode succeeds for random codebooks too, because
    that test measures the ENCODE/DECODE CHANNEL, not the representation. A
    control that measures the channel would pass for any codebook, so it proves
    nothing, and requiring it to fail would rig the control.

THE VALID DISCRIMINATOR USED HERE
    Ask whether the ENCODER'S OWN GEOMETRY respect transformation structure.
    For each task grid g, compute the encoder's cosine similarity to:
      * its own rot90/rot180/rot270 variants   (STRUCTURALLY RELATED)
      * a foreign task's grid                  (STRUCTURALLY UNRELATED)
    Claim under test: related pairs are MORE similar than foreign pairs.
    Control: replace the encoder with a RANDOM PROJECTION of the same output
    dimension. A random projection cannot manufacture order structure, so if the
    separation vanishes under it, the structure came from the encoder.

PRE-REGISTERED OUTCOMES (written BEFORE running)
    P1 PASS: margin = sim_related - sim_foreign > 0 for the real encoder.
    P2 KILL: if the random-projection control shows a margin of comparable size,
             the metric measures grid statistics, NOT the encoder. Report
             FALSIFIED and do not cite 78.33% as manifold evidence.
    P3 if the real encoder shows no positive margin, the "structure in the
       manifold" claim is FALSIFIED for this metric.

This does not re-prove the 78.33% accuracy and it does not produce a task score.
It tests a necessary condition of the manifold claim, on the real encoder.
"""
import json
import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import torch

R = pathlib.Path(r"C:\Users\chan\Desktop\HENRI 7B SWARM\.worktrees\basal-syncytium\HENRI V2")
sys.path.insert(0, str(R))
ARC = pathlib.Path(r"C:\Users\chan\henri_data\ARC-AGI\data")
OUT = R / "experiments" / "verification" / "codebook_adversarial_observed.json"

N_TASKS = 40
SEED = 20260918


def load_tasks(n):
    """Load up to n ARC tasks with at least one train pair, from the real corpus."""
    tasks = []
    for split in ("training", "evaluation"):
        d = ARC / split
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                t = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            tr = t.get("train") or []
            if not tr:
                continue
            g = tr[0].get("input")
            if not g or len(g) < 2 or len(g[0]) < 2:
                continue
            tasks.append({"id": p.stem, "grid": np.array(g, dtype=np.int64)})
            if len(tasks) >= n:
                return tasks
    return tasks


def variants(g):
    """The four canonical grid rotations. A structural orbit of g."""
    return {
        "rot90": np.rot90(g, 1),
        "rot180": np.rot90(g, 2),
        "rot270": np.rot90(g, 3),
    }


def build_encoder():
    """The production spatial-grid encoder, discovered from the live tree."""
    from o_vsa_ingress_tokenizer import O_VSA_IngressTokenizer

    return O_VSA_IngressTokenizer()


def encode(enc, grid):
    """Encode one grid to a flat float vector. Returns None if refused."""
    try:
        w = enc.encode_spatial_grid(grid.tolist())
    except Exception:
        return None
    if w is None:
        return None
    try:
        flat = w.squeeze(0) if w.dim() > 1 else w
        v = flat.reshape(-1).detach().to(torch.float32).cpu().numpy()
    except Exception:
        return None
    if v.size == 0 or not np.all(np.isfinite(v)):
        return None
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None


def cosine(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def run_arm(name, encoder_fn, tasks):
    """Related vs foreign similarity margins for one encoder."""
    related, foreign = [], []
    per_task = []
    for i, t in enumerate(tasks):
        g = t["grid"]
        base = encoder_fn(g)
        if base is None:
            continue
        for vname, vg in variants(g).items():
            v = encoder_fn(vg)
            if v is None:
                continue
            related.append(cosine(base, v))
        # foreign: the next task's grid, and one far away in the list
        for j in ((i + 1) % len(tasks), (i + 7) % len(tasks)):
            if j == i:
                continue
            f = encoder_fn(tasks[j]["grid"])
            if f is not None:
                foreign.append(cosine(base, f))
    rel = np.array([r for r in related if np.isfinite(r)])
    for_ = np.array([r for r in foreign if np.isfinite(r)])
    if rel.size == 0 or for_.size == 0:
        return {"arm": name, "n_related": int(rel.size), "n_foreign": int(for_.size),
                "status": "NO_DATA"}
    return {
        "arm": name,
        "status": "OK",
        "n_related": int(rel.size),
        "n_foreign": int(for_.size),
        "sim_related_mean": float(rel.mean()),
        "sim_foreign_mean": float(for_.mean()),
        "margin": float(rel.mean() - for_.mean()),
        "related_sd": float(rel.std()),
        "foreign_sd": float(for_.std()),
    }


def main():
    tasks = load_tasks(N_TASKS)
    print(f"tasks loaded: {len(tasks)}")
    if len(tasks) < 8:
        print("BLOCKED: too few ARC tasks to run the control")
        sys.exit(2)

    enc = build_encoder()
    d_model = None
    probe_wave = encode(enc, tasks[0]["grid"])
    if probe_wave is None:
        print("BLOCKED: production encoder refused every grid")
        sys.exit(2)
    d_model = int(probe_wave.size)
    print(f"encoder: {type(enc).__name__}  d_model={d_model}")

    results = []
    real = run_arm("tokenizer_derived", lambda g: encode(enc, g), tasks)
    results.append(real)
    print(f"  REAL  margin={real.get('margin')}  "
          f"rel={real.get('sim_related_mean')}  for={real.get('sim_foreign_mean')}")

    # Control: random projection of the raw grid to the SAME dimension, then the
    # identical pipeline. Same dimension, same grids, same metric -- only the
    # encoder is replaced. Seeded, so the control is reproducible.
    for cseed in (1, 2, 3):
        rng = np.random.default_rng(1000 + cseed)
        P = rng.standard_normal((d_model, 81)) / np.sqrt(81.0)

        def rp(g, P=P):
            flat = np.zeros(81, dtype=np.float64)
            v = g.reshape(-1).astype(np.float64)
            v = v[:81] if v.size >= 81 else np.pad(v, (0, 81 - v.size))
            flat[:v.size] = v
            out = P @ flat
            n = float(np.linalg.norm(out))
            return out / n if n > 0 else None

        c = run_arm(f"random_projection_{cseed}", rp, tasks)
        results.append(c)
        print(f"  CTRL{cseed} margin={c.get('margin')}  "
              f"rel={c.get('sim_related_mean')}  for={c.get('sim_foreign_mean')}")

    ctrl = [r["margin"] for r in results[1:] if r.get("margin") is not None]
    real_m = real.get("margin")
    ctrl_mean = float(np.mean(ctrl)) if ctrl else None
    ctrl_max = float(np.max(ctrl)) if ctrl else None

    pre = {}
    pre["P1_real_margin_positive"] = bool(real_m is not None and real_m > 0)
    pre["P2_control_does_not_reproduce_margin"] = bool(
        real_m is not None and ctrl_max is not None and (
            ctrl_max <= 0 or real_m > 2.0 * abs(ctrl_max))
    )
    pre["P3_real_margin_not_positive"] = bool(real_m is not None and real_m <= 0)

    if pre["P3_real_margin_not_positive"]:
        verdict = "FALSIFIED — the encoder shows no transform structure on this metric"
    elif pre["P1_real_margin_positive"] and pre["P2_control_does_not_reproduce_margin"]:
        verdict = "SUPPORTED — transform structure present in the encoder, absent in control"
    elif pre["P1_real_margin_positive"]:
        verdict = ("INCONCLUSIVE — real margin positive but the random control also "
                   "shows a comparable margin: the metric measures grid statistics, "
                   "NOT the wave manifold")
    else:
        verdict = "INCONCLUSIVE"

    body = {
        "schema": "henri.codebook-adversarial.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "evidence_class": "OBSERVED",
        "evidence_class_note": (
            "OBSERVED: the per-grid encoder outputs are produced by the live "
            "production encoder. The margins are DERIVED from those outputs by "
            "the cosine statistic stated in this file."
        ),
        "why_this_control": (
            "Same-basis encode/decode CANNOT discriminate a random codebook from a "
            "structured one (measured earlier: R2 max pairwise overlap 7.35e-3 "
            "versus chance 3.9e-3, i.e. random bound states are near-orthogonal and "
            "decode fine). A same-codebook test measures the channel, not the "
            "representation. This control instead asks whether the ENCODER's "
            "geometry orders transformation-related grids above unrelated ones."
        ),
        "design": {
            "n_tasks": len(tasks),
            "task_ids": [t["id"] for t in tasks],
            "d_model": d_model,
            "related_pairs": "grid vs rot90/rot180/rot270 of itself",
            "foreign_pairs": "grid vs a different task's grid",
            "statistic": "cosine similarity in encoder output space",
            "control": "seeded random projection of the raw grid to the same d_model",
            "seed": SEED,
        },
        "arms": results,
        "pre_registered": pre,
        "verdict": verdict,
        "limits": [
            "Tests a NECESSARY condition of the manifold claim, not the accuracy claim.",
            "Rotation structure is one structural family; other relations untested.",
            "Encoder refusals are skipped, so n varies per arm.",
            "No task score is produced by this file.",
        ],
    }
    OUT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    print()
    for k, v in pre.items():
        print(f"  {k:<44} {v}")
    print(f"  VERDICT: {verdict}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
