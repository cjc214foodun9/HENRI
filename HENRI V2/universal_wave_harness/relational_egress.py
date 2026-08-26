"""G1 relational egress kernel (frozen, default-OFF, zero trainable params).

Pre-registered 2026-08-25 (henri-g1-relational-egress-prereg-20260825-001,
audit f6a1d267d1b6e4c0). Refuses to run without HENRI_RELATIONAL_EGRESS=1.
Never imported by the production runner.

Stage 1 (frame sub-gate): construct a leakage-safe semantic frame from PUBLIC
question fields only; if no such frame exists, emit BLOCKED_MISSING_SEMANTIC_FRAME.
Stage 2: relational scores s_j = mean_b cos(q_b, k_j_b) (per-block cosine Gram
feature). This feature is invariant under ANY per-block orthogonal transform
applied jointly to q and all keys (G0 measured joint err 1.19e-07) and is
sensitive when only q rotates (x-only delta > 1e-2).

Diagnostic-only: self-retrieval on public fields. NO correctness, NO AAII score,
NO semantic composition claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import torch

try:
    from .envelope import EVALUATOR_ONLY_FIELDS, assert_no_evaluator_fields
    from .ingress.text import TextWaveAdapter
    from .gauge_audit import random_orthogonal
except ImportError:  # pragma: no cover - bare-script execution path
    from universal_wave_harness.envelope import (  # type: ignore
        EVALUATOR_ONLY_FIELDS, assert_no_evaluator_fields)
    from universal_wave_harness.ingress.text import TextWaveAdapter  # type: ignore
    from universal_wave_harness.gauge_audit import random_orthogonal  # type: ignore

CANONICAL_NUM_BLOCKS = 8192
CANONICAL_BLOCK_DIM = 8


def is_enabled() -> bool:
    return os.environ.get("HENRI_RELATIONAL_EGRESS", "0") == "1"


def per_block_cosine_mean(q: torch.Tensor, k: torch.Tensor) -> float:
    """Gauge-safe relational score: mean over blocks of row cosine.

    Invariant under any per-block orthogonal transform applied jointly to
    q and k; sensitive when only one side transforms.
    """
    q = q.to(torch.float64)
    k = k.to(torch.float64)
    dot = (q * k).sum(dim=-1)
    qn = q.norm(dim=-1) + 1e-12
    kn = k.norm(dim=-1) + 1e-12
    return float((dot / (qn * kn)).mean().item())


def rotor_sandwich_matrix(theta: float, biv_idx: int) -> torch.Tensor:
    """8x8 matrix of Psi -> R Psi R^dagger for Cl(3,0) rotor R."""
    from .gauge_audit import left_mult_matrix, reversion_matrix, right_mult_matrix
    biv = torch.zeros(8, dtype=torch.float64)
    biv[biv_idx] = 1.0
    c = math_cos(theta / 2.0)
    s = math_sin(theta / 2.0)
    r = torch.zeros(8, dtype=torch.float64)
    r[0] = c
    r[biv_idx] = -s
    rv = reversion_matrix()
    return right_mult_matrix(rv @ r) @ left_mult_matrix(r)


def math_cos(x: float) -> float:
    import math
    return math.cos(x)


def math_sin(x: float) -> float:
    import math
    return math.sin(x)


def frame_from_rows(rows: List[dict], device: str) -> Tuple[torch.Tensor, List[dict], str]:
    """Stage-1 frame sub-gate: build key waves from PUBLIC fields only.

    Returns (keys [n, 8192, 8], item_meta, frame_sha256).
    Raises ValueError on any evaluator-only field in the model-facing rows.
    """
    model_facing = []
    for row in rows:
        mf = {k: v for k, v in row.items() if k not in EVALUATOR_ONLY_FIELDS}
        assert_no_evaluator_fields(mf)  # structural quarantine
        model_facing.append(mf)

    adapter = TextWaveAdapter(device=device)
    keys = []
    meta = []
    for mf in model_facing:
        q = mf.get("question") or mf.get("prompt") or mf.get("text") or ""
        if not q:
            raise ValueError("frame row has no public question/prompt/text field")
        packet = adapter.encode(q, source_uri="frame", item_id=mf.get("question_id", ""))
        keys.append(packet.wave.to(device))
        meta.append({"question_id": mf.get("question_id", ""),
                     "question_len": len(q),
                     "question_sha": hashlib.sha256(q.encode("utf-8")).hexdigest()})
    keys = torch.stack(keys)
    # frozen hash: codec manifest + per-key question hashes
    frame_sha = hashlib.sha256()
    frame_sha.update(adapter.encoder_sha256.encode())
    for m in meta:
        frame_sha.update(m["question_sha"].encode())
    return keys, meta, frame_sha.hexdigest()


def score_all(q: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    return torch.tensor([per_block_cosine_mean(q, k) for k in keys],
                        dtype=torch.float64)


def run_g1(csv_path: str, n: int, seed: int, out_dir: str,
           device: str) -> dict:
    import csv as _csv
    import random as _random

    os.makedirs(out_dir, exist_ok=True)
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    rng = _random.Random(seed)
    subset = rng.sample(rows, min(n, len(rows)))

    # ---- stage 1: frame sub-gate ----
    try:
        keys, meta, frame_sha = frame_from_rows(subset, device)
    except Exception as exc:
        result = {"verdict": "BLOCKED_MISSING_SEMANTIC_FRAME",
                  "error": str(exc)}
        with open(os.path.join(out_dir, "scorecard.json"), "w") as f:
            json.dump(result, f, indent=2)
        return result

    adapter = TextWaveAdapter(device=device)
    q_waves = []
    for mf in subset:
        mf = {k: v for k, v in mf.items() if k not in EVALUATOR_ONLY_FIELDS}
        assert_no_evaluator_fields(mf)
        q = mf.get("question") or mf.get("prompt") or mf.get("text") or ""
        packet = adapter.encode(q, source_uri="query", item_id=mf.get("question_id", ""))
        q_waves.append(packet.wave.to(device))
    q_waves = torch.stack(q_waves)

    # ---- stage 2: relational scores ----
    n_items = q_waves.shape[0]
    identity_scores = []
    self_hits = 0
    for i in range(n_items):
        s = score_all(q_waves[i], keys)
        identity_scores.append(s.tolist())
        if int(s.argmax().item()) == i:
            self_hits += 1
    self_hit_rate = self_hits / n_items

    # identity hash
    identity_hash = hashlib.sha256(
        json.dumps(identity_scores, sort_keys=True).encode()).hexdigest()

    # ---- joint gauge invariance: Spin3 rotors on q AND keys ----
    g = torch.Generator().manual_seed(seed * 7 + 1)
    joint_err = 0.0
    for trial in range(6):
        T = []
        for _ in range(CANONICAL_NUM_BLOCKS):
            biv = int(torch.randint(0, 3, (1,), generator=g).item()) + 4
            th = float(torch.rand(1, generator=g).item()) * 1.5
            T.append(rotor_sandwich_matrix(th, biv))
        T = torch.stack(T).to(device)
        qT = torch.einsum("kab,nkb->nka", T, q_waves.to(torch.float64))
        kT = torch.einsum("kab,nkb->nka", T, keys.to(torch.float64))
        for i in range(min(n_items, 6)):
            s0 = score_all(q_waves[i], keys)
            s1 = score_all(qT[i], kT)  # full key batch, jointly transformed
            joint_err = max(joint_err, float((s0 - s1).abs().max()))
    joint_err = float(joint_err)

    # ---- x-only sensitivity (rotate q only, keys fixed) ----
    T2 = []
    for _ in range(CANONICAL_NUM_BLOCKS):
        biv = int(torch.randint(0, 3, (1,), generator=g).item()) + 4
        th = float(torch.rand(1, generator=g).item()) * 1.5
        T2.append(rotor_sandwich_matrix(th, biv))
    T2 = torch.stack(T2).to(device)
    x_only_max_delta = 0.0
    for i in range(min(n_items, 6)):
        s0 = score_all(q_waves[i], keys)
        qx = torch.einsum("kab,kb->ka", T2, q_waves[i].to(torch.float64))
        s1 = score_all(qx, keys)
        x_only_max_delta = max(x_only_max_delta,
                               float((torch.tensor(s0) - torch.tensor(s1)).abs().max()))

    # ---- mismatched-key control: keys in a MISALIGNED gauge ----
    # Per-block arbitrary O(8) on keys only is NOT a valid joint Cl(3,0)
    # gauge (G0: grade_scramble ~5600). It decorrelates self-pairs to chance:
    # E[cos(v, Qv)] = 0 for random Q per block. Bounded-angle Spin(3) rotors
    # leave expected self-cos ~0.64 (measured 4/4 self-hits survive) — NOT a
    # valid mismatch control.
    T3 = random_orthogonal(seed=seed * 13 + 5, dim=8,
                           count=CANONICAL_NUM_BLOCKS).to(device)
    k_mismatch = torch.einsum("kab,nkb->nka", T3, keys.to(torch.float64))
    mismatched_hits = 0
    for i in range(n_items):
        s = score_all(q_waves[i], k_mismatch)  # full misaligned key batch
        if int(s.argmax().item()) == i:
            mismatched_hits += 1
    mismatched_self_hit_rate = mismatched_hits / n_items

    # ---- verdict ----
    if joint_err > 1e-6:
        verdict = "FALSIFIED_JOINT_INVARIANCE"
    elif x_only_max_delta < 1e-2:
        verdict = "FALSIFIED_NO_SENSITIVITY"
    elif self_hit_rate < 1.0:
        verdict = "FALSIFIED_NO_SELF_HIT"
    elif mismatched_self_hit_rate >= 0.5:
        verdict = "FALSIFIED_MISMATCHED_KEYS_RETAIN_DISCRIMINATION"
    else:
        verdict = "RELATIONAL_EGRESS_ENGAGED"

    scorecard = {
        "verdict": verdict,
        "self_hit_rate": self_hit_rate,
        "mismatched_self_hit_rate": mismatched_self_hit_rate,
        "joint_err": joint_err,
        "x_only_max_delta": x_only_max_delta,
        "identity_hash": identity_hash,
        "frame_sha256": frame_sha,
        "encoder_sha256": adapter.encoder_sha256,
        "n": n_items, "seed": seed, "device": device,
        "prereg": "henri-g1-relational-egress-prereg-20260825-001 f6a1d267d1b6e4c0",
    }
    with open(os.path.join(out_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=2)
    with open(os.path.join(out_dir, "items.jsonl"), "w") as f:
        for i in range(n_items):
            f.write(json.dumps({
                **meta[i],
                "identity_rank": int(torch.tensor(identity_scores[i]).argmax().item()),
            }) + "\n")
    return scorecard


def main() -> None:
    ap = argparse.ArgumentParser(description="G1 relational egress kernel")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out-dir", default="artifacts/g1_relational_egress")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not is_enabled():
        print("BLOCKED_DEFAULT_OFF: set HENRI_RELATIONAL_EGRESS=1 to run")
        return

    sc = run_g1(args.csv, args.n, args.seed, args.out_dir, args.device)
    print(json.dumps(sc, indent=2, default=str))


if __name__ == "__main__":
    main()
