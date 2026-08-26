"""G2 lexical snapping carrier (frozen, default-OFF, zero trainable params).

Pre-registered 2026-08-25 (henri-g2-lexical-snap-prereg-20260825-001,
audit f7a66733b65f27d7; amendment henri-g2-lexical-snap-prereg-amendment-20260825-001).

Refuses to run without HENRI_LEXICAL_SNAP=1. Never imported by production.

Input: the G1 CONTINUOUS relational score vector s(q) = [s_1..s_n],
s_j = mean_b cos(q_b, k_j_b) against the frozen provenance-pinned char-engram
frame (public question fields only; evaluator fields structurally quarantined).
NOT legacy unbinder logits.

Hopfield lexical snap (pre-registered continuous energy, stable log-sum-exp):
    E(q) = -tau * logsumexp_k( s_k(q) / tau )
    p_k  = softmax_k( s_k(q) / tau )
    snap = argmax_k s_k(q)
tau = 0.125 (beta = 8.0) is a USER_SPECIFIED / project-specified diagnostic
value (corpus consult, INFERRED, bank ca4bb787); it is NOT tuned post-hoc.

Pre-snap telemetry is persisted BEFORE the snap for every arm and item:
raw scores, normalized pre-snap activations p, Hopfield energy, entropy,
top-1/top-2 margins, participation ratio (effective top-k diversity),
saturation indicator, snapped candidate id, and all frozen hashes.

Arms (all explicit):
  A        identity: q_i vs frozen frame keys; snap = argmax.
  A-repeat deterministic re-run; requires byte-identical telemetry.
  B        joint Spin(3) rotor on q AND all keys (valid Cl(3,0) gauge):
           pre-snap scores invariant within 1e-9; snap identical.
  C        query-only Spin(3) (keys fixed): pre-snap activations must move
           (max |delta s| > 1e-2).
  D        mismatched frame: keys per-block rotated by ARBITRARY O(8)
           (INVALID Cl(3,0) gauge, G0: grade_scramble ~5600). Negative
           control. Correct self-hit rate must collapse to <= 0.5.
  E        dead memory: score-equalized (all-zero) scores. Must NOT produce a
           confident lexical snap (p_top1 <= 0.5); empty memory fails closed.
  F        out-of-frame query control: queries from rows NOT in the frame;
           confident snap rate (p_top1 > 0.5) must be <= 0.5.
  A-restore re-run of A; requires byte-identical telemetry (frozen state).

Verdict precedence (pre-registered):
  ERROR_FAIL_CLOSED > FALSIFIED_NO_SELF_HIT (A < 1.0)
  > FALSIFIED_NO_PRE_SNAP_SIGNAL (nothing moves)
  > FALSIFIED_DEAD_MEMORY (dead memory snaps confidently)
  > FALSIFIED_MISMATCHED_KEYS_SURVIVE (D correct rate > 0.5)
  > LEXICAL_SNAP_ENGAGED (continuous relational evidence causally reaches
    lexical snapping).

LEXICAL_SNAP_ENGAGED is ENGAGEMENT ONLY. It is NOT semantic correctness,
NOT CEGIS capability, NOT AAII progress, NOT production promotion.

Diagnostic-only; NO correctness / AAII score / semantic composition claim.
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
    from .relational_egress import per_block_cosine_mean, rotor_sandwich_matrix
except ImportError:  # pragma: no cover - bare-script execution path
    from universal_wave_harness.envelope import (  # type: ignore
        EVALUATOR_ONLY_FIELDS, assert_no_evaluator_fields)
    from universal_wave_harness.ingress.text import TextWaveAdapter  # type: ignore
    from universal_wave_harness.gauge_audit import random_orthogonal  # type: ignore
    from universal_wave_harness.relational_egress import (  # type: ignore
        per_block_cosine_mean, rotor_sandwich_matrix)

CANONICAL_NUM_BLOCKS = 8192
DEFAULT_TAU = 0.125  # beta = 8.0, USER_SPECIFIED diagnostic value


def is_enabled() -> bool:
    return os.environ.get("HENRI_LEXICAL_SNAP", "0") == "1"


def scores_for(q: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Continuous relational score vector s(q) = [s_1..s_n] (G1 feature)."""
    return torch.tensor([per_block_cosine_mean(q, k) for k in keys],
                        dtype=torch.float64)


def hopfield_energy(scores: torch.Tensor, tau: float) -> float:
    """E(q) = -tau * logsumexp_k(s_k / tau), numerically stable."""
    s = scores.to(torch.float64)
    return float((-tau * torch.logsumexp(s / tau, dim=0)).item())


def pre_snap_stats(scores: torch.Tensor, tau: float) -> Dict:
    """Full pre-snap telemetry digest computed BEFORE any snap decision."""
    s = scores.to(torch.float64)
    if s.numel() == 0:
        raise ValueError("empty score vector: fail-closed (no vacuous all([]))")
    if not torch.isfinite(s).all():
        raise ValueError("non-finite score vector: fail-closed")
    e = hopfield_energy(s, tau)
    p = torch.softmax(s / tau, dim=0)
    ent = float(-(p * torch.log(p + 1e-300)).sum().item())
    ptop = torch.topk(p, k=min(5, p.numel()))
    stop = torch.topk(s, k=min(5, s.numel()))
    pm = float((ptop.values[0] - ptop.values[1]).item()) if p.numel() > 1 else 1.0
    sm = float((stop.values[0] - stop.values[1]).item()) if s.numel() > 1 else 0.0
    pr = float((p.sum() ** 2 / (p ** 2).sum()).item())
    return {
        "hopfield_energy": e,
        "entropy_nats": ent,
        "p_top1": float(ptop.values[0].item()),
        "p_top1_idx": int(ptop.indices[0].item()),
        "p_margin": pm,
        "s_margin": sm,
        "participation_ratio": pr,
        "top5_scores": stop.values.tolist(),
        "top5_scores_idx": stop.indices.tolist(),
        "scores": s.tolist(),
        "saturation": float((p > 0.99).sum().item()) / float(p.numel()),
        "snapped_id": int(s.argmax().item()),
        "tau": tau,
        "beta": 1.0 / tau,
    }


def memory_sha256(keys: torch.Tensor, frame_sha: str, encoder_sha: str) -> str:
    """Frozen lexical memory fingerprint: canonical float64 bytes + manifests."""
    h = hashlib.sha256()
    h.update(frame_sha.encode())
    h.update(encoder_sha.encode())
    h.update(keys.detach().to(torch.float64).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def implementation_sha256() -> str:
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _apply_blockwise(T: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """T [B,8,8] applied per block to x [n,B,8] -> [n,B,8] (or [B,8] -> [B,8])."""
    if x.ndim == 3:
        return torch.einsum("kab,nkb->nka", T.to(torch.float64),
                            x.to(torch.float64))
    return torch.einsum("kab,kb->ka", T.to(torch.float64), x.to(torch.float64))


def run_g2(csv_path: str, n: int, seed: int, out_dir: str,
           device: str, tau: float = DEFAULT_TAU) -> Dict:
    import csv as _csv
    import random as _random

    os.makedirs(out_dir, exist_ok=True)
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    rng = _random.Random(seed)
    need = min(2 * n, len(rows))
    subset = rng.sample(rows, need)
    in_rows = subset[:n]
    out_rows = subset[n:2 * n]

    if len(in_rows) < 2:
        return {"verdict": "ERROR_FAIL_CLOSED",
                "error": "insufficient rows for frame (n<2); no vacuous all([])"}

    adapter = TextWaveAdapter(device=device)

    def encode_row(mf: dict, item_id: str, uri: str) -> torch.Tensor:
        mf = {k: v for k, v in mf.items() if k not in EVALUATOR_ONLY_FIELDS}
        assert_no_evaluator_fields(mf)
        q = mf.get("question") or mf.get("prompt") or mf.get("text") or ""
        if not q:
            raise ValueError("row has no public question/prompt/text field")
        return adapter.encode(q, source_uri=uri, item_id=item_id).wave.to(device)

    # ---- frozen frame (leakage-safe: public fields only) ----
    try:
        keys = torch.stack([encode_row(r, str(i), "frame") for i, r in enumerate(in_rows)])
    except Exception as exc:
        return {"verdict": "ERROR_FAIL_CLOSED", "error": f"frame build failed: {exc}"}
    meta = []
    for r in in_rows:
        q = r.get("question") or r.get("prompt") or r.get("text") or ""
        meta.append({"question_id": r.get("question_id", ""), "question_len": len(q),
                     "question_sha": hashlib.sha256(q.encode("utf-8")).hexdigest()})
    frame_sha = hashlib.sha256()
    frame_sha.update(adapter.encoder_sha256.encode())
    for m in meta:
        frame_sha.update(m["question_sha"].encode())
    frame_sha = frame_sha.hexdigest()
    mem_sha = memory_sha256(keys, frame_sha, adapter.encoder_sha256)
    impl_sha = implementation_sha256()
    split_sha = hashlib.sha256(
        json.dumps([m["question_sha"] for m in meta], sort_keys=True).encode()).hexdigest()

    # query waves: same questions as the frame (identity content)
    q_waves = torch.stack([encode_row(r, str(i), "query") for i, r in enumerate(in_rows)])
    # out-of-frame queries for arm F
    f_waves = torch.stack([encode_row(r, str(n + i), "query-out") for i, r in enumerate(out_rows)])

    g = torch.Generator().manual_seed(seed * 7 + 1)
    n_items = n

    def spin3_blocks() -> torch.Tensor:
        T = []
        for _ in range(CANONICAL_NUM_BLOCKS):
            biv = int(torch.randint(0, 3, (1,), generator=g).item()) + 4
            th = float(torch.rand(1, generator=g).item()) * 1.5
            T.append(rotor_sandwich_matrix(th, biv))
        return torch.stack(T).to(device)

    def o8_blocks() -> torch.Tensor:
        return random_orthogonal(seed=seed * 13 + 5, dim=8,
                                 count=CANONICAL_NUM_BLOCKS).to(device)

    items_path = os.path.join(out_dir, "items.jsonl")
    with open(items_path, "w", encoding="utf-8") as f:
        f.flush()  # create file before any arm computation (incremental protocol)

        # ---- arm A: identity ----
        arm_a = []
        a_self_hits = 0
        for i in range(n_items):
            st = pre_snap_stats(scores_for(q_waves[i], keys), tau)
            arm_a.append(st)
            if st["snapped_id"] == i:
                a_self_hits += 1
            row = {"item": i, "arm": "A", **meta[i], **st,
                   "frame_sha256": frame_sha, "encoder_sha256": adapter.encoder_sha256,
                   "memory_sha256": mem_sha, "split_sha256": split_sha,
                   "impl_sha256": impl_sha}
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        a_hash = hashlib.sha256(json.dumps(arm_a, sort_keys=True).encode()).hexdigest()

        # ---- arm A-repeat ----
        arm_ar = []
        for i in range(n_items):
            st = pre_snap_stats(scores_for(q_waves[i], keys), tau)
            arm_ar.append(st)
            row = {"item": i, "arm": "A-repeat", **meta[i], **st,
                   "frame_sha256": frame_sha, "encoder_sha256": adapter.encoder_sha256,
                   "memory_sha256": mem_sha, "split_sha256": split_sha,
                   "impl_sha256": impl_sha}
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        a_rpt_hash = hashlib.sha256(json.dumps(arm_ar, sort_keys=True).encode()).hexdigest()
        repeat_identical = a_hash == a_rpt_hash

        # ---- arm B: joint Spin(3) on q AND keys ----
        Tb = spin3_blocks()
        qB = _apply_blockwise(Tb, q_waves)
        kB = _apply_blockwise(Tb, keys)
        b_err = 0.0
        for i in range(n_items):
            st0 = arm_a[i]
            st1 = pre_snap_stats(scores_for(qB[i], kB), tau)
            b_err = max(b_err, max(abs(x - y) for x, y in zip(st0["scores"], st1["scores"])))
            row = {"item": i, "arm": "B", **meta[i], **st1,
                   "joint_max_score_delta": max(abs(x - y) for x, y in zip(st0["scores"], st1["scores"])),
                   "frame_sha256": frame_sha, "encoder_sha256": adapter.encoder_sha256,
                   "memory_sha256": mem_sha, "split_sha256": split_sha, "impl_sha256": impl_sha}
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()

        # ---- arm C: query-only Spin(3) ----
        Tc = spin3_blocks()
        qC = _apply_blockwise(Tc, q_waves)
        c_max_delta = 0.0
        for i in range(n_items):
            st0 = arm_a[i]
            st1 = pre_snap_stats(scores_for(qC[i], keys), tau)
            c_max_delta = max(c_max_delta, max(abs(x - y) for x, y in zip(st0["scores"], st1["scores"])))
            row = {"item": i, "arm": "C", **meta[i], **st1,
                   "max_score_delta_vs_A": max(abs(x - y) for x, y in zip(st0["scores"], st1["scores"])),
                   "frame_sha256": frame_sha, "encoder_sha256": adapter.encoder_sha256,
                   "memory_sha256": mem_sha, "split_sha256": split_sha, "impl_sha256": impl_sha}
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()

        # ---- arm D: mismatched frame (per-block arbitrary O(8) on keys) ----
        Td = o8_blocks()
        kD = _apply_blockwise(Td, keys)
        d_hits = 0
        for i in range(n_items):
            st = pre_snap_stats(scores_for(q_waves[i], kD), tau)
            if st["snapped_id"] == i:
                d_hits += 1
            row = {"item": i, "arm": "D", **meta[i], **st,
                   "frame_sha256": frame_sha, "encoder_sha256": adapter.encoder_sha256,
                   "memory_sha256": mem_sha, "split_sha256": split_sha, "impl_sha256": impl_sha}
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        d_correct_rate = d_hits / n_items

        # ---- arm E: dead memory (score-equalized) ----
        dead = torch.zeros(n_items, dtype=torch.float64)
        e_stats = pre_snap_stats(dead, tau)
        e_confident = e_stats["p_top1"] > 0.5
        row = {"item": -1, "arm": "E", "question_id": "", "question_len": 0,
               "question_sha": "", **e_stats,
               "frame_sha256": frame_sha, "encoder_sha256": adapter.encoder_sha256,
               "memory_sha256": mem_sha, "split_sha256": split_sha, "impl_sha256": impl_sha}
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()

        # ---- arm F: out-of-frame queries ----
        f_confident = 0
        for i in range(min(len(f_waves), n_items)):
            st = pre_snap_stats(scores_for(f_waves[i], keys), tau)
            if st["p_top1"] > 0.5:
                f_confident += 1
            row = {"item": i, "arm": "F", **meta[i], **st,
                   "frame_sha256": frame_sha, "encoder_sha256": adapter.encoder_sha256,
                   "memory_sha256": mem_sha, "split_sha256": split_sha, "impl_sha256": impl_sha}
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        f_confident_rate = f_confident / max(1, min(len(f_waves), n_items))

        # ---- arm A-restore ----
        arm_ar2 = []
        for i in range(n_items):
            st = pre_snap_stats(scores_for(q_waves[i], keys), tau)
            arm_ar2.append(st)
            row = {"item": i, "arm": "A-restore", **meta[i], **st,
                   "frame_sha256": frame_sha, "encoder_sha256": adapter.encoder_sha256,
                   "memory_sha256": mem_sha, "split_sha256": split_sha, "impl_sha256": impl_sha}
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        a_restore_hash = hashlib.sha256(json.dumps(arm_ar2, sort_keys=True).encode()).hexdigest()
        restore_identical = a_hash == a_restore_hash

    # ---- verdict precedence (pre-registered) ----
    a_self_hit_rate = a_self_hits / n_items
    verdict = "LEXICAL_SNAP_ENGAGED"
    if not repeat_identical or not restore_identical:
        verdict = "ERROR_FAIL_CLOSED"
    elif not torch.isfinite(torch.tensor([st["hopfield_energy"] for st in arm_a])).all():
        verdict = "ERROR_FAIL_CLOSED"
    elif a_self_hit_rate < 1.0:
        verdict = "FALSIFIED_NO_SELF_HIT"
    elif c_max_delta < 1e-2 and b_err < 1e-9:
        verdict = "FALSIFIED_NO_PRE_SNAP_SIGNAL"
    elif e_confident:
        verdict = "FALSIFIED_DEAD_MEMORY"
    elif d_correct_rate > 0.5:
        verdict = "FALSIFIED_MISMATCHED_KEYS_SURVIVE"
    elif f_confident_rate > 0.5:
        verdict = "FALSIFIED_MISMATCHED_KEYS_SURVIVE"

    scorecard = {
        "verdict": verdict,
        "n": n_items, "seed": seed, "device": device, "tau": tau, "beta": 1.0 / tau,
        "tau_label": "USER_SPECIFIED/project-specified diagnostic (corpus INFERRED, bank ca4bb787); not tuned post-hoc",
        "a_self_hit_rate": a_self_hit_rate,
        "a_repeat_identical": repeat_identical,
        "a_restore_identical": restore_identical,
        "joint_max_score_delta": b_err,
        "query_only_max_score_delta": c_max_delta,
        "mismatched_frame_correct_rate": d_correct_rate,
        "dead_memory_p_top1": e_stats["p_top1"],
        "dead_memory_confident_snap": e_confident,
        "out_of_frame_confident_rate": f_confident_rate,
        "frame_sha256": frame_sha,
        "encoder_sha256": adapter.encoder_sha256,
        "memory_sha256": mem_sha,
        "split_sha256": split_sha,
        "impl_sha256": impl_sha,
        "prereg": "henri-g2-lexical-snap-prereg-20260825-001 f7a66733b65f27d7 + amendment 20260825-001",
        "no_claim": "ENGAGEMENT ONLY. No semantic correctness, no CEGIS capability, no AAII progress, no production promotion.",
    }
    with open(os.path.join(out_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=2)
    return scorecard


def main() -> None:
    ap = argparse.ArgumentParser(description="G2 lexical snapping carrier")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out-dir", default="artifacts/g2_lexical_snap")
    ap.add_argument("--device", default=None)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    args = ap.parse_args()

    if not is_enabled():
        print("BLOCKED_DEFAULT_OFF: set HENRI_LEXICAL_SNAP=1 to run")
        return

    sc = run_g2(args.csv, args.n, args.seed, args.out_dir, args.device, args.tau)
    print(json.dumps(sc, indent=2, default=str))


if __name__ == "__main__":
    main()
