"""G0 gauge-group and information-feasibility audit probe (default-OFF).

Pre-registered 2026-08-25 (event henri-g0-gauge-audit-prereg-20260825-001,
audit c8b5239700ed6198). Refuses to run without HENRI_GAUGE_AUDIT=1.
Never imported by the production runner.

Measures, for the canonical [8192,8] Channel-T wave:
  A) grade_scramble_mean  - cross-grade operator energy per transformation arm
     (identity=0, per-block Spin(3) rotor sandwich ~0, arbitrary O(8) >> 0)
  B) invariant_collision_rate - collision of the ONLY O(8)^K invariants
     (per-block row norms; Channel-T rows are unit-norm cos phasors -> all 1)
  C) fixed_frame_discrimination - legacy linear unbinder argmax change rate
     under each arm (proves the coordinate head is not equivariant even
     under the VALID Spin(3) group)
  D) relational_stability_err - |sim(Gx, GK) - sim(x, K)| with the semantic
     frame transformed jointly (relational readout is the gauge-safe primitive)

Cl(3,0) basis: [1, e1, e2, e3, e12, e23, e31, e123] with e_i^2 = +1,
e_i e_j = -e_j e_i, bivectors/pseudoscalar square to -1.
Reversion: scalars/vectors +1, bivectors + pseudoscalar -1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Dict, List, Tuple

import torch

CANONICAL_NUM_BLOCKS = 8192
CANONICAL_BLOCK_DIM = 8
K_BINS = 256

# basis words in DEFINED order (e31 is defined as e3 e1, hence descending)
BASIS_WORDS: List[List[int]] = [
    [], [1], [2], [3], [1, 2], [2, 3], [3, 1], [1, 2, 3],
]
# grade of each basis element: 0:1, 1:vectors, 2:bivectors, 3:pseudoscalar
GRADE_OF: List[int] = [0, 1, 1, 1, 2, 2, 2, 3]


def _reduce_word(word: List[int]) -> Tuple[int, int]:
    """Return (mask, sign) for a generator word under Cl(3,0) rules.

    Sort ascending counting inversions; equal generators cancel with +1
    (e_i^2 = +1 in Cl(3,0)); then convert to the basis-defined word
    (e31 is defined descending -> conv_sign[0b101] = -1).
    """
    conv_sign = {0b101: -1}  # e31 = e3 e1 = -e1 e3
    w = list(word)
    inv = 0
    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            if w[i] > w[j]:
                inv += 1
    mask = 0
    for g in w:
        mask ^= 1 << (g - 1)
    sign = (-1) ** inv
    sign *= conv_sign.get(mask, 1)
    return mask, sign


def cl30_table() -> Dict[Tuple[int, int], Tuple[int, int]]:
    """Full 8x8 multiplication table: (i, j) -> (basis_index, sign)."""
    mask_to_idx = {0: 0, 1: 1, 2: 2, 4: 3, 3: 4, 6: 5, 5: 6, 7: 7}
    table = {}
    for i in range(8):
        for j in range(8):
            mask, sign = _reduce_word(BASIS_WORDS[i] + BASIS_WORDS[j])
            table[(i, j)] = (mask_to_idx[mask], sign)
    return table


TABLE = cl30_table()


def left_mult_matrix(r: torch.Tensor) -> torch.Tensor:
    """8x8 matrix L: (L @ x)_k = coefficient of e_k in (r * x)."""
    L = torch.zeros(8, 8, dtype=torch.float64)
    for i in range(8):
        for j in range(8):
            k, s = TABLE[(i, j)]
            L[k, j] += float(r[i]) * s
    return L


def right_mult_matrix(x: torch.Tensor) -> torch.Tensor:
    """8x8 matrix R: (R @ v)_k = coefficient of e_k in (v * x)."""
    R = torch.zeros(8, 8, dtype=torch.float64)
    for i in range(8):
        for j in range(8):
            k, s = TABLE[(i, j)]
            R[k, i] += float(x[j]) * s
    return R


def reversion_matrix() -> torch.Tensor:
    d = torch.ones(8, dtype=torch.float64)
    for i in range(8):
        if GRADE_OF[i] in (2, 3):
            d[i] = -1.0
    return torch.diag(d)


def rotor_sandwich(theta: float, biv_idx: int) -> torch.Tensor:
    """8x8 matrix of Psi -> R Psi R^dagger with R = exp(-theta/2 * e_biv)."""
    biv = torch.zeros(8, dtype=torch.float64)
    biv[biv_idx] = 1.0
    biv_sq_norm = biv[biv_idx] ** 2 * -1.0  # bivector squares to -1
    norm = abs(biv_sq_norm) ** 0.5
    c = math_cos(theta / 2.0)
    s = math_sin(theta / 2.0) / norm if norm > 0 else 0.0
    r = torch.zeros(8, dtype=torch.float64)
    r[0] = c
    r[biv_idx] = -s
    rv = reversion_matrix()
    L = left_mult_matrix(r)
    Rr = right_mult_matrix(rv @ r)
    return Rr @ L


def math_cos(x: float) -> float:
    import math
    return math.cos(x)


def math_sin(x: float) -> float:
    import math
    return math.sin(x)


def random_orthogonal(seed: int, dim: int = 8, count: int = 1) -> torch.Tensor:
    """QR-based random orthogonal matrices [count, dim, dim]."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(count):
        m = torch.randn(dim, dim, generator=g, dtype=torch.float64)
        q, r = torch.linalg.qr(m)
        d = torch.diag(r).sign()
        q = q * d
        out.append(q)
    return torch.stack(out)


def grade_projectors() -> torch.Tensor:
    P = torch.zeros(4, 8, 8, dtype=torch.float64)
    for i in range(8):
        P[GRADE_OF[i], i, i] = 1.0
    return P


def grade_scramble(T: torch.Tensor, P: torch.Tensor) -> float:
    """Cross-grade operator energy per block: sum_g ||P_g T P_g^perp||_F^2 / 8."""
    I = torch.eye(8, dtype=torch.float64)
    total = 0.0
    for g in range(4):
        total += float(((P[g] @ T @ (I - P[g])) ** 2).sum())
    return total / 8.0


def is_enabled() -> bool:
    return os.environ.get("HENRI_GAUGE_AUDIT", "0") == "1"


def _encode_questions(csv_path: str, n: int, seed: int, device: str):
    """Deterministic subset of question fields -> [n, 8192, 8] waves."""
    import csv as _csv

    try:
        from henri_goal_adapter import HenriPromptCodec
    except Exception as exc:  # pragma: no cover
        raise ImportError(f"HenriPromptCodec unavailable: {exc}")
    codec = HenriPromptCodec(device=device)
    g = torch.Generator().manual_seed(seed)
    rows = list(_csv.DictReader(open(csv_path, encoding="utf-8")))
    if len(rows) < n:
        raise ValueError(f"csv has {len(rows)} rows < n={n}")
    idx = torch.randperm(len(rows), generator=g)[:n].tolist()
    waves = []
    items = []
    for i in idx:
        row = rows[i]
        q = row.get("question") or row.get("prompt") or row.get("text") or ""
        if not q:
            raise ValueError(f"row {i}: no question/prompt/text field")
        w = codec.encode_prompt(q).to(device)
        waves.append(w)
        items.append({"idx": int(i), "question_len": len(q),
                      "question_sha": hashlib.sha256(q.encode("utf-8")).hexdigest()})
    return torch.stack(waves), items


def _unbinder_greedy_token(wave: torch.Tensor, unbinder) -> Tuple[int, List[int]]:
    flat = wave.reshape(-1)
    with torch.no_grad():
        logits = unbinder(flat)
    top = torch.topk(logits, 5).indices.flatten().tolist()
    return int(logits.argmax().item()), [int(x) for x in top]


def run_audit(csv_path: str, n: int, seed: int, out_dir: str,
              checkpoint: str, device: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    waves, items = _encode_questions(csv_path, n, seed, dev)
    P = grade_projectors()

    # --- transformation arms ---
    per_block_o8 = random_orthogonal(seed=seed * 31 + 7, dim=8,
                                     count=CANONICAL_NUM_BLOCKS)
    shared_o8 = random_orthogonal(seed=seed * 17 + 3, dim=8, count=1)[0]
    # per-block Spin(3) rotors: random unit bivector + angle, seeded
    biv_idx_pool = [4, 5, 6]
    g = torch.Generator().manual_seed(seed * 41 + 11)
    per_block_spin3 = []
    for _ in range(CANONICAL_NUM_BLOCKS):
        b = int(torch.randint(0, 3, (1,), generator=g).item())
        th = float(torch.rand(1, generator=g).item()) * 1.5
        per_block_spin3.append(rotor_sandwich(th, biv_idx_pool[b]))
    per_block_spin3 = torch.stack(per_block_spin3)

    arms = {
        "identity": torch.eye(8, dtype=torch.float64).repeat(1, 1, 1),
        "arbitrary_per_block_O8": per_block_o8,
        "shared_O8": shared_o8.repeat(CANONICAL_NUM_BLOCKS, 1, 1),
        "per_block_Spin3_rotor": per_block_spin3,
    }

    # --- A) grade scramble (per-arm, data-independent) ---
    scramble = {name: float(grade_scramble(T, P))
                for name, T in arms.items()}

    # --- B) invariant collision (row-norm features) ---
    row_norms = waves.norm(dim=-1)  # [n, 8192]
    invariant_feats = row_norms
    # pairwise identical-feature rate
    n_items = waves.shape[0]
    same = 0
    pairs = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            pairs += 1
            if torch.allclose(invariant_feats[i], invariant_feats[j], atol=1e-6):
                same += 1
    invariant_collision_rate = same / pairs if pairs else 0.0

    # --- C) fixed-frame discrimination via legacy unbinder ---
    fixed_frame = {}
    if checkpoint and os.path.exists(checkpoint):
        from henri_decoder import HENRINeuralEgressUnbinder
        unb = HENRINeuralEgressUnbinder(d_model=65536, d_hidden=2048,
                                        vocab_size=32000, device=dev)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        unb.down_proj.weight.data.copy_(ckpt["down_proj.weight"])
        unb.layer_norm.weight.data.copy_(ckpt["layer_norm.weight"])
        unb.layer_norm.bias.data.copy_(ckpt["layer_norm.bias"])
        unb.lm_head.weight.data.copy_(ckpt["lm_head.weight"])
        unb.to(dev)
        unb.eval()
        for name, T in arms.items():
            tokens = []
            for b in range(n_items):
                x = waves[b]
                xT = torch.einsum(
                    "kab,kb->ka", T.to(dev).to(torch.float64),
                    x.to(torch.float64)).to(torch.float32)
                tok, _ = _unbinder_greedy_token(xT, unb)
                tokens.append(tok)
            fixed_frame[name] = {
                "distinct_tokens": len(set(tokens)),
                "change_rate_vs_identity": (
                    sum(1 for a, b_ in zip(fixed_frame.get("identity", {}).get(
                        "tokens", []), tokens) if a != b_) / n_items
                    if "identity" in fixed_frame else None),
                "tokens": tokens,
            }
    else:
        for name in arms:
            fixed_frame[name] = {"distinct_tokens": None,
                                 "change_rate_vs_identity": None,
                                 "tokens": None}

    # --- D) relational stability (joint transform of wave and frame) ---
    rel = {}
    for name, T in arms.items():
        max_err = 0.0
        max_err_x_only = 0.0
        for i in range(n_items):
            x = waves[i]
            for j in range(n_items):
                k = waves[j]
                s = float(torch.cosine_similarity(x.reshape(-1),
                                                  k.reshape(-1), dim=0))
                Tx = torch.einsum("kab,kb->ka", T.to(dev).to(torch.float64),
                                  x.to(torch.float64))
                Tk = torch.einsum("kab,kb->ka", T.to(dev).to(torch.float64),
                                  k.to(torch.float64))
                s_joint = float(torch.cosine_similarity(
                    Tx.reshape(-1), Tk.reshape(-1), dim=0))
                max_err = max(max_err, abs(s - s_joint))
                # gauge test: rotate ONLY x, keep frame fixed
                s_xonly = float(torch.cosine_similarity(
                    Tx.reshape(-1), k.reshape(-1), dim=0))
                max_err_x_only = max(max_err_x_only, abs(s - s_xonly))
        rel[name] = {"joint_max_err": max_err, "x_only_max_err": max_err_x_only}

    # --- verdict ---
    verdict = None
    if scramble["per_block_Spin3_rotor"] > 1e-3 or \
       rel["per_block_Spin3_rotor"]["joint_max_err"] > 1e-2:
        verdict = "ERROR_FAIL_CLOSED"
    elif scramble["per_block_Spin3_rotor"] < 1e-3 and \
            scramble["arbitrary_per_block_O8"] > 0.2 and \
            rel["per_block_Spin3_rotor"]["joint_max_err"] < 1e-4 and \
            invariant_collision_rate > 0.9:
        verdict = "GAUGE_GROUP_VERIFIED"
    elif invariant_collision_rate > 0.9:
        verdict = "FALSIFIED_INVARIANT_INFORMATION_COLLAPSE"
    else:
        verdict = "FALSIFIED_INVALID_GAUGE_GROUP"

    scorecard = {
        "verdict": verdict,
        "grade_scramble": scramble,
        "invariant_collision_rate": invariant_collision_rate,
        "fixed_frame": {k: {kk: vv for kk, vv in v.items()
                            if kk != "tokens"} for k, v in fixed_frame.items()},
        "relational": rel,
        "seed": seed, "n": n, "device": dev,
        "checkpoint_sha256": (hashlib.sha256(
            open(checkpoint, "rb").read()).hexdigest()
            if checkpoint and os.path.exists(checkpoint) else None),
        "arms": list(arms.keys()),
        "prereg": "henri-g0-gauge-audit-prereg-20260825-001 c8b5239700ed6198",
    }
    with open(os.path.join(out_dir, "scorecard.json"), "w") as f:
        json.dump(scorecard, f, indent=2)
    # per-item telemetry (question hashes + per-arm tokens)
    item_rows = []
    for b in range(n_items):
        row = dict(items[b])
        for name in arms:
            row[f"token_{name}"] = (fixed_frame[name].get("tokens") or [None])[b]
        item_rows.append(row)
    with open(os.path.join(out_dir, "items.jsonl"), "w") as f:
        for row in item_rows:
            f.write(json.dumps(row) + "\n")
    return scorecard


def main() -> None:
    ap = argparse.ArgumentParser(description="G0 gauge audit probe")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--out-dir", default="artifacts/g0_gauge_audit")
    ap.add_argument("--checkpoint", default="models/henri_decoder_checkpoint.pt")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not is_enabled():
        print("BLOCKED_DEFAULT_OFF: set HENRI_GAUGE_AUDIT=1 to run")
        return

    sc = run_audit(args.csv, args.n, args.seed, args.out_dir,
                   args.checkpoint, args.device)
    print(json.dumps(sc, indent=2, default=str))


if __name__ == "__main__":
    main()
