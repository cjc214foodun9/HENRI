"""Gate 2 (Carrier C) validation: held-out checks of the LIVE Procrustes goal adapter.

Imports the existing HenriTaskOperator / HenriGoalAdapter from henri_goal_adapter.py
(no reimplementation). Runs three pre-registered controls (C1 known transform,
C2 held-out reconstruction, C3 shuffled-pair negative), emits diagnostics
(conditioning, orthogonality, shapes), and writes a JSON receipt.

Usage:
    python promotion_gate2_validation.py --out <dir> [--seed 20260826]

Exit codes: 0 = GATE2_VALIDATION_PASS, 2 = FAIL_*, 3 = BLOCKED_INFRA.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from henri_goal_adapter import HenriGoalAdapter, HenriTaskOperator  # live path

NUM_BLOCKS = 8192
BLOCK_DIM = 8
M_TOTAL = 14
M_CAL = 10
M_HELD = 4


def _unit_rows(t: torch.Tensor) -> torch.Tensor:
    return t / (t.norm(dim=-1, keepdim=True) + 1e-12)


def _known_orthogonal(seed: int, num_blocks: int, dim: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(num_blocks, dim, dim, generator=g)
    q, r = torch.linalg.qr(z)
    # Fix sign ambiguity so the transform is recoverable up to exact O_k.
    s = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    return q * s.unsqueeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict = {"fixture": {"seed": args.seed, "num_blocks": NUM_BLOCKS,
                                 "block_dim": BLOCK_DIM, "m_total": M_TOTAL,
                                 "m_cal": M_CAL, "m_held": M_HELD}}
    fail = None
    try:
        torch.manual_seed(args.seed)
        # Fixture
        x_all = _unit_rows(torch.randn(M_TOTAL, NUM_BLOCKS, BLOCK_DIM))
        o_known = _known_orthogonal(args.seed, NUM_BLOCKS, BLOCK_DIM)
        y_known = _unit_rows(torch.einsum("mka,kab->mkb", x_all, o_known))

        op = HenriTaskOperator()

        # C1: known-transform positive (fit on all demos, reconstruct held-out)
        w1 = op.compile_from_demos(x_all, y_known)
        rec1 = torch.einsum("kab,kb->ka", w1, x_all[M_CAL:])
        c1_cos = float(torch.cosine_similarity(
            rec1.reshape(M_HELD, -1), y_known[M_CAL:].reshape(M_HELD, -1), dim=1).mean())
        orth_err = op.orthogonality_error(w1)

        # C2: held-out reconstruction (calibration-only fit)
        w2 = op.compile_from_demos(x_all[:M_CAL], y_known[:M_CAL])
        rec2 = torch.einsum("kab,kb->ka", w2, x_all[M_CAL:])
        c2_cos = float(torch.cosine_similarity(
            rec2.reshape(M_HELD, -1), y_known[M_CAL:].reshape(M_HELD, -1), dim=1).mean())

        # C3: shuffled-pair negative (derangement on calibration X side)
        pi = [(i + 1) % M_CAL for i in range(M_CAL)]  # rotation by 1, no fixed points
        x_shuf = x_all[:M_CAL][pi]
        w3 = op.compile_from_demos(x_shuf, y_known[:M_CAL])
        rec3 = torch.einsum("kab,kb->ka", w3, x_all[M_CAL:])
        c3_cos = float(torch.cosine_similarity(
            rec3.reshape(M_HELD, -1), y_known[M_CAL:].reshape(M_HELD, -1), dim=1).mean())

        # Diagnostics: conditioning of M_k for C2 fit
        m = torch.einsum("mka,mkb->kab", x_all[:M_CAL], y_known[:M_CAL])
        s = torch.linalg.svdvals(m)  # [K, 8]
        cond = (s[:, 0] / (s[:, -1] + 1e-12)).detach().cpu().numpy()
        s_p5 = float(s[:, -1].quantile(0.05).item())
        cond_median = float(float(torch.tensor(cond).median()))
        cond_p95 = float(torch.tensor(cond).quantile(0.95).item())

        receipt["controls"] = {
            "C1_known_transform_cos": c1_cos,
            "C2_heldout_recon_cos": c2_cos,
            "C3_shuffled_recon_cos": c3_cos,
            "orthogonality_err": orth_err,
        }
        receipt["diagnostics"] = {
            "cond_median": cond_median, "cond_p95": cond_p95,
            "min_singular_p5": s_p5,
            "operator_shape": list(w2.shape), "wave_shape": list(x_all.shape[1:]),
        }

        # Verdict per frozen precedence (margins from contract)
        nan = any(math.isnan(v) for v in (c1_cos, c2_cos, c3_cos, orth_err))
        if nan:
            fail = "BLOCKED_INFRA"
        elif list(w2.shape) != [NUM_BLOCKS, BLOCK_DIM, BLOCK_DIM] or list(x_all.shape[1:]) != [NUM_BLOCKS, BLOCK_DIM]:
            fail = "FAIL_SHAPE"
        elif orth_err > 1e-4:
            fail = "FAIL_ORTHOGONALITY"
        elif c1_cos < 0.95:
            fail = "FAIL_KNOWN_TRANSFORM"
        elif c3_cos > 0.60:
            fail = "FAIL_SHUFFLE_CONTROL"
        elif c2_cos < 0.95:
            fail = "FAIL_RECONSTRUCTION"
        else:
            fail = None
    except Exception as exc:  # fail-closed on harness defects
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        fail = "BLOCKED_INFRA"

    verdict = fail if fail is not None else "GATE2_VALIDATION_PASS"
    receipt["verdict"] = verdict
    receipt["margins"] = {"recon_cos_min": 0.95, "known_cos_min": 0.95,
                          "shuffled_cos_max": 0.60, "orthogonality_err_max": 1e-4}
    (out_dir / "promotion_gate2_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(f"VERDICT {verdict}")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if verdict == "GATE2_VALIDATION_PASS" else (3 if verdict == "BLOCKED_INFRA" else 2)


if __name__ == "__main__":
    sys.exit(main())
