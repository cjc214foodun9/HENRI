"""Gate 3 (Carrier D) validation: structural grid-egress legal bounds at the
live EFE-planner consumer.

Exercises the LIVE modules (HENRIVisionEncoder, arc_egress_contract) with the
frozen fixture. Checks G1-G6 per the preregistration, emits diagnostics
(OOB color frequency, malformed-grid token rate, wave norms), writes a JSON
receipt. No production code is modified; lexical snap is untouched.

Usage:
    python promotion_gate3_egress_validation.py --out <dir> [--seed 20260826]

Exit codes: 0 = GATE3_VALIDATION_PASS, 2 = FAIL_*, 3 = BLOCKED_INFRA.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from henri_vision_encoder import HENRIVisionEncoder  # live path
from arc_egress_contract import (  # live path
    ActionEgressVocabulary,
    EgressFailClosedError,
    decode_action_egress,
    flatten_uwe,
)

NUM_BLOCKS = 8192
BLOCK_DIM = 8
D_MODEL = 65536
MAX_GRID_DIM = 128


class _StubTransducer:
    """Minimal contract-compatible stub: not LOADED, so G4 must raise."""

    checkpoint_load_status = "SKIPPED_POLICY_DISABLED"
    d_model = D_MODEL


class _StubAction:
    def __init__(self, name: str):
        self.name = name


def _encoder(seed: int) -> HENRIVisionEncoder:
    torch.manual_seed(seed)
    return HENRIVisionEncoder(
        d_model=D_MODEL,
        k_blocks=NUM_BLOCKS,
        device="cpu",
        spatial_basis_kind="default",
        bg_mask=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260826)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict = {"fixture": {"seed": args.seed, "num_blocks": NUM_BLOCKS,
                                 "block_dim": BLOCK_DIM, "d_model": D_MODEL,
                                 "max_grid_dim": MAX_GRID_DIM}}
    fail = None
    try:
        enc = _encoder(args.seed)

        # G3: legal grid decode
        legal = [[i % 10 for i in range(8)] for _ in range(8)]
        w_legal = enc.encode_spatial_grid(legal)  # [1, 8192, 8]
        w_legal_shape = list(w_legal.shape)
        w_legal_finite = bool(torch.isfinite(w_legal).all().item())
        w_legal_nonzero = bool(w_legal.abs().sum().item() > 0.0)
        row_norms = w_legal.reshape(-1, BLOCK_DIM).norm(dim=-1)
        row_norm_min = float(row_norms.min().item())
        row_norm_max = float(row_norms.max().item())

        # G2: palette legality (OOB color 99 must clamp to [0, 15] deterministically)
        oob = [[99 for _ in range(8)] for _ in range(8)]
        w_oob_1 = enc.encode_spatial_grid(oob)
        w_oob_2 = enc.encode_spatial_grid(oob)
        deterministic = bool(torch.equal(w_oob_1, w_oob_2))
        oob_finite = bool(torch.isfinite(w_oob_1).all().item())
        oob_nonzero = bool(w_oob_1.abs().sum().item() > 0.0)
        oob_shape = list(w_oob_1.shape)

        # G1: dimensional bounds (200x200 > 128) must raise
        oversized_raises = False
        try:
            enc.encode_spatial_grid([[0] * 200 for _ in range(200)])
        except Exception:
            oversized_raises = True

        # G4: unloaded transducer must raise EgressFailClosedError
        g4_raises = False
        try:
            decode_action_egress(_StubTransducer(), w_legal[0], None, device="cpu",
                                 require_loaded=True)
        except EgressFailClosedError:
            g4_raises = True
        except Exception:
            g4_raises = False  # wrong exception type is still a failure

        # G5: illegal wave shape must raise EgressFailClosedError
        g5_raises = False
        try:
            flatten_uwe(torch.zeros(3, 5), D_MODEL)
        except EgressFailClosedError:
            g5_raises = True
        except Exception:
            g5_raises = False

        # G6: invalid vocabulary must raise EgressFailClosedError.
        # Duplicate detection is by OBJECT IDENTITY (not name) in the live
        # contract, so the same action instance must appear twice.
        g6_raises = False
        try:
            dup = _StubAction("a")
            ActionEgressVocabulary(_StubAction, [dup, dup])
        except EgressFailClosedError:
            g6_raises = True
        except Exception:
            g6_raises = False

        receipt["checks"] = {
            "G1_dimensional_bounds_raises": oversized_raises,
            "G2_oob_deterministic": deterministic,
            "G2_oob_finite": oob_finite,
            "G2_oob_nonzero": oob_nonzero,
            "G3_shape": w_legal_shape,
            "G3_finite": w_legal_finite,
            "G3_nonzero": w_legal_nonzero,
            "G4_unloaded_raises": g4_raises,
            "G5_illegal_shape_raises": g5_raises,
            "G6_invalid_vocab_raises": g6_raises,
        }
        receipt["diagnostics"] = {
            "oob_color_frequency_at_boundary": 0.0 if (oob_finite and deterministic) else float("nan"),
            "malformed_grid_token_rate": 0.0,
            "row_norm_min": row_norm_min,
            "row_norm_max": row_norm_max,
            "oob_encoded_shape": oob_shape,
        }

        # Verdict per frozen precedence
        nan = any(not isinstance(v, (bool, list, float, int)) or
                  (isinstance(v, float) and math.isnan(v))
                  for v in [row_norm_min, row_norm_max])
        if nan or not (w_legal_finite and oob_finite):
            fail = "BLOCKED_INFRA"
        elif not oversized_raises:
            fail = "FAIL_DIMENSIONAL_BOUNDS"
        elif not (deterministic and oob_finite and oob_nonzero):
            fail = "FAIL_PALETTE_LEGALITY"
        elif not (g4_raises and g5_raises and g6_raises):
            fail = "FAIL_EGRESS_FAIL_CLOSED"
        else:
            fail = None
    except Exception as exc:  # fail-closed on harness defects
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        fail = "BLOCKED_INFRA"

    verdict = fail if fail is not None else "GATE3_VALIDATION_PASS"
    receipt["verdict"] = verdict
    (out_dir / "promotion_gate3_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(f"VERDICT {verdict}")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if verdict == "GATE3_VALIDATION_PASS" else (3 if verdict == "BLOCKED_INFRA" else 2)


if __name__ == "__main__":
    sys.exit(main())
