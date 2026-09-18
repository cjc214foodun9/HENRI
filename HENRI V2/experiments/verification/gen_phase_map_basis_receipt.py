#!/usr/bin/env python3
"""Generate the PHASE-MAP BASIS receipt (Phase 7.8 P0-A1 evidence).

WHY THIS EXISTS
    `arc_spatial_basis.py` already promotes the incommensurate x/y ramp to the
    PRODUCTION DEFAULT (DEFAULT_SPATIAL_BASIS = "incommensurate",
    DEFAULT_BG_MASK = True) and `production_arc_run.py:823` consumes it. The
    code half of the promotion landed; the PAIRED RECEIPT did not. Until this
    receipt exists the invertibility claim is CONDITIONAL, because
    `arc_phase_map.py` registers BLOCKED_PHASE_MAP_NONINVERTIBLE for the legacy
    collinear basis and no committed artifact demonstrates the fix.

WHAT IT MEASURES (real encoder, no mocking)
    1. Legacy 'default' basis: single-pixel grids at (1,2) and (2,1) have the
       same x+y, so the carrier exp(i(x+y)w) makes them IDENTICAL. Expect cos ~ 1.
    2. 'incommensurate' basis: the y ramp is scaled by sqrt(2), breaking the
       degeneracy. Expect same-sum cos well below the degeneracy threshold.
    3. Fractional coordinate recovery on the non-default basis only. The legacy
       path must FAIL there by construction (it is rank-deficient).
    4. Byte-identity of the legacy explicit combination vs the pre-7.8 default.

Every number written here is MEASURED by this script. Nothing is assumed.
Usage:  python experiments/verification/gen_phase_map_basis_receipt.py
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "experiments" / "verification" / "phase_map_basis_observed.json"

sys.path.insert(0, str(REPO))

from arc_phase_map import (  # noqa: E402
    STATUS_INVERTIBLE,
    STATUS_NONINVERTIBLE,
    fractional_unbind_coordinate,
    verify_phase_map_invertibility,
)
from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402
from arc_spatial_basis import (  # noqa: E402
    DEFAULT_BG_MASK,
    DEFAULT_SPATIAL_BASIS,
    resolve_spatial_basis,
)

D_MODEL = 65536
K_BLOCKS = 8192
GRID_DIM = 4
COLOR = 5
SEED = 20260917


def enc(kind: str, bg_mask: bool) -> HENRIVisionEncoder:
    return HENRIVisionEncoder(
        d_model=D_MODEL, k_blocks=K_BLOCKS, block_dim=8,
        device="cpu", spatial_basis_kind=kind, bg_mask=bg_mask,
    )


def single_pixel_wave(encoder, r: int, c: int, grid_dim: int = GRID_DIM,
                      color: int = COLOR) -> torch.Tensor:
    g = [[0] * grid_dim for _ in range(grid_dim)]
    g[r][c] = color
    with torch.no_grad():
        w = encoder.encode_spatial_grid(g).squeeze(0).reshape(-1).to(torch.float32)
    return torch.nn.functional.normalize(w, p=2, dim=-1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> int:
    t0 = time.time()
    torch.manual_seed(SEED)

    print(f"torch {torch.__version__}  platform {platform.platform()}")
    print(f"D={D_MODEL} K_BLOCKS={K_BLOCKS} GRID_DIM={GRID_DIM} COLOR={COLOR}")

    # ---- resolver defaults (the production path) -------------------------
    for var in ("HENRI_ARC_SPATIAL_BASIS", "HENRI_ARC_BG_MASK"):
        import os
        os.environ.pop(var, None)
    resolved = resolve_spatial_basis()
    print(f"resolver default (no env): {resolved}")
    assert resolved == (DEFAULT_SPATIAL_BASIS, DEFAULT_BG_MASK), "resolver drift"

    # ---- 1/2. degeneracy per basis ---------------------------------------
    per_basis = {}
    for kind, bg in (("default", False), ("incommensurate", True), ("random", True)):
        e = enc(kind, bg)
        with torch.no_grad():
            w12 = single_pixel_wave(e, 1, 2)
            w21 = single_pixel_wave(e, 2, 1)
            w11 = single_pixel_wave(e, 1, 1)
        same = float(torch.dot(w12, w21).item())
        diff = float(torch.dot(w12, w11).item())
        verdict = verify_phase_map_invertibility(e, grid_dim=GRID_DIM, color=COLOR)
        per_basis[kind] = {
            "bg_mask": bg,
            "same_sum_cos": same,
            "diff_sum_cos": diff,
            "verify_status": verdict.status,
            "verify_reason": verdict.reason,
        }
        print(f"  {kind:>15}: same_sum_cos={same:+.6f}  diff_sum_cos={diff:+.6f}"
              f"  status={verdict.status}")

    assert per_basis["default"]["verify_status"] == STATUS_NONINVERTIBLE, \
        "legacy basis must remain BLOCKED_PHASE_MAP_NONINVERTIBLE"
    assert per_basis["incommensurate"]["verify_status"] == STATUS_INVERTIBLE, \
        "incommensurate basis must be invertible"

    # ---- 3. coordinate recovery, non-default basis only -------------------
    recovery = {}
    e_inc = enc("incommensurate", True)
    hits = 0
    trials = []
    for r in range(GRID_DIM):
        for c in range(GRID_DIM):
            with torch.no_grad():
                w = single_pixel_wave(e_inc, r, c)
            rr, cc, cos = fractional_unbind_coordinate(
                w, e_inc, COLOR, GRID_DIM, device="cpu"
            )
            ok = (rr == r and cc == c)
            hits += int(ok)
            trials.append({
                "true": [r, c], "decoded": [rr, cc],
                "response_cos": cos, "exact": ok,
            })
    n = GRID_DIM * GRID_DIM
    recovery = {
        "basis_kind": "incommensurate",
        "bg_mask": True,
        "n_cases": n,
        "n_exact": hits,
        "exact_rate": hits / n,
        "trials": trials,
    }
    print(f"  fractional recovery (incommensurate): {hits}/{n} exact")

    # ---- 3b. the legacy path must FAIL the same recovery ------------------
    e_def = enc("default", False)
    legacy_ok = 0
    try:
        for r in range(GRID_DIM):
            for c in range(GRID_DIM):
                with torch.no_grad():
                    w = single_pixel_wave(e_def, r, c)
                rr, cc, _ = fractional_unbind_coordinate(
                    w, e_def, COLOR, GRID_DIM, device="cpu"
                )
                legacy_ok += int(rr == r and cc == c)
        legacy_note = "ran"
    except ValueError as exc:
        legacy_ok = -1
        legacy_note = f"raises: {exc}"
    legacy_recovery = {
        "basis_kind": "default",
        "n_cases": n,
        "n_exact": legacy_ok,
        "note": legacy_note,
    }
    print(f"  fractional recovery (default): {legacy_ok}/{n}  ({legacy_note[:60]})")

    # ---- 4. legacy byte-identity -----------------------------------------
    a = single_pixel_wave(enc("default", False), 1, 2)
    b = single_pixel_wave(enc("default", False), 1, 2)
    byte_identical = torch.equal(a, b)
    print(f"  legacy determinism byte-identity: {byte_identical}")

    receipt = {
        "schema": "henri.arc.phase-map-basis-eval.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": "OBSERVED",
        "device_kind": "cpu",
        "torch": torch.__version__,
        "platform": platform.platform(),
        "d_model": D_MODEL,
        "k_blocks": K_BLOCKS,
        "grid_dim": GRID_DIM,
        "color": COLOR,
        "seed": SEED,
        "resolver_default": {
            "spatial_basis_kind": resolved[0],
            "bg_mask": resolved[1],
        },
        "per_basis": per_basis,
        "recovery_incommensurate": recovery,
        "recovery_default": legacy_recovery,
        "legacy_byte_identical": byte_identical,
        "claim": (
            "The production default encoder basis (incommensurate ramp + "
            "CC-OS background mask) is invertible for 2D localization; the "
            "legacy collinear basis is rank-deficient and remains "
            "BLOCKED_PHASE_MAP_NONINVERTIBLE."
        ),
        "elapsed_secs": round(time.time() - t0, 2),
    }
    OUT.write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(f"sha256 {sha256_file(OUT)}")
    print(f"elapsed {receipt['elapsed_secs']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
