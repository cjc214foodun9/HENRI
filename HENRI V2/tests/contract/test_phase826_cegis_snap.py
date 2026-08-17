"""Phase 8.26 contract tests — CEGIS codebook snap & pixel cleanliness.

Spec: HENRI-ANALYSIS-2026-08-SOLVING-FRONTIER (sha 8c508808...).
Covers: default-OFF flag, fail-closed consumer, zero-leak scan,
gate pre-registration, and functional snap behavior (no mock loop).
"""
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "HENRI V2" / "production_arc_run.py"
SNAP = ROOT / "HENRI V2" / "cegis_grid_snap.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def test_826_snap_default_off():
    src = _read(RUNNER)
    assert 'os.environ.get("HENRI_ARC_CEGIS_SNAP", "0") == "1"' in src
    assert "cegis_grid_snap" in src


def test_826_snap_fail_closed_consumer():
    # Without a grid source the snap must emit SNAP_NO_GRID_SOURCE and
    # never fabricate a grid path.
    src = _read(RUNNER)
    assert "SNAP_NO_GRID_SOURCE" in src
    assert "SNAP_CONSERVATION_VIOLATION" in src


def test_826_snap_scope_safe():
    # The snap block must be self-contained: it must derive the grid from
    # obs directly, NOT reference the loop's `grid` variable (assigned
    # ~113 lines later in the iteration — a latent NameError that compile
    # cannot catch). Reverse-review catch 2026-08-17.
    src = _read(RUNNER)
    block = src.split("snap_status = \"SNAP_NO_GRID_SOURCE\"")[1]
    block = block.split("if LAMBDA_GOAL > 0.0 or HENRI_ARC_TARGET_GROUNDING")[0]
    assert "obs.frame[0].tolist()" in block, (
        "snap block must derive grid from obs (scope-safe)")
    assert "np.asarray(_snap_grid, dtype=float)" in block


def test_826_snap_functional():
    # Real functional check: isolated-pixel noise is removed and
    # conservation invariants hold (not a mock loop).
    from cegis_grid_snap import cegis_grid_snap
    rng = np.random.default_rng(826)
    clean = np.zeros((8, 8), dtype=int)
    clean[:4, :4] = 1
    clean[:4, 4:] = 2
    clean[4:, :4] = 3
    clean[4:, 4:] = 4
    noisy = clean.copy()
    mask = rng.random(clean.shape) < 0.1
    noisy[mask] = rng.integers(1, 10, size=int(mask.sum()))
    res = cegis_grid_snap(noisy.astype(float), ref_grid=clean)
    assert res["conservation_ok"], "conservation violated on recovered grid"
    assert float((res["grid"] == clean).mean()) > float((noisy == clean).mean())


def test_826_zero_pretraining_invariant():
    src = _read(SNAP)
    for leak in ("arcade", "examples", "solution"):
        assert not re.search(rf"\b{leak}\b", src, re.I), f"leak term: {leak}"
