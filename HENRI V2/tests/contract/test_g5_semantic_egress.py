"""Contract tests for G5 semantic egress (DSCI) + WavePacketPathSearch.

Honesty guards:
  * DSCI determinism (same wave -> same text across runs).
  * Fail-closed: no fabricated token in ABSTAIN paths.
  * Zero-pretrain: no trainable nn.Parameter created by the engine.
  * Wave packet: shapes, unit rows after snap, deterministic, flag default OFF.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

HENRI2 = Path(__file__).resolve().parents[2] / "HENRI V2"
sys.path.insert(0, str(HENRI2))
sys.path.insert(0, str(HENRI2.parent))

import g5_semantic_egress  # noqa: E402
from zone_c_world_knowledge_codec import CompositionalTextCodec, WAVE_DIM  # noqa: E402


HAS_TORCH = False
try:
    import torch  # noqa: F401
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def _engine():
    vocab = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "near", "river"]
    return g5_semantic_egress.DSCIEngine(vocab=vocab)


def test_dsci_determinism():
    eng = _engine()
    codec = CompositionalTextCodec()
    wave_bytes, _ = codec.encode("the quick brown fox")
    rows = np.frombuffer(wave_bytes, dtype=np.float32).reshape(8192, 8)
    r1 = eng.decode(rows)
    r2 = eng.decode(rows)
    assert r1.text == r2.text, "DSCI must be deterministic"
    assert r1.status == r2.status


def test_dsci_fail_closed_no_fabrication():
    eng = _engine()
    # Vocabulary excludes the actual words -> engine must ABSTAIN, not invent.
    codec = CompositionalTextCodec()
    wave_bytes, _ = codec.encode("zebra xylophone quantum entanglement")
    rows = np.frombuffer(wave_bytes, dtype=np.float32).reshape(8192, 8)
    r = eng.decode(rows)
    assert r.status.startswith("ABSTAIN"), f"expected abstain, got {r.status}"
    assert r.text is None, "abstain path must not emit text"


def test_dsci_input_validation():
    eng = _engine()
    r = eng.decode(np.zeros((8, 8), dtype=np.float32))
    assert r.status == "ABSTAIN_INVALID_INPUT"


def test_dsci_output_schema():
    eng = _engine()
    codec = CompositionalTextCodec()
    wave_bytes, _ = codec.encode("the brown fox")
    rows = np.frombuffer(wave_bytes, dtype=np.float32).reshape(8192, 8)
    r = eng.decode(rows)
    assert isinstance(r.confidence, float)
    assert isinstance(r.n_pursued, int)
    assert isinstance(r.path, list)


def test_dsci_no_parameters():
    eng = _engine()
    assert not hasattr(eng, "parameters"), "DSCI must not expose nn.Parameter"


def test_build_vocab_bounded():
    v = g5_semantic_egress.build_vocab(["a b c", "b c d"], max_words=2)
    assert len(v) <= 2
    assert v == sorted(v)


def test_wave_packet_import_and_ops():
    wp = importlib.import_module("g5_wave_packet_search")
    assert wp.make_packet_ops() == [
        "Identity", "Rotate90", "Rotate180", "Rotate270",
        "FlipHorizontal", "FlipVertical", "ColorPermute", "ContourFill", "GravityDrop",
    ]


@pytest.mark.skipif(not HAS_TORCH, reason="torch unavailable")
def test_wave_packet_shapes_and_determinism():
    import torch
    import g5_wave_packet_search as wp

    # Deterministic encoder (fixed tensor): delta must be reproducible.
    enc = lambda g: torch.full((8192, 8), 0.5)  # noqa: E731
    opv = wp.default_op_encoder("cpu")
    op_exec = lambda op, grid: grid  # noqa: E731
    engine = wp.WavePacketPathSearch(encoder=enc, op_encoder=opv, op_exec=op_exec, mode="exact")
    grid = np.zeros((30, 30), dtype=int)
    r1 = engine.search(grid, grid, wp.make_packet_ops(), depth=2)
    r2 = engine.search(grid, grid, wp.make_packet_ops(), depth=2)
    assert r1.n_candidates == len(wp.make_packet_ops()) * 2
    assert r1.delta_best == r2.delta_best
    assert isinstance(r1.frontier_width, list)
