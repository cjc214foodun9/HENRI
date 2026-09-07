"""Contract tests for G5 v5 separable codec (exact support-membership decode).

Honesty guards:
  * '013' / 'duryee' (K5 aliasing pair: same h mod 8192) decode DISTINCTLY.
  * OOV input (words not in vocab) -> ABSTAIN, never text.
  * Determinism; zero trainable parameters.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HENRI2 = Path(__file__).resolve().parents[2] / "HENRI V2"
sys.path.insert(0, str(HENRI2))

import g5_separable_codec as g5v

VOCAB = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
         "near", "river", "bank", "013", "duryee"]


def _codec():
    return g5v.SeparableCodec(vocab=VOCAB)


def test_k5_aliasing_pair_now_distinct():
    c = _codec()
    for w in ("013", "duryee"):
        wb, _ = c.encode(w)
        rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
        r = c.decode(rows)
        assert r.status == "OK", f"{w}: {r.status}"
        assert r.text == w, f"{w} decoded as {r.text!r}"
    # distinct signatures
    c1 = g5v.v5_feature_cells("w:013")
    c2 = g5v.v5_feature_cells("w:duryee")
    assert not np.array_equal(c1, c2), "v5 cells must be full-entropy distinct"


def test_in_vocab_exact_recovery():
    c = _codec()
    text = "the quick brown fox"
    wb, _ = c.encode(text)
    rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
    r = c.decode(rows)
    assert r.status == "OK"
    assert r.text == text, f"got {r.text!r}"


def test_oov_abstains_no_fabrication():
    c = _codec()
    # 'zebra', 'quantum', 'entanglement' NOT in VOCAB
    wb, _ = c.encode("zebra quantum entanglement")
    rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
    r = c.decode(rows)
    assert r.status.startswith("ABSTAIN"), f"expected abstain, got {r.status}"
    assert r.text is None


def test_determinism():
    c = _codec()
    wb, _ = c.encode("lazy dog near river")
    rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
    assert c.decode(rows).text == c.decode(rows).text


def test_no_parameters():
    c = _codec()
    assert not hasattr(c, "parameters")


def test_row_unit_payload():
    c = _codec()
    wb, _ = c.encode("the quick brown fox jumps over the lazy dog")
    rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
    norms = np.linalg.norm(rows, axis=1)
    assert float(norms.min()) >= 0.999
    assert float(norms.max()) <= 1.001
    assert (np.any(np.abs(rows) > 1e-9, axis=1)).sum() == 8192  # no all-zero rows
