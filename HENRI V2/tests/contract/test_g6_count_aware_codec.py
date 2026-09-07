"""Contract tests for G6 v6 count-aware codec (multiplicity + directed order).

Honesty guards:
  * repeated words decode with exact multiplicity ("go go go" -> 3x).
  * order discrimination: "a b c" vs "c b a" decode correctly (bigram chain).
  * OOV input -> ABSTAIN, never text.
  * K5 aliasing pair '013'/'duryee' decode DISTINCTLY.
  * Determinism; zero trainable parameters; payload is NOT row-unit (v6
    contract deviation: count-bearing sparse [8192,8], zero rows allowed).
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HENRI2 = Path(__file__).resolve().parents[2] / "HENRI V2"
sys.path.insert(0, str(HENRI2))

import g6_count_aware_codec as g6v

VOCAB = ["the", "cat", "and", "dog", "mouse", "a", "b", "c", "go",
         "river", "013", "duryee"]


def _codec():
    return g6v.CountAwareCodec(vocab=VOCAB)


def _decode(text: str):
    c = _codec()
    wb, _ = c.encode(text)
    rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
    return c.decode(rows)


def test_multiplicity_exact():
    r = _decode("the cat and the dog and the mouse")
    assert r.status == "OK", r.status
    assert r.text == "the cat and the dog and the mouse", r.text
    # ground truth: "the" occurs 3x (pos 0,3,6), "and" occurs 2x (pos 2,5)
    assert r.counts["the"] == 3 and r.counts["and"] == 2


def test_repeated_streak():
    r = _decode("go go go")
    assert r.status == "OK", r.status
    assert r.text == "go go go", r.text
    assert r.counts["go"] == 3


def test_order_discrimination():
    r1 = _decode("a b c")
    r2 = _decode("c b a")
    assert r1.status == "OK" and r1.text == "a b c", r1.text
    assert r2.status == "OK" and r2.text == "c b a", r2.text


def test_oov_abstains_no_fabrication():
    r = _decode("zebra quantum entanglement")
    assert r.status.startswith("ABSTAIN"), r.status
    assert r.text is None


def test_determinism():
    r1 = _decode("the cat and the dog")
    r2 = _decode("the cat and the dog")
    assert r1.text == r2.text and r1.status == r2.status
    assert r1.counts == r2.counts


def test_no_parameters():
    c = _codec()
    assert not hasattr(c, "parameters")


def test_k5_aliasing_pair_now_distinct():
    for w in ("013", "duryee"):
        r = _decode(w)
        assert r.status == "OK", f"{w}: {r.status}"
        assert r.text == w, f"{w} decoded as {r.text!r}"
    c1 = g6v.g5v_feature_cells if hasattr(g6v, "g5v_feature_cells") else None
    from g5_separable_codec import v5_feature_cells as v5c
    assert not np.array_equal(v5c("w:013"), v5c("w:duryee"))


def test_payload_shape_and_not_row_unit():
    c = _codec()
    wb, _ = c.encode("cat dog")
    rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
    assert rows.shape == (8192, 8)
    norms = np.linalg.norm(rows, axis=1)
    assert float(norms.min()) == 0.0  # sparse: v6 has no fill
    assert float(norms.max()) > 0.0


def test_no_fabrication_no_extra_words():
    r = _decode("cat dog")
    assert r.status == "OK"
    assert Counter(r.text.split()) == Counter(["cat", "dog"])


def test_count_estimate_two_repeats():
    r = _decode("river river")
    assert r.status == "OK", r.status
    assert r.text == "river river", r.text
    assert r.counts["river"] == 2
