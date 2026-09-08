"""Contract tests for G7 v7 high-order count-aware codec.

Honesty guards (mirror v6 conventions + v7-specific):
  * repeated words decode with exact multiplicity ("go go go" -> 3x).
  * order discrimination: "a b c" vs "c b a" decode correctly.
  * high-order exact walk: "a b a c" resolves to the unique true walk under
    bigram+trigram constraints (1st-order edges alone also permit it; the
    enumeration must return the TRUE sequence, not reject it).
  * OOV input -> ABSTAIN, never text.
  * K5 aliasing pair '013'/'duryee' decode DISTINCTLY.
  * determinism; zero trainable parameters; payload [8192,8] count-bearing.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

HENRI2 = Path(__file__).resolve().parents[2] / "HENRI V2"
sys.path.insert(0, str(HENRI2))

import g7_highorder_codec as g7v
from zone_c_world_knowledge_codec import tokenize

VOCAB = ["the", "cat", "and", "dog", "mouse", "a", "b", "c", "go",
         "river", "013", "duryee"]


def _codec():
    return g7v.HighOrderCodec(vocab=VOCAB)


def _decode(text: str):
    c = _codec()
    wb, _ = c.encode(text)
    rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
    return c.decode(rows)


def test_multiplicity_exact():
    r = _decode("the cat and the dog and the mouse")
    assert r.status == "OK", r.status
    assert r.text == "the cat and the dog and the mouse", r.text
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


def test_highorder_walk_repeats_true_sequence():
    # true walk a-b-a-c: bigrams a b, b a, a c all written; counts a=2 b=1 c=1.
    r = _decode("a b a c")
    assert r.status == "OK", r.status
    assert r.text == "a b a c", r.text
    assert r.n_walks == 1 and not r.cap_hit


def test_oov_abstains_no_fabrication():
    r = _decode("zebra quantum entanglement")
    assert r.status.startswith("ABSTAIN"), r.status
    assert r.text is None


def test_determinism():
    r1 = _decode("the cat and the dog")
    r2 = _decode("the cat and the dog")
    assert r1.text == r2.text and r1.status == r2.status
    assert r1.counts == r2.counts and r1.n_walks == r2.n_walks


def test_no_parameters():
    c = _codec()
    assert not hasattr(c, "parameters")


def test_k5_aliasing_pair_now_distinct():
    for w in ("013", "duryee"):
        r = _decode(w)
        assert r.status == "OK" and r.text == w, (w, r.status, r.text)


def test_payload_shape_not_row_unit():
    c = _codec()
    wb, _ = c.encode("a b c")
    assert len(wb) == 8192 * 8 * 4
    rows = np.frombuffer(wb, dtype=np.float32).reshape(8192, 8)
    assert rows.shape == (8192, 8)
    # count-bearing sparse payload: some rows may be all-zero (deviation from
    # v5 row-unit contract, documented in v6/v7)
    assert not np.allclose(np.linalg.norm(rows, axis=1), 1.0)


def test_no_fabrication_no_extra_words():
    r = _decode("the dog and the cat")
    if r.status == "OK":
        pred = Counter(tokenize(r.text))
        assert set(pred) <= {"the", "dog", "and", "cat"}
        assert pred["the"] == 2
