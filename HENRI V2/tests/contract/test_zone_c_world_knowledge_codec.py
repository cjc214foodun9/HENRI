"""Contract tests — K5 C2 compositional text codec (zone_c_world_knowledge_codec).

Covers the v4 codec contract (the module the sealed ingest runner imports):
  encode(text) -> (wave_bytes: 262144, proj: [2000] L2-normalized)
  row-unit [8192,8] wave geometry
  locality (end 1-edit), order sensitivity (reversal), random cross-text
  baseline, and offset-free fragment retrieval margin (no position binding).

Gates calibrated on measured distributions (2026-09-04): random cross-text
~0.032; end_1edit ~0.905; reversal ~0.476; fragment_retrieval ~0.32.
CPU-only; dimension-independent by construction.
"""

import numpy as np
import pytest

from zone_c_world_knowledge_codec import (
    NUM_BLOCKS,
    BLOCK_DIM,
    PROJ_DIM,
    CompositionalTextCodec,
    clean_text,
    encode,
    features_of,
    tokenize,
)

SENT = "the quick brown fox jumps over the lazy dog near the riverbank"


def test_encode_contract_shape():
    wave_bytes, proj = encode(SENT)
    assert len(wave_bytes) == NUM_BLOCKS * BLOCK_DIM * 4  # 262144
    assert proj.shape == (PROJ_DIM,)
    assert proj.dtype == np.float32
    rows = np.frombuffer(wave_bytes, dtype=np.float32).reshape(NUM_BLOCKS, BLOCK_DIM)
    assert rows.shape == (NUM_BLOCKS, BLOCK_DIM)
    assert rows.dtype == np.float32


def test_encode_row_unit_geometry():
    wave_bytes, _ = encode(SENT)
    rows = np.frombuffer(wave_bytes, dtype=np.float32).reshape(NUM_BLOCKS, BLOCK_DIM)
    norms = np.linalg.norm(rows, axis=1)
    assert norms.min() >= 0.999, norms.min()
    assert norms.max() <= 1.001, norms.max()
    # no all-zero row: every [8]-row is non-empty
    assert (norms < 1e-6).sum() == 0


def test_encode_proj_unit_norm():
    _, proj = encode(SENT)
    assert abs(float(np.linalg.norm(proj)) - 1.0) < 1e-4


def test_clean_text_deterministic_and_alnum():
    assert clean_text("Hello, World! 123") == "hello world 123"
    assert clean_text("  a   b  ") == "a b"


def test_features_dedup_and_order_sensitive_grams():
    feats = features_of("a b c d e", ngram_max=2)
    assert len(feats) == len(set(feats))  # deduped
    assert any(f.startswith("w:") for f in feats)   # unigrams
    assert any(f.startswith("b:") for f in feats)   # bigrams (order-sensitive)
    feats3 = features_of("a b c d e", ngram_max=3)
    assert any(f.startswith("t:") for f in feats3)  # trigrams
    # order: bigrams of reversed text differ
    rev = features_of("e d c b a", ngram_max=2)
    assert set(feats) != set(rev)


def test_identical_deterministic():
    c = CompositionalTextCodec()
    assert c.geometry()["identical"] >= 0.999


def test_locality_end_1edit():
    c = CompositionalTextCodec()
    g = c.geometry()
    assert g["end_1edit"] >= 0.80, f"end_1edit {g['end_1edit']:.4f}"


def test_reversal_order_sensitivity():
    c = CompositionalTextCodec()
    g = c.geometry()
    assert g["reversal"] <= 0.60, f"reversal {g['reversal']:.4f}"
    assert g["reversal"] < g["end_1edit"], "order change must hurt more than a 1-char edit"


def test_random_cross_baseline():
    c = CompositionalTextCodec()
    g = c.geometry()
    assert g["random_cross"] <= 0.20, f"random_cross {g['random_cross']:.4f}"


def test_fragment_retrieval_margin():
    # offset-free retrieval: a query fragment must match a longer text that
    # contains it far above the random cross-text baseline (no position binding).
    c = CompositionalTextCodec()
    g = c.geometry()
    assert g["fragment_retrieval"] > 2.0 * g["random_cross"], (
        f"fragment {g['fragment_retrieval']:.4f} vs random {g['random_cross']:.4f}"
    )


def test_encode_same_text_same_bytes():
    a, pa = encode(SENT)
    b, pb = encode(SENT)
    assert a == b
    assert np.array_equal(pa, pb)


def test_tokenize_words():
    assert tokenize("The quick, brown fox!") == ["the", "quick", "brown", "fox"]
    assert tokenize("") == []
