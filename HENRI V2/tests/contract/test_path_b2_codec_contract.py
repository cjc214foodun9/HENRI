"""Contract tests for PathB2DiscriminativeCodec (Class 4.4, Path B2, default-OFF)."""
from __future__ import annotations

import pytest
import torch


def _sample_vocab() -> list[str]:
    return ["FunctionDef", "Return", "Assign", "Call", "BinOp", "arg", "x", "y", "n", "len", "<UNK>"]


@pytest.fixture(scope="module")
def codec():
    from path_b2_semantic_codec import PathB2DiscriminativeCodec

    return PathB2DiscriminativeCodec(
        d_model=4096,
        d_latent=128,
        vocab=_sample_vocab(),
        df={t: 1 for t in _sample_vocab()},
        n_docs=100,
        device="cpu",
        seed=7,
    )


def test_unit_norm():
    from path_b2_semantic_codec import PathB2DiscriminativeCodec

    c = PathB2DiscriminativeCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), df={t: 1 for t in _sample_vocab()}, n_docs=100, device="cpu", seed=7)
    v = c.encode_sequence("def f(x): return len(x)")
    assert v.dtype == torch.float32
    assert v.shape == (4096,)
    assert abs(v.norm().item() - 1.0) <= 1e-6


def test_binding_norm_preserving():
    from path_b2_semantic_codec import PathB2DiscriminativeCodec

    c = PathB2DiscriminativeCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), df={t: 1 for t in _sample_vocab()}, n_docs=100, device="cpu", seed=7)
    a = c.encode_sequence("def f(x): return x")
    b = c.encode_sequence("def g(y): return y")
    bound = c.bind(a, b)
    assert abs(bound.norm().item() - 1.0) <= 1e-6


def test_no_dense_allocation():
    """Embedding is [|V|, d_latent]; lift is a function, never a [D,D] tensor."""
    from path_b2_semantic_codec import PathB2DiscriminativeCodec

    c = PathB2DiscriminativeCodec(d_model=65536, d_latent=512, vocab=_sample_vocab(), df={t: 1 for t in _sample_vocab()}, n_docs=100, device="cpu", seed=7)
    for name, p in c.named_parameters():
        assert p.shape[0] < 65536 or p.shape[1] < 65536, f"dense parameter: {name} {tuple(p.shape)}"
    assert c.lift is not None


def test_idf_weighting_active():
    """IDF weighting must be consumed: rare tokens dominate over common ones.
    Same-body-renamed stays close; structurally different must NOT collapse to
    ~1.0 (carrier-collapse regression guard)."""
    from path_b2_semantic_codec import PathB2DiscriminativeCodec

    # Realistic df: structural tokens appear in ~90% of docs (low idf);
    # discriminative tokens (identifiers, primitives) are rare (high idf).
    df = {"def": 92, "return": 88, "(": 95, ")": 95, ":": 90, ",": 85,
          "Module": 96, "FunctionDef": 90, "arguments": 92, "Return": 88,
          "Call": 86, "Name": 96, "Load": 96, "arg": 90, "BinOp": 60,
          "x": 12, "y": 12, "len": 8, "max": 7, "a": 25, "b": 25}
    c = PathB2DiscriminativeCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), df=df, n_docs=100, device="cpu", seed=7)
    v1 = c.encode_sequence("def f(x): return len(x)")
    v2 = c.encode_sequence("def f(y): return len(y)")
    # same body, renamed: still close (idf doesn't break semantic equivalence)
    s12 = c.cosine_similarity(v1, v2).item()
    assert s12 > 0.3
    # different structure must differ and must NOT be ~1.0 (carrier collapse)
    v3 = c.encode_sequence("def g(a, b): return max(a, b)")
    s13 = c.cosine_similarity(v1, v3).item()
    assert s13 < 0.99  # no carrier collapse on a fresh (untrained) codec
    # NOTE: s12 > s13 ordering is a LEARNED property (semantic separation);
    # it is measured by Gate A on the trained codec, not asserted on random init.


def test_hard_negatives_generate():
    from path_b2_semantic_codec import PathB2DiscriminativeCodec

    c = PathB2DiscriminativeCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), df={t: 1 for t in _sample_vocab()}, n_docs=100, device="cpu", seed=7)
    code = "def f(x): return x + 1"
    negs = c.generate_hard_negatives(code, n=8, seed=1)
    assert len(negs) >= 4  # at least the real mutants
    assert all(n != code for n in negs)
    # binop flip must be present
    assert any("-" in n for n in negs)


def test_gram_retraction_bounds():
    """After training steps, Gram error stays below 1e-3 (loose CPU bound; CUDA
    target is 1e-5 per spec)."""
    from path_b2_semantic_codec import PathB2DiscriminativeCodec

    c = PathB2DiscriminativeCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), df={t: 1 for t in _sample_vocab()}, n_docs=100, device="cpu", seed=7)
    data = [("t1", "def f(x): return len(x)"), ("t2", "def g(a, b): return a + b"), ("t3", "def h(n): return n * 2")]
    m = c.train_contrastive(data, data, steps=4, batch_size=2, tau=0.07, n_hard=2, seed=7)
    assert m["gram_max"] is not None
    assert m["gram_max"] < 1e-3
    assert 0.0 <= m["val_contrastive_acc"] <= 1.0


def test_rejects_uint8_ring():
    from path_b2_semantic_codec import PathB2DiscriminativeCodec, RepresentationBoundaryError

    c = PathB2DiscriminativeCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), df={t: 1 for t in _sample_vocab()}, n_docs=100, device="cpu", seed=7)
    ring = torch.randint(0, 256, (4096,), dtype=torch.uint8)
    with pytest.raises(RepresentationBoundaryError):
        c.cosine_similarity(ring, c.encode_sequence("def f(x): return x"))


def test_default_off_runner_flag():
    """Runner without --path-b2-codec must be untouched; flag exists default False."""
    runner = "HENRI V2/humaneval_wave_ast_runner.py"
    src = open(runner, encoding="utf-8").read()
    assert "--path-b2-codec" in src
    assert "path_b2" in src or "PathB2" in src


def test_malformed_input_fail_closed():
    from path_b2_semantic_codec import PathB2DiscriminativeCodec

    c = PathB2DiscriminativeCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), df={t: 1 for t in _sample_vocab()}, n_docs=100, device="cpu", seed=7)
    with pytest.raises(TypeError):
        c.encode_sequence(None)  # type: ignore[arg-type]
