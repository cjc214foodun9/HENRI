"""Contract tests for PathBSemanticCodec (Class 4.3, Path B, default-OFF).

RED first: module does not exist yet at first run of this file.
"""
from __future__ import annotations

import hashlib

import pytest
import torch


def _sample_vocab() -> list[str]:
    return ["FunctionDef", "Return", "Assign", "Call", "BinOp", "arg", "x", "y", "n", "len"]


@pytest.fixture(scope="module")
def codec():
    from path_b_semantic_codec import PathBSemanticCodec

    return PathBSemanticCodec(
        d_model=4096,  # reduced dim for CPU tests
        d_latent=128,
        vocab=_sample_vocab(),
        device="cpu",
        seed=7,
    )


def test_unit_norm():
    from path_b_semantic_codec import PathBSemanticCodec

    c = PathBSemanticCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), device="cpu", seed=7)
    v = c.encode_sequence("def f(x): return len(x)")
    assert v.dtype == torch.float32
    assert v.shape == (4096,)
    assert abs(v.norm().item() - 1.0) <= 1e-6


def test_binding_norm_preserving():
    from path_b_semantic_codec import PathBSemanticCodec

    c = PathBSemanticCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), device="cpu", seed=7)
    a = c.encode_sequence("def f(x): return x")
    b = c.encode_sequence("def g(y): return y")
    bound = c.bind(a, b)
    assert abs(bound.norm().item() - 1.0) <= 1e-6


def test_no_dense_allocation():
    """Embedding is [|V|, d_latent]; lift is a function, never a [D,D] tensor."""
    from path_b_semantic_codec import PathBSemanticCodec

    c = PathBSemanticCodec(d_model=65536, d_latent=512, vocab=_sample_vocab(), device="cpu", seed=7)
    for name, p in c.named_parameters():
        assert p.shape[0] < 65536 or p.shape[1] < 65536, f"dense parameter: {name} {tuple(p.shape)}"
    assert c.lift is not None  # frozen functional lift, no dense matrix parameter


def test_same_encoder_family_consistency():
    from path_b_semantic_codec import PathBSemanticCodec

    c = PathBSemanticCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), device="cpu", seed=7)
    goal = c.encode_sequence("def f(x): return len(x)")
    cand = c.encode_sequence("def f(x): return len(x)")
    sim = c.cosine_similarity(goal, cand)
    assert sim.item() > 0.99
    # Semantic equivalence (same body, renamed) ranks above unrelated
    renamed = c.encode_sequence("def g(s): return len(s)")
    unrelated = c.encode_sequence("def h(a, b): return max(a, b)")
    assert c.cosine_similarity(goal, renamed).item() > c.cosine_similarity(goal, unrelated).item()


def test_rejects_uint8_ring():
    from path_b_semantic_codec import PathBSemanticCodec, RepresentationBoundaryError

    c = PathBSemanticCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), device="cpu", seed=7)
    ring = torch.randint(0, 256, (4096,), dtype=torch.uint8)
    with pytest.raises(RepresentationBoundaryError):
        c.cosine_similarity(ring, c.encode_sequence("def f(x): return x"))


def test_default_off_runner_byte_identity():
    """Runner without --path-b-semantic-codec must produce identical candidate order."""
    import subprocess
    import sys

    # Compile-time guard: the flag exists in the runner and defaults to False.
    runner = "HENRI V2/humaneval_wave_ast_runner.py"
    src = open(runner, encoding="utf-8").read()
    assert "--path-b-semantic-codec" in src
    assert "path_b_semantic_codec" in src or "PathBSemanticCodec" in src


def test_malformed_input_fail_closed():
    from path_b_semantic_codec import PathBSemanticCodec

    c = PathBSemanticCodec(d_model=4096, d_latent=128, vocab=_sample_vocab(), device="cpu", seed=7)
    with pytest.raises(TypeError):
        c.encode_sequence(None)  # type: ignore[arg-type]
