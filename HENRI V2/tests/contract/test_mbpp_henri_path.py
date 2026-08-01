"""Contract tests for the MBPP HENRI-path v2 (few-shot W_task compilation)."""

import torch

import mbpp_heldout_pilot as P
from zone_c_epistemic_axiom_harness import HolographicTaskFunctorCompiler, qFHRREpistemicCodec


def test_exemplar_split_bounds():
    ex = P.load_exemplars()
    assert [int(e["task_id"]) for e in ex] == list(range(1, 11))
    for e in ex:
        assert isinstance(e["text"], str) and isinstance(e["code"], str)


def test_fewshot_contract_digest_matches_manifest():
    manifest = P.load_json(P.MANIFEST_PATH)
    info = manifest["prompt_contracts"]["henri_fewshot10"]
    assert info["sha256"] == P.sha256_lf_path(P.ROOT / info["artifact"])


def test_validate_static_bundle_both_paths():
    _, items_h = P.validate_static_bundle("henri")
    assert len(items_h) == 500
    _, items_l = P.validate_static_bundle("legacy")
    assert len(items_l) == 500


def test_w_task_compilation_deterministic():
    codec = qFHRREpistemicCodec(d_model=4096, device="cpu")
    compiler = HolographicTaskFunctorCompiler(codec)
    pairs = [
        (codec.encode_text("demo x one"), codec.encode_text("demo y one")),
        (codec.encode_text("demo x two"), codec.encode_text("demo y two")),
    ]
    w1 = compiler.compile_functor(pairs)
    w2 = compiler.compile_functor(pairs)
    assert torch.equal(w1, w2)
    assert w1.dtype == torch.uint8 and w1.shape == (4096,)
