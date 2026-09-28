"""Contract tests for henri_prefix_kv.py — the honest Task-16 remedy.

The module implements what is ACTUALLY possible on a pointwise egress core:
a default-OFF PrefixConditioner, an adapter from prefix embeddings, a
cache-equivalence check, and a fail-closed capability gate.  A real KV cache is
impossible because henri_decoder.py has no attention core (0 occurrences of
q_proj/k_proj/v_proj/MultiheadAttention/scaled_dot_product/past_key_values/attn/
causal_mask — measured 2026-09-27).

These tests follow the defect classes the architecture catalog records:
  * DEFAULT-PATH BYTE IDENTITY (flag OFF must not change the tensor at all)
  * DEAD-FLAG detection (declared != forwarded != reaches-consumer != has-effect)
  * FAIL-CLOSED on shape / device / non-finite input
  * DIFFERENTIAL EFFECT (two different prefixes must give different outputs)
  * CACHE-EQUIVALENCE (stepwise == batch, exactly)
"""
import os
import sys

import pytest
import torch

def _project_root() -> str:
    """Resolve HENRI V2/ from this test file.

    DEFECT FIXED 2026-09-27: `dirname(dirname(__file__))` from
    `HENRI V2/tests/unit/test_prefix_kv.py` yields `HENRI V2/tests`, so
    `test_decoder_really_has_no_attention_core` hit its `pytest.skip` branch and
    the Case-B determination was NEVER re-derived in CI. Walking up until the
    module under test is present makes the check real.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    d = here
    for _ in range(6):
        if os.path.exists(os.path.join(d, "henri_prefix_kv.py")):
            return d
        d = os.path.dirname(d)
    return here


C = _project_root()
if C not in sys.path:
    sys.path.insert(0, C)

from henri_prefix_kv import (  # noqa: E402
    DEFAULT_USE_PREFIX, PrefixConditioner, PrefixKVError,
    cache_equivalence_stepwise, pool_prefix_embeddings, verify_prefix_capability,
)

D_HID = 32
B, T, P = 2, 5, 3


def _tensors(seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(B, T, D_HID, generator=g).to(device)
    prefix = torch.randn(B, P, D_HID, generator=g).to(device)
    return hidden, prefix


# --------------------------------------------------------------- default off
def test_default_flag_is_off():
    assert DEFAULT_USE_PREFIX is False


def test_default_path_is_byte_identical():
    """Flag OFF + prefix given -> the SAME tensor object, max abs diff exactly 0."""
    hidden, prefix = _tensors()
    cond = PrefixConditioner(D_HID)          # use_prefix defaults to OFF
    out = cond(hidden, prefix)
    assert out is hidden, "default path must return the input tensor unchanged"
    assert float((out - hidden).abs().max()) == 0.0


def test_default_path_ignores_prefix_entirely():
    hidden, prefix = _tensors()
    cond = PrefixConditioner(D_HID)
    a = cond(hidden, prefix)
    b = cond(hidden, None)
    assert float((a - b).abs().max()) == 0.0


# ------------------------------------------------------------------ effect on
def test_flag_on_changes_the_output():
    """DIFFERENTIAL EFFECT: with the flag ON the prefix must move the hidden state."""
    hidden, prefix = _tensors()
    cond = PrefixConditioner(D_HID, use_prefix=True)
    with torch.no_grad():
        cond.prefix_gain.fill_(0.5)
    out = cond(hidden, prefix)
    assert float((out - hidden).abs().max()) > 1e-6


def test_two_different_prefixes_give_different_outputs():
    hidden, prefix = _tensors(seed=1)
    other = _tensors(seed=2)[1]
    cond = PrefixConditioner(D_HID, use_prefix=True)
    with torch.no_grad():
        cond.prefix_gain.fill_(0.5)
    d = float((cond(hidden, prefix) - cond(hidden, other)).abs().max())
    assert d > 1e-6, "the conditioner must be sensitive to WHICH prefix is supplied"


def test_zero_gain_is_inert_even_when_on():
    """A gain of exactly zero means an untrained conditioner cannot perturb the path."""
    hidden, prefix = _tensors()
    cond = PrefixConditioner(D_HID, use_prefix=True)   # gain initialised to zeros
    assert float((cond(hidden, prefix) - hidden).abs().max()) == 0.0


def test_prefix_None_with_flag_on_is_identity():
    hidden, _ = _tensors()
    cond = PrefixConditioner(D_HID, use_prefix=True)
    with torch.no_grad():
        cond.prefix_gain.fill_(0.5)
    assert float((cond(hidden, None) - hidden).abs().max()) == 0.0


# -------------------------------------------------------------- poolings
@pytest.mark.parametrize("pooling", ["mean", "last", "max"])
def test_all_poolings_run(pooling):
    hidden, prefix = _tensors()
    cond = PrefixConditioner(D_HID, use_prefix=True, pooling=pooling)
    out = cond(hidden, prefix)
    assert tuple(out.shape) == (B, T, D_HID)
    assert torch.isfinite(out).all()


def test_bad_pooling_raises():
    with pytest.raises(PrefixKVError):
        PrefixConditioner(D_HID, pooling="median")


# ------------------------------------------------------------- fail closed
def test_wrong_hidden_width_raises():
    cond = PrefixConditioner(D_HID, use_prefix=True)
    with pytest.raises(PrefixKVError):
        cond(torch.randn(B, T, D_HID + 1), torch.randn(B, P, D_HID))


def test_wrong_prefix_width_raises():
    cond = PrefixConditioner(D_HID, use_prefix=True)
    with pytest.raises(PrefixKVError):
        cond(torch.randn(B, T, D_HID), torch.randn(B, P, D_HID + 1))


def test_hidden_2d_is_accepted_and_agrees_with_the_3d_path():
    """2-D [B, d] IS supported: the egress hidden state is [B, d_hidden], a
    pointwise MLP having no sequence axis. The 2-D and 3-D paths must agree."""
    hidden, prefix = _tensors()
    cond = PrefixConditioner(D_HID, use_prefix=True, pooling="last")
    with torch.no_grad():
        cond.prefix_gain.fill_(0.3)
    flat = cond(hidden[:, 0, :], prefix)                 # [B, d]
    assert tuple(flat.shape) == (B, D_HID)
    full = cond(hidden, prefix)
    assert float((flat - full[:, 0, :]).abs().max()) < 1e-12


def test_hidden_rank_1_or_4_raises():
    """Rank other than 2 or 3 must fail closed (the earlier contract said 3)."""
    cond = PrefixConditioner(D_HID, use_prefix=True)
    with pytest.raises(PrefixKVError):
        cond(torch.randn(D_HID), torch.randn(B, P, D_HID))
    with pytest.raises(PrefixKVError):
        cond(torch.randn(B, T, D_HID, 1), torch.randn(B, P, D_HID))


def test_batch_mismatch_fails_closed():
    cond = PrefixConditioner(D_HID, use_prefix=True)
    with torch.no_grad():
        cond.prefix_gain.fill_(0.3)
    with pytest.raises(PrefixKVError):
        cond(torch.randn(B, T, D_HID), torch.randn(B + 1, P, D_HID))


def test_prefix_must_be_3d():
    cond = PrefixConditioner(D_HID, use_prefix=True)
    with pytest.raises(PrefixKVError):
        cond(torch.randn(B, T, D_HID), torch.randn(P, D_HID))


def test_empty_prefix_raises_on_pool():
    cond = PrefixConditioner(D_HID, use_prefix=True)
    with pytest.raises(PrefixKVError):
        cond(torch.randn(B, T, D_HID), torch.randn(B, 0, D_HID))


def test_d_hidden_must_be_positive():
    with pytest.raises(PrefixKVError):
        PrefixConditioner(0)


def test_device_mismatch_fails_closed():
    """A device mismatch must RAISE, never silently move tensors (CUDA-only bug class)."""
    cond = PrefixConditioner(D_HID, use_prefix=True)
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available for a genuine cross-device test")
    with pytest.raises(PrefixKVError):
        cond(torch.randn(B, T, D_HID, device="cuda"), torch.randn(B, P, D_HID, device="cpu"))


# ------------------------------------------------------- cache equivalence
def test_stepwise_equals_batch_for_mean_pooling():
    """A cache/conditioner must be equivalent to recomputation (neutral pooling)."""
    hidden, prefix = _tensors()
    cond = PrefixConditioner(D_HID, use_prefix=True, pooling="mean")
    with torch.no_grad():
        cond.prefix_gain.fill_(0.3)
    assert cache_equivalence_stepwise(cond, hidden, prefix) == 0.0


def test_stepwise_equals_batch_on_default_path():
    hidden, prefix = _tensors()
    cond = PrefixConditioner(D_HID)          # OFF
    assert cache_equivalence_stepwise(cond, hidden, prefix) == 0.0


# --------------------------------------------------------- prefix adapter
def test_adapter_identity_when_width_matches():
    x = torch.randn(1, P, D_HID)
    assert torch.equal(pool_prefix_embeddings(x, D_HID), x)


def test_adapter_folds_when_wider():
    x = torch.randn(1, P, D_HID * 2)
    out = pool_prefix_embeddings(x, D_HID)
    assert tuple(out.shape) == (1, P * 2, D_HID)


def test_adapter_tiles_when_narrower():
    x = torch.randn(1, P, D_HID // 2)
    out = pool_prefix_embeddings(x, D_HID)
    assert tuple(out.shape) == (1, P, D_HID)


def test_adapter_accepts_2d():
    out = pool_prefix_embeddings(torch.randn(P, D_HID), D_HID)
    assert tuple(out.shape) == (1, P, D_HID)


def test_adapter_rejects_non_divisor_fold():
    with pytest.raises(PrefixKVError):
        pool_prefix_embeddings(torch.randn(1, P, D_HID * 2 + 1), D_HID)


def test_adapter_rejects_non_finite():
    x = torch.randn(1, P, D_HID)
    x[0, 0, 0] = float("nan")
    with pytest.raises(PrefixKVError):
        pool_prefix_embeddings(x, D_HID)


def test_adapter_rejects_bad_rank():
    with pytest.raises(PrefixKVError):
        pool_prefix_embeddings(torch.randn(2, 3, 4, 5), D_HID)


# ---------------------------------------------------------- capability gate
def test_gate_passes_only_when_all_four_hold():
    good = verify_prefix_capability(flag_declared=True, flag_forwarded=True,
                                    flag_reaches_consumer=True, output_changed_when_on=True)
    assert good["passed"] is True
    assert good["verdict"] == "PREFIX_WIRING_LIVE_DIAGNOSTIC"


@pytest.mark.parametrize("kwargs", [
    dict(flag_declared=True, flag_forwarded=True, flag_reaches_consumer=True,
         output_changed_when_on=False),                       # declared but NO EFFECT
    dict(flag_declared=True, flag_forwarded=False, flag_reaches_consumer=False,
         output_changed_when_on=False),                       # the dead-store defect
    dict(flag_declared=False, flag_forwarded=True, flag_reaches_consumer=True,
         output_changed_when_on=True),                        # forwarded but undeclared
    dict(flag_declared=True, flag_forwarded=True, flag_reaches_consumer=False,
         output_changed_when_on=True),                        # never reaches a consumer
])
def test_gate_refuses_any_partial_wiring(kwargs):
    r = verify_prefix_capability(**kwargs)
    assert r["passed"] is False
    assert r["verdict"] == "PREFIX_WIRING_REFUSED_DEAD_FLAG_OR_NO_EFFECT"


def test_gate_always_marks_diagnostic_only():
    r = verify_prefix_capability(flag_declared=True, flag_forwarded=True,
                                 flag_reaches_consumer=True, output_changed_when_on=True)
    assert "DIAGNOSTIC" in r["note"]


# --------------------------------------------------------- honest boundary
def test_capability_report_states_no_kv_cache():
    r = PrefixConditioner(D_HID).capability_report()
    assert r["kv_cache_possible"] is False
    assert r["score_eligible"] is False
    assert r["scicode_rerun"] == "BLOCKED"
    assert r["default_path_changed"] is False


def test_report_evidence_class_flips_with_flag():
    assert PrefixConditioner(D_HID).capability_report()["evidence_class"] == "OBSERVED"
    assert PrefixConditioner(D_HID, use_prefix=True).capability_report()["evidence_class"] == "DIAGNOSTIC"


def _strip_comments_and_docstrings(src: str) -> str:
    """Remove comments and string literals before a SYMBOL audit.

    DEFECT FIXED 2026-09-27: the first version counted raw text, so an
    explanatory comment in henri_decoder.py that NAMED the absent attention
    symbols made this test report "attention core present" -- the detector fired
    on its own documentation. A symbol audit must read CODE, never prose.
    """
    import io
    import tokenize
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out.append(tok.string)
    except (tokenize.TokenError, IndentationError):
        return src
    return " ".join(out)


def test_decoder_really_has_no_attention_core():
    """The Case-B determination, re-derived from the live decoder CODE (comments
    and string literals excluded, so prose cannot satisfy or break the check)."""
    p = os.path.join(C, "henri_decoder.py")
    if not os.path.exists(p):
        pytest.skip("henri_decoder.py not present")
    code = _strip_comments_and_docstrings(open(p, encoding="utf-8").read())
    for sym in ("q_proj", "k_proj", "v_proj", "MultiheadAttention",
                "scaled_dot_product", "past_key_values", "causal_mask"):
        assert code.count(sym) == 0, f"{sym} appeared in CODE; re-classify Case A"
