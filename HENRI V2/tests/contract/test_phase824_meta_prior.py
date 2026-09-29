"""Phase 8.24 contract tests — Meta-D_a fast-adaptation prior.

Spec: HENRI-ANALYSIS-2026-08-SOLVING-FRONTIER (sha 8c508808...).
Covers: default-OFF flag, causal consumer proof (flag -> pretrain call),
and source-level gate pre-registration.
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "HENRI V2" / "production_arc_run.py"
MODULE = ROOT / "HENRI V2" / "henri_external_outcome_refactor_module.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def test_824_meta_prior_default_off():
    src = _read(RUNNER)
    assert 'os.environ.get("HENRI_ARC_META_PRIORS", "0") == "1"' in src
    assert "pretrain_action_generators" in src


def test_824_meta_prior_causal_consumer():
    # Flag must reach a computation that changes the store BEFORE the
    # planner consumes it (wired at store construction, default OFF).
    #
    # STALE PIN ADVANCED 2026-09-28 (measured). This assertion previously
    # required the literal `num_channels=8192`. That literal was removed when the
    # macro-field RESOLUTION defect was fixed: the store is now constructed from
    # the shared `_macro_num_blocks()` helper so that BOTH coupled consumers
    # (ActionOutcomeGeneratorStore and _pad_su3_field) resolve the same block
    # count. A hardcoded 8192 against a 64-block field was MEASURED to raise
    # `einsum(): subscript n has size 8192 ... does not broadcast with 64`.
    #
    # The PIN IS NOT DELETED -- it is advanced to the invariant that actually
    # discriminates now: the store's channel count must come from the single
    # shared helper, and the prior must receive THAT SAME value. Pinning the
    # literal again would re-assert the defect. The dedicated coverage for the
    # coupling lives in tests/contract/test_macro_field_resolution.py (M1/M1b).
    src = _read(RUNNER)
    assert "pretrain_action_generators(" in src
    assert "_num_channels = _macro_num_blocks()" in src, (
        "the store's channel count must come from the shared resolution helper")
    assert "num_channels=_num_channels" in src, (
        "the store must be constructed at the shared resolution")
    # the prior is passed the SAME resolved value (not a second literal).
    # Bounded window: the call spans two lines and the first ')' belongs to
    # `str(DEVICE)`, so a split on ')' truncates before `num_channels`.
    _callsite = src.find("pretrain_action_generators(")
    window = src[_callsite:_callsite + 400]
    assert "num_channels=_num_channels" in window, (
        "the prior must use the same resolved channel count as the store; "
        f"call window was {window[:120]!r}")


def test_824_zero_pretraining_invariant():
    # The prior must use synthetic SU(3) fields only — no ARC grids,
    # no solution labels, no task content.
    src = _read(MODULE)
    assert "_rand_special_unitary" in src
    assert "synthetic" in src.lower()
    for leak in ("grid", "solution", "examples", "arcade"):
        assert not re.search(rf"\b{leak}\b", src, re.I), f"leak term: {leak}"


def test_824_gate_pre_registered():
    src = _read(MODULE)
    assert "gate <= 3" in src
    assert "expect >= 15" in src
    assert "default-OFF" in src.lower() or "inert" in src.lower()


def test_824_cuda_device_threading_d43():
    # D43 (2026-08-17): pretrain crashed on the live CUDA promotion path —
    # RuntimeError in torch.einsum (bmm): mat2 on cuda:0 vs other on cpu.
    # _affine_family_theta allocates on CPU (seedable generator) but the
    # caller einsums against the CUDA gell_mann basis. Fix: thread device
    # through _affine_family_theta and .to(device) the result. Local CPU
    # gates cannot see this; the live-mode consumer trace can.
    src = _read(MODULE)
    assert 'device: str = "cpu"' in src, "device param missing"
    # Whitespace-normalized match: the call site spans a plain newline.
    norm = "".join(src.split())
    assert "_affine_family_theta(a,num_channels,seed=seed+a,device=device)" in norm, (
        "call site must pass device")
    assert "torch.einsum(\"n,a->na\", pat, d)).to(device)" in src, "result must .to(device)"
