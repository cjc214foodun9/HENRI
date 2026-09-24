"""UHR-05 contract tests — egress common-mode removal (flag-gated, default OFF).

REWRITTEN after two of my own tests failed. Both failures were DEFECTS IN MY WORK,
and both are worth recording because they are recurring classes:

  DEFECT A (implementation, in `remove_common_mode`): I subtracted the batch mean and
    then renormalized each row. Per-row rescaling by a DIFFERENT positive scalar
    always reintroduces a nonzero batch mean (measured residual 1.25e-3), so the
    function's stated contract was false. FIX: pure subtraction, no renormalization.
    Verified equivalent for the logits: dividing a vector by a positive scalar cannot
    change `F.normalize(h @ proj)`; `argmax_identical_pure_vs_renorm = True`.

  DEFECT B (fixture, the self-confirming class): I built 40 prompts differing by ONE
    integer ("solve task 7 using rotation translation reflection filling"). That
    planted the property under test in the data: it made phasor_bind look common-mode
    dominated (0.687) when varied prompts give 0.087. FIX: varied prompts.

MEASURED WITH VARIED PROMPTS (my own probe, 40 prompts), which the thresholds below
are set against:
    fractional_shift  common_cos 0.8020 -> equivalence 0.425->0.775, order 0.550->0.750
                      random-control diversity delta = +0.04999  (AT the +0.05 bound)
    phasor_bind       common_cos 0.0868 -> equivalence 0.075->0.100, order 1.000->0.975
                      random-control diversity delta = +0.00000

HONEST WEAKNESS (stated, not hidden): on the fractional_shift fixture the random
control gains +0.04999 in top-1 diversity, i.e. `centering` is NOT a pure content
operator on this substrate; it also marginally increases readout diversity. The
inference licensed here is therefore NARROW: centering repairs one measured
shared-component defect and improves semantic invariance on that arm. It is NOT
established as a general fix, and it is default OFF.
"""
from __future__ import annotations

import pathlib
import random
import sys

import pytest
import torch

SB = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SB))

from henri_vla_tokenizer import (  # noqa: E402
    HENRI_EGRESS_CENTER,
    HoloEgressCodebook,
    HoloVLAConfig,
    HoloVLATokenizer,
    remove_common_mode,
)

N = 40
SEED = 20260924
WORDS = [
    "rotation", "translation", "reflection", "gravity", "contour", "symmetry", "parity",
    "filling", "boundary", "spatial", "vector", "manifold", "gradient", "entropy",
    "coherence", "phase", "amplitude", "lattice", "orbit", "generator", "invariant",
    "topology", "holographic", "resonance", "diffusion", "curvature", "tensor",
    "basis", "projection", "subspace", "kernel", "spectral", "harmonic", "wavelet",
    "clifford", "quaternion", "rotor", "bivector", "pseudoscalar", "conjugate",
    "unitary", "hermitian", "adjoint", "canonical", "gauge", "flux", "engram", "prior",
]


def _cfg(mode: str, vocab: int) -> HoloVLAConfig:
    return HoloVLAConfig(
        ambient_dim_D=2048, num_blocks=256, block_slots=8, grid_size_S=16,
        vocab_size_V=vocab, feat_dim=256, position_binding=mode, seed=SEED,
    )


def _prompts(n: int = N) -> list:
    """VARIED prompts. A near-identical fixture plants the property under test."""
    rng = random.Random(SEED)
    return [" ".join(rng.sample(WORDS, 6)) for _ in range(n)]


def _build(mode: str = "fractional_shift"):
    vocab = [f"w{i:03d}" for i in range(48)]
    cfg = _cfg(mode, len(vocab))
    tok = HoloVLATokenizer(cfg)
    code = HoloEgressCodebook(cfg, tok, vocab)
    return tok, code, _prompts()


def _top1(code, tok, strings, center):
    with torch.no_grad():
        return code.logits(tok.encode_text(list(strings)), center=center).argmax(-1).tolist()


def _order_sensitivity(code, tok, prompts, center):
    rng = random.Random(SEED)
    shuf = []
    for p in prompts:
        c = list(p)
        rng.shuffle(c)
        shuf.append("".join(c))
    a = _top1(code, tok, prompts, center)
    b = _top1(code, tok, shuf, center)
    return sum(x != y for x, y in zip(a, b)) / len(prompts)


def _equivalence(code, tok, prompts, center):
    near = [p.replace(" ", "  ") + "   " for p in prompts]
    assert all(nn != p for nn, p in zip(near, prompts)), "near view must differ in bytes"
    a = _top1(code, tok, prompts, center)
    b = _top1(code, tok, near, center)
    return sum(x == y for x, y in zip(a, b)) / len(prompts)


def _random_diversity(code, n=N, center=False):
    g = torch.Generator().manual_seed(SEED + 7)
    with torch.no_grad():
        rw = torch.randn(n, code.cfg.ambient_dim_D, generator=g).to(torch.complex64)
        rw = rw / rw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        ids = code.logits(rw, center=center).argmax(dim=-1).tolist()
    return len(set(ids)) / n


# --------------------------------------------------------------------- controls
def test_default_flag_is_off():
    """The default constant must be OFF, so the default path cannot change."""
    assert HENRI_EGRESS_CENTER is False


def test_default_path_is_byte_identical():
    """`center=False` must equal the pre-UHR-05 behaviour, bit for bit."""
    tok, code, prompts = _build()
    with torch.no_grad():
        w = tok.encode_text(list(prompts))
        a = code.logits(w, center=False)
        b = code.logits(w)          # flag OFF -> resolves to False
    assert torch.equal(a, b), f"default path drifted: max diff {(a - b).abs().max()}"


def test_remove_common_mode_is_exact():
    """EXACTNESS CONTRACT: the batch mean of the output is zero.

    Asserts on the batch-mean NORM, not a cosine: when the mean is exactly zero the
    cosine is 0/0 and carries no information. This is the defect-A fix under test.
    """
    tok, code, prompts = _build()
    with torch.no_grad():
        w = tok.encode_text(list(prompts))
        wc = remove_common_mode(w)
        rel = wc.mean(0).norm().item() / wc.norm(dim=-1).mean().item()
    assert rel < 1e-6, f"batch mean survived centering: relative norm {rel:.3e}"


def test_renormalizing_would_break_the_contract():
    """Regression guard for defect A: re-adding F.normalize reintroduces the mean.

    Documents WHY the function must not renormalize, and proves the old form failed.
    """
    tok, code, prompts = _build()
    with torch.no_grad():
        w = tok.encode_text(list(prompts))
        pure = remove_common_mode(w)
        renorm = torch.nn.functional.normalize(w - w.mean(0, keepdim=True), p=2.0, dim=-1)
        rel_pure = pure.mean(0).norm().item() / pure.norm(dim=-1).mean().item()
        rel_renorm = renorm.mean(0).norm().item() / renorm.norm(dim=-1).mean().item()
    assert rel_pure < 1e-6 < rel_renorm, (
        f"expected pure-subtraction exactness; pure={rel_pure:.3e} renorm={rel_renorm:.3e}")


def test_scale_invariance_no_behaviour_change():
    """Pure subtraction vs renormalized subtraction give the SAME argmax.

    Justifies the defect-A fix as behaviour-preserving on the logits path.
    """
    tok, code, prompts = _build()
    with torch.no_grad():
        w = tok.encode_text(list(prompts))
        renorm = torch.nn.functional.normalize(w - w.mean(0, keepdim=True), p=2.0, dim=-1)
        a = code.logits(remove_common_mode(w), center=False).argmax(-1).tolist()
        b = code.logits(renorm.to(torch.complex64), center=False).argmax(-1).tolist()
    assert a == b, "scale invariance broken: argmax differs between pure and renorm"


def test_centering_improves_fractional_shift_invariance():
    """Reproduce the pre-registered ACCEPT on the arm with a measured shared component."""
    tok, code, prompts = _build("fractional_shift")
    eq0, eq1 = _equivalence(code, tok, prompts, False), _equivalence(code, tok, prompts, True)
    os0, os1 = _order_sensitivity(code, tok, prompts, False), _order_sensitivity(code, tok, prompts, True)
    assert max(eq1 - eq0, os1 - os0) >= 0.05, (
        f"centering did not improve a selector: equivalence {eq0:.3f}->{eq1:.3f}, "
        f"order {os0:.3f}->{os1:.3f}")


def test_phasor_bind_centering_is_not_claimed():
    """Documented REJECT: phasor_bind has no large shared component.

    Guards the narrow claim: if phasor_bind starts showing a big shared component,
    the diagnosis in the module comment must be re-derived. Uses VARIED prompts
    (defect B: the old uniform fixture made this read 0.687 instead of ~0.087).
    """
    tok, code, prompts = _build("phasor_bind")
    with torch.no_grad():
        w = tok.encode_text(list(prompts))
        wn = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        common = (wn @ wn.mean(0, keepdim=True).conj().T).abs().mean().item()
    assert common < 0.5, f"phasor_bind now shows a shared component: {common:.4f}"


def test_random_control_bound_is_stated_not_asserted_tight():
    """C4, HONESTLY BOUNDED. Measured random-control diversity delta = +0.04999.

    The pre-registered bound was +0.05 and the measurement sits ON it, so a tight
    assertion would be knife-edge flaky. This test asserts the DOCUMENTED bound and
    fails if centering starts inflating the random control materially, which is the
    property that actually matters: centering must not become a diversity dial.
    """
    tok, code, prompts = _build("fractional_shift")
    d0 = _random_diversity(code, center=False)
    d1 = _random_diversity(code, center=True)
    delta = d1 - d0
    assert delta <= 0.05, (
        f"centering boosted the RANDOM control by {delta:+.4f}; it would then be a "
        f"diversity dial rather than a content repair")


def test_dead_input_cannot_manufacture_diversity():
    """DEAD-INPUT negative control: identical input yields ONE top-1, centered or not.

    A batch of identical waves has zero common mode after subtraction, so every
    centered row is the zero vector and all logits are identical. If this ever yields
    more than one distinct top-1, centering is manufacturing signal from nothing.
    """
    tok, code, _ = _build()
    same = ["identical prompt for every row"] * 8
    a = _top1(code, tok, same, False)
    b = _top1(code, tok, same, True)
    assert len(set(a)) == 1, "baseline already non-deterministic on identical input"
    assert len(set(b)) == 1, f"centering manufactured diversity: {len(set(b))} distinct top-1"
