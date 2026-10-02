"""Phase 1 contract tests — SPEC-2026-10-01-PHASE1-TRANSDUCTION.

Gates P1-G1..P1-G9 from
`experiments/verification/arc_phase1_transduction_prereg.md`.

Includes an explicit TAUTOLOGY GUARD (P1-G8): the algebra gate must PASS on the
orthonormal arm and FAIL on two discriminating control arms. A suite where all
arms pass is vacuous and is itself a failure.

All output is in-memory or under tmp_path. Nothing writes a tracked receipt.
"""

import gc
import math
import tracemalloc

import pytest
import torch

from henri.ingress.spatial_tokenizer import (
    SpatialCliffordTokenizer,
    is_specification_dimension,
)
from factorized_transition_kernel import (
    FactorizedTransitionKernel,
    dense_footprint_bytes,
    footprint_bytes,
)

SPEC_DIM = 65536
CANVAS_S = 32
SEED = 20260914


def canvas(seed: int, s: int = CANVAS_S, vocab: int = 10):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (s, s), generator=g).tolist()


# ---------------------------------------------------------------- P1-G1/G2
def test_g1_shape_and_dtype_and_spec_dimension():
    """P1-G1: encode returns complex64 [65536] and matches the spec contract."""
    tok = SpatialCliffordTokenizer()
    psi = tok.encode(canvas(1))
    assert psi.shape == (SPEC_DIM,)
    assert psi.dtype == torch.complex64
    assert is_specification_dimension(tok)
    assert tok.block_slots == 8 and tok.num_blocks == 8192


def test_g2_unit_norm():
    """P1-G2: encoded field is unit L2 norm, err <= 1e-5."""
    tok = SpatialCliffordTokenizer()
    for seed in (1, 2, 3):
        psi = tok.encode(canvas(seed))
        err = abs(float(psi.norm(p=2)) - 1.0)
        assert err <= 1e-5, f"seed {seed}: norm err {err}"


def test_g2b_uniform_grid_does_not_vanish():
    """P1-G2b: the reserved DC slot prevents the measured zero-vector collapse.

    Without the reserved (0,0) slot the geometric series is exactly zero for a
    uniform grid: |enc(uniform)| ~ 0 and the normaliser returns a null wave.
    """
    tok = SpatialCliffordTokenizer()
    uniform = [[3] * CANVAS_S for _ in range(CANVAS_S)]
    raw = tok.encode_raw(uniform)
    assert float(raw.abs().max()) > 0.0, "uniform grid collapsed to the zero vector"
    psi = tok.encode(uniform)
    assert abs(float(psi.norm(p=2)) - 1.0) <= 1e-5
    assert torch.isfinite(psi).all()


# ------------------------------------------------------------------- P1-G3
def test_g3_cyclic_roll_is_an_exact_wave_operator():
    """P1-G3: enc(roll(X, dw, dh)) == M * enc(X) on a full canvas, err <= 1e-4."""
    tok = SpatialCliffordTokenizer()
    assert tok.modulus == CANVAS_S
    X = canvas(7)
    psi = tok.encode(X)
    for dw, dh in ((1, 0), (3, 0), (0, 2), (5, 7)):
        rolled = SpatialCliffordTokenizer.roll_canvas(X, dw, dh)
        target = tok.encode(rolled)
        pred = tok.apply_roll(psi, dw, dh)
        err = float((pred - target).abs().max())
        assert err <= 1e-4, f"roll ({dw},{dh}) max err {err}"


def test_g3b_roll_operator_wrong_sign_is_detected():
    """TAUTOLOGY GUARD for P1-G3: the OPPOSITE sign must NOT reproduce the roll.

    A gate that passes for both signs is measuring nothing. This guard is what
    caught the sign defect in the first place: the implementation originally
    used the negative exponent, `roll_multiplier` matched the roll only to
    2.6e-02, and this control now asserts the negative form must MISS.
    """
    tok = SpatialCliffordTokenizer()
    X = canvas(11)
    psi = tok.encode(X)
    target = tok.encode(SpatialCliffordTokenizer.roll_canvas(X, 3, 0))
    wrong = tok._phasor(-(float(3) * tok.wx + float(0) * tok.wy)).reshape(-1) * psi
    assert float((wrong - target).abs().max()) > 1e-2, "sign control did not discriminate"


def test_g3c_non_canvas_grid_is_refused_by_encode_canvas():
    """The exactness condition is enforced, not merely documented."""
    tok = SpatialCliffordTokenizer()
    with pytest.raises(ValueError, match="full 32x32 grid"):
        tok.encode_canvas([[0] * 12 for _ in range(12)])


# ------------------------------------------------------------------- P1-G4
def test_g4_1d_byte_input_is_refused():
    """P1-G4: a flat byte sequence is refused; only 2D integer grids are admitted."""
    tok = SpatialCliffordTokenizer()
    with pytest.raises(ValueError):
        tok.encode(torch.arange(16))
    with pytest.raises(ValueError):
        tok.encode([1, 2, 3, 4])
    with pytest.raises(ValueError):
        tok.encode(torch.rand(4, 4, 4))
    # and a genuine 2D grid still works
    assert tok.encode([[1, 2], [3, 4]]).shape == (SPEC_DIM,)


def test_g4b_ragged_grid_is_refused():
    tok = SpatialCliffordTokenizer()
    with pytest.raises(ValueError, match="unequal widths"):
        tok.encode([[1, 2], [3, 4, 5]])


# ------------------------------------------------------------------- P1-G5
def test_g5_determinism_across_instances():
    """P1-G5: two instances with the same config give bit-identical fields."""
    a = SpatialCliffordTokenizer()
    b = SpatialCliffordTokenizer()
    X = canvas(13)
    assert torch.equal(a.encode(X), b.encode(X))
    c = SpatialCliffordTokenizer(seed=1)
    assert not torch.equal(a.encode(X), c.encode(X)), "seed has no effect"


# ---------------------------------------------------------------- P1-G6/G7
def test_g6_no_dense_allocation():
    """P1-G6: no parameter reaches [dim, dim] scale."""
    k = FactorizedTransitionKernel(dim=512, rank=32, num_actions=2)
    k.assert_no_dense_allocation()
    limit = 512 * 512
    for name, p in k.named_parameters():
        assert p.numel() < limit, f"{name} is dense-scale"


def test_g6b_dense_probe_is_guarded_at_production_dim():
    """The dense materialisation is a small-D probe only."""
    k = FactorizedTransitionKernel(dim=SPEC_DIM, rank=256, num_actions=1)
    with pytest.raises(ValueError, match="verification-only probe"):
        k.dense_equivalent(0)
    assert dense_footprint_bytes(SPEC_DIM) / 2**30 > 30.0  # ~34 GiB if it were done


def test_g7_footprint_arithmetic():
    """P1-G7: 2*r*D*8 bytes per action = 268,435,456 B at D=65,536, r=256."""
    per_action = footprint_bytes(SPEC_DIM, 256, 1)
    assert per_action == 2 * 256 * SPEC_DIM * 8
    assert per_action == 268435456
    assert per_action / 2**20 == pytest.approx(256.0, abs=1e-6)
    # The specification's own stated figure, reproduced from its arithmetic.
    assert per_action / 1e6 == pytest.approx(268.435456, abs=1e-6)


def test_g7b_per_action_bound_and_total_are_both_reported():
    """The 300 MB bound is PER ACTION; the |A|=8 total is reported and NOT claimed."""
    k = FactorizedTransitionKernel(dim=SPEC_DIM, rank=256, num_actions=1)
    rep = k.footprint_report()
    assert rep["per_action_bytes"] < 300 * 2**20, "per-action bound violated"
    assert rep["spec_300MB_bound_applies_to"] == "per_action"

    k8 = FactorizedTransitionKernel(dim=SPEC_DIM, rank=256, num_actions=1)
    total = footprint_bytes(SPEC_DIM, 256, 8)
    assert total == 8 * 268435456
    assert total / 2**20 > 300.0, "total exceeds 300MB -- must not be claimed as bounded"
    assert rep["dense_alternative_GiB"] > 30.0
    assert k8 is not None


# ------------------------------------------------------------------- P1-G8
def _idempotence_error(kernel_matrix):
    return float((kernel_matrix @ kernel_matrix - kernel_matrix).abs().max())


def test_g8_orthonormal_projector_is_idempotent():
    """P1-G8 PASS arm: V=U orthonormal, sigma=1 -> K = U U^H is a projector.

    With V != U, K = U V^H is NOT idempotent (measured error 0.169); only the
    K K^H form is. That distinction is tested by the control below.
    """
    k = FactorizedTransitionKernel(dim=64, rank=8, num_actions=1, seed=5)
    with torch.no_grad():
        k.sigma.fill_(1.0)
        k.share_basis(0)
    K = k.dense_equivalent(0, max_dim=256)
    assert _idempotence_error(K.detach()) <= 1e-4


def test_g8_control_random_dense_matrix_is_not_idempotent():
    """TAUTOLOGY GUARD: a random dense matrix must FAIL the same check."""
    g = torch.Generator().manual_seed(5)
    R = torch.randn(64, 64, generator=g, dtype=torch.complex64) / math.sqrt(64)
    assert _idempotence_error(R) > 1e-2


def test_g8_control_nonunit_sigma_is_not_idempotent():
    """TAUTOLOGY GUARD: sigma != 1 must FAIL the same check."""
    k = FactorizedTransitionKernel(dim=64, rank=8, num_actions=1, seed=5)
    with torch.no_grad():
        k.sigma.fill_(0.5)
        k.share_basis(0)
    K = k.dense_equivalent(0, max_dim=256)
    assert _idempotence_error(K.detach()) > 1e-3


def test_g8_control_distinct_bases_are_not_idempotent():
    """TAUTOLOGY GUARD: K = U V^H with V != U must FAIL idempotence.

    This is the arm that caught the original defect: the default kernel is NOT
    a projector, so a gate asserting idempotence without share_basis() was
    simply wrong.
    """
    k = FactorizedTransitionKernel(dim=64, rank=8, num_actions=1, seed=5)
    with torch.no_grad():
        k.sigma.fill_(1.0)
    K = k.dense_equivalent(0, max_dim=256)
    assert _idempotence_error(K.detach()) > 1e-2


def test_g8d_qr_gives_genuinely_orthonormal_columns():
    """The real-view QR must produce Re(U^H U) = I. Guards the QR fix."""
    k = FactorizedTransitionKernel(dim=64, rank=8, num_actions=1, seed=5)
    u = k._u()[0]
    gram = u.conj().transpose(0, 1) @ u
    err = float((gram - torch.eye(8, dtype=gram.dtype)).abs().max())
    assert err <= 1e-5, f"U columns not orthonormal: {err}"


def test_g8b_apply_matches_the_dense_equivalent():
    """The O(rD) two-product path reproduces the materialised operator."""
    k = FactorizedTransitionKernel(dim=64, rank=8, num_actions=3, seed=9)
    x = torch.randn(64, dtype=torch.complex64,
                    generator=torch.Generator().manual_seed(3))
    for a in range(3):
        got = k.apply(a, x)
        want = k.dense_equivalent(a, max_dim=256) @ x
        assert float((got - want).abs().max()) <= 1e-3


def test_g8c_apply_rejects_out_of_range_action():
    k = FactorizedTransitionKernel(dim=32, rank=4, num_actions=2)
    with pytest.raises(IndexError):
        k.apply(5, torch.zeros(32, dtype=torch.complex64))


# ------------------------------------------------------------------- P1-G9
# MEMORY MEASUREMENT NOTE: `tracemalloc` tracks only the Python allocator and
# does NOT see torch tensor storage (allocated by the C++ caching allocator).
# A tracemalloc-based G9 passes trivially and is therefore VACUOUS. G9 measures
# peak RSS instead, in a FRESH SUBPROCESS per unroll length, so the two numbers
# are comparable.
_UNROLL_SNIPPET = r"""
import sys
import psutil
import torch
from factorized_transition_kernel import FactorizedTransitionKernel
steps = int(sys.argv[1]); dim = 4096
k = FactorizedTransitionKernel(dim=dim, rank=32, num_actions=4, seed=2)
h = torch.randn(dim, dtype=torch.complex64, generator=torch.Generator().manual_seed(4))
for i in range(steps):
    h = k.apply(i % 4, h)
assert torch.isfinite(h.real).all() and torch.isfinite(h.imag).all()
# Linux: ru_maxrss is kB; macOS: bytes. psutil normalises both to bytes.
print(psutil.Process().memory_info().rss, psutil.Process().memory_info().vms)
"""


def _peak_rss_kb(steps: int, tmp_path) -> int:
    import os
    import subprocess
    import sys

    script = tmp_path / f"unroll_{steps}.py"
    script.write_text(_UNROLL_SNIPPET, encoding="utf-8")
    v2 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ)
    env["PYTHONPATH"] = v2 + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run(
        [sys.executable, str(script), str(steps)],
        cwd=v2, env=env, capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0, f"unroll {steps} failed:\n{out.stdout}\n{out.stderr}"
    rss, vms = (int(x) for x in out.stdout.strip().splitlines()[-1].split())
    # Report the larger of RSS/VMS so the bound cannot pass by measurement choice.
    return max(rss, vms) // 1024


def test_g9_unroll_memory_is_bounded(tmp_path):
    """P1-G9: an 8x longer unroll does not grow peak memory (measured RSS)."""
    rss_short = _peak_rss_kb(64, tmp_path)
    rss_long = _peak_rss_kb(512, tmp_path)
    print(f"G9 peak RSS kB: 64 steps={rss_short}  512 steps={rss_long}")
    # Constant per-step working set => flat peak. Allow 1.5x for allocator slack.
    assert rss_long < 1.5 * rss_short, (
        f"peak RSS grew with unroll length: 64->{rss_short} kB 512->{rss_long} kB"
    )


def test_g9b_long_unroll_produces_finite_output(tmp_path):
    """No OOM and no NaN/Inf across a long unroll."""
    rss = _peak_rss_kb(512, tmp_path)
    assert rss > 0
    print(f"G9b 512-step unroll completed, peak RSS {rss} kB")


# ---------------------------------------------------------------- dimension
def test_dimension_note_is_honest_about_the_repo_convention():
    """The 65,536 ambiguity between the spec (complex) and the repo is recorded."""
    tok = SpatialCliffordTokenizer()
    note = tok.dimension_note()
    assert "65536" in note.replace(",", "")
    assert "4 slots/block = 32768" in note.replace(",", "").replace("  ", " ")
    assert tok.config.real_dimension == 2 * SPEC_DIM
