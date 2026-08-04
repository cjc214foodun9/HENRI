"""Contract tests for the default-off projective Hopfield reference component.

These tests are intended for the CUDA verification environment. Running this
file without CUDA skips the tests because local CPU execution is not HENRI
verification evidence.
"""

from __future__ import annotations

import math

import pytest
import torch

from projective_hopfield import ProjectiveHopfieldCleanup


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="HENRI contract tests require CUDA"
)


def _unit_complex(shape: tuple[int, ...], *, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn(shape, generator=generator, device="cpu").to(device)
    imag = torch.randn(shape, generator=generator, device="cpu").to(device)
    value = torch.complex(real, imag)
    return value / torch.linalg.vector_norm(value, dim=-1, keepdim=True)


def test_projective_fidelity_is_invariant_to_independent_local_phases():
    device = torch.device("cuda")
    states = _unit_complex((2, 5, 4), seed=11, device=device)
    memories = _unit_complex((3, 5, 4), seed=12, device=device)
    phases = torch.linspace(0.0, 2.0 * math.pi, 10, device=device).reshape(2, 5)
    phased = states * torch.exp(1j * phases).unsqueeze(-1)

    original = ProjectiveHopfieldCleanup.projective_fidelity(states, memories)
    transformed = ProjectiveHopfieldCleanup.projective_fidelity(phased, memories)

    assert torch.allclose(original, transformed, atol=2e-6, rtol=2e-6)


def test_kernel_matches_independent_leave_one_out_reference():
    device = torch.device("cuda")
    states = _unit_complex((2, 5, 4), seed=21, device=device)
    memories = _unit_complex((3, 5, 4), seed=22, device=device)
    cleanup = ProjectiveHopfieldCleanup(qudit_dim=4).to(device)
    cleanup.store_memories(memories)

    neuron = 2
    kernel = cleanup.local_kernel(states, neuron)

    overlaps = torch.einsum("bnd,pnd->bpn", states.conj(), memories).abs().square()
    n = states.shape[1]
    weights = (2.0 / n) * (
        overlaps.sum(dim=-1) - overlaps[..., neuron] - (n - 1) / memories.shape[-1]
    )
    local = memories[:, neuron, :]
    reference = torch.einsum("bp,pd,pe->bde", weights, local, local.conj())
    reference = 0.5 * (reference + reference.mH)

    assert torch.allclose(kernel, reference, atol=3e-5, rtol=3e-5)
    assert torch.allclose(kernel, kernel.mH, atol=3e-5, rtol=3e-5)


def test_top_eigenvector_diagnostic_matches_cuda_eigh_reference():
    device = torch.device("cuda")
    states = _unit_complex((1, 6, 5), seed=31, device=device)
    memories = _unit_complex((4, 6, 5), seed=32, device=device)
    cleanup = ProjectiveHopfieldCleanup(qudit_dim=5).to(device)
    cleanup.store_memories(memories)

    result = cleanup.retrieve(states, sweeps=1)
    assert torch.isfinite(result.state).all()
    assert torch.isfinite(result.spectral_gap).all()
    assert torch.all(result.spectral_gap >= -3e-5)
    assert torch.allclose(
        result.rayleigh_after,
        result.rayleigh_after.clamp_min(result.rayleigh_before),
        atol=4e-5,
        rtol=4e-5,
    )

    # The returned eigenvalue must be the largest eigenvalue of the exact
    # Hermitian kernel for the first neuron before the asynchronous update.
    initial_kernel = cleanup.local_kernel(states, 0)
    eigenvalues = torch.linalg.eigvalsh(initial_kernel)
    assert torch.allclose(
        result.rayleigh_after[:, 0], eigenvalues[:, -1], atol=4e-5, rtol=4e-5
    )


def test_retrieval_preserves_local_unit_norm_and_returns_selected_memory():
    device = torch.device("cuda")
    memories = _unit_complex((5, 7, 4), seed=41, device=device)
    cue = memories[2:3].clone()
    noise = _unit_complex((1, 7, 4), seed=42, device=device)
    cue = cue + 0.05 * noise

    cleanup = ProjectiveHopfieldCleanup(qudit_dim=4).to(device)
    cleanup.store_memories(memories)
    result = cleanup.retrieve(cue, sweeps=2)

    norms = torch.linalg.vector_norm(result.state, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=3e-5, rtol=3e-5)
    assert result.selected_memory.shape == cue.shape
    assert int(result.selected_memory_index.item()) == 2
    assert torch.isfinite(result.energy).all()


def test_kernel_safety_limit_rejects_unbounded_matrix_allocation():
    device = torch.device("cuda")
    states = _unit_complex((2, 3, 4), seed=51, device=device)
    memories = _unit_complex((2, 3, 4), seed=52, device=device)
    cleanup = ProjectiveHopfieldCleanup(qudit_dim=4, max_kernel_elements=1).to(device)
    cleanup.store_memories(memories)

    with pytest.raises(MemoryError, match="safety limit"):
        cleanup.local_kernel(states, 0)


def test_input_contract_rejects_real_states():
    device = torch.device("cuda")
    cleanup = ProjectiveHopfieldCleanup(qudit_dim=3).to(device)
    with pytest.raises(TypeError, match="complex"):
        cleanup.store_memories(torch.randn(2, 3, 3, device=device))
