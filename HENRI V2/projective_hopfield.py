"""Reference projective Hopfield cleanup on :math:`CP^{d-1}`.

This module is an experiment-only reference implementation of the local update
in Galitski, *High-Capacity Generalized Hopfield Networks*. It is deliberately
separate from ``ContinuousHopfieldCleanup``. It does not reinterpret HENRI
qFHRR rings, Clifford rows, or Zone C records as qudit states.

State contract
--------------
``states`` has shape ``[B, N, d]`` and ``memories`` has shape ``[P, N, d]``.
Both tensors must be complex. Each local qudit is normalized and is treated as
an element of ``CP^(d-1)``: multiplication by an independent unit phase does
not change its fidelity.

The implementation uses the paper's leave-one-neuron-out overlap and local
Hermitian kernel::

    O_mu^(i) = (2/N) * sum_{j != i} (|<s_j, xi_j^mu>|^2 - 1/d)
    K_i      = sum_mu O_mu^(i) |xi_i^mu><xi_i^mu|

The asynchronous sweep selects the largest-eigenvalue eigenvector of ``K_i``
for one neuron at a time. The paper's energy monotonicity result applies to
its stated generalized LLG flow and assumptions. This finite-precision
asynchronous reference does not claim global energy monotonicity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ProjectiveRetrievalResult:
    """Outputs and diagnostics from one or more asynchronous sweeps."""

    state: torch.Tensor
    selected_memory: torch.Tensor
    selected_memory_index: torch.Tensor
    projective_fidelity: torch.Tensor
    spectral_gap: torch.Tensor
    hermitian_residual: torch.Tensor
    energy: torch.Tensor
    rayleigh_before: torch.Tensor
    rayleigh_after: torch.Tensor
    sweeps: int


class ProjectiveHopfieldCleanup(nn.Module):
    """Reference SU(d) Hopfield update on local projective qudit states.

    Parameters
    ----------
    qudit_dim:
        Local complex state dimension ``d``.
    max_kernel_elements:
        Safety limit for one local kernel batch. The reference implementation
        is intended for small and medium experimental ``d``; it must not be
        used as an implicit production ``D=65,536`` allocator.
    """

    def __init__(self, qudit_dim: int, *, max_kernel_elements: int = 16_777_216):
        super().__init__()
        if qudit_dim < 2:
            raise ValueError("qudit_dim must be >= 2")
        if max_kernel_elements < 1:
            raise ValueError("max_kernel_elements must be positive")
        self.qudit_dim = int(qudit_dim)
        self.max_kernel_elements = int(max_kernel_elements)
        self.register_buffer(
            "memories", torch.empty(0, 0, self.qudit_dim, dtype=torch.complex64)
        )

    @staticmethod
    def _require_states(states: torch.Tensor, *, name: str) -> None:
        if not isinstance(states, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if states.ndim != 3:
            raise ValueError(f"{name} must have shape [B,N,d] or [P,N,d]")
        if not states.is_complex():
            raise TypeError(f"{name} must use a complex dtype")
        if not torch.isfinite(states.real).all() or not torch.isfinite(states.imag).all():
            raise ValueError(f"{name} contains NaN or Inf")

    def _validate_pair(
        self, states: torch.Tensor, memories: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._require_states(states, name="states")
        self._require_states(memories, name="memories")
        if states.shape[-1] != self.qudit_dim or memories.shape[-1] != self.qudit_dim:
            raise ValueError(
                f"last dimension must equal qudit_dim={self.qudit_dim}; "
                f"got states={states.shape[-1]}, memories={memories.shape[-1]}"
            )
        if states.shape[1] != memories.shape[1]:
            raise ValueError("states and memories must have the same neuron count")
        if states.shape[1] < 2:
            raise ValueError("at least two neurons are required for leave-one-out overlap")
        if states.device != memories.device:
            raise ValueError("states and memories must be on the same device")
        if states.dtype != memories.dtype:
            memories = memories.to(dtype=states.dtype)
        return self._normalize(states), self._normalize(memories)

    @staticmethod
    def _normalize(x: torch.Tensor) -> torch.Tensor:
        norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        if torch.any(norms <= torch.finfo(norms.dtype).eps):
            raise ValueError("projective states must have non-zero local norm")
        return x / norms

    @torch.no_grad()
    def store_memories(self, memories: torch.Tensor) -> int:
        """Replace the frozen memory bank and return its pattern count."""
        self._require_states(memories, name="memories")
        if memories.shape[-1] != self.qudit_dim:
            raise ValueError(
                f"memories last dimension {memories.shape[-1]} != {self.qudit_dim}"
            )
        if memories.shape[1] < 2:
            raise ValueError("at least two neurons are required")
        self.memories = self._normalize(memories).detach().clone()
        return int(self.memories.shape[0])

    def _resolve_memories(self, memories: Optional[torch.Tensor]) -> torch.Tensor:
        resolved = self.memories if memories is None else memories
        if resolved.numel() == 0:
            raise RuntimeError("no memories stored; call store_memories first")
        return resolved

    @staticmethod
    def projective_fidelity(
        states: torch.Tensor, memories: torch.Tensor
    ) -> torch.Tensor:
        """Return mean local projective fidelity with shape ``[B,P]``.

        The absolute square removes the arbitrary phase of each local qudit.
        """
        state = ProjectiveHopfieldCleanup._normalize(states)
        memory = ProjectiveHopfieldCleanup._normalize(memories)
        overlap = torch.einsum("bnd,pnd->bpn", state.conj(), memory)
        return overlap.abs().square().mean(dim=-1)

    def _leave_one_out_weights(
        self, states: torch.Tensor, memories: torch.Tensor, neuron: int
    ) -> torch.Tensor:
        """Return ``O_mu^(i)`` with shape ``[B,P]``."""
        overlap = torch.einsum("bnd,pnd->bpn", states.conj(), memories).abs().square()
        n = states.shape[1]
        centered_sum = overlap.sum(dim=-1) - overlap[..., neuron]
        return (2.0 / n) * (centered_sum - (n - 1) / self.qudit_dim)

    def _raw_local_kernel(
        self,
        states: torch.Tensor,
        neuron: int,
        memories: torch.Tensor,
    ) -> torch.Tensor:
        """Build the unsymmetrized finite-precision kernel."""
        state, memory = self._validate_pair(states, memories)
        n = state.shape[1]
        if not 0 <= neuron < n:
            raise IndexError(f"neuron index {neuron} outside [0,{n})")
        batch_size, _, d = state.shape[0], memory.shape[0], self.qudit_dim
        if batch_size * d * d > self.max_kernel_elements:
            raise MemoryError(
                "projective kernel safety limit exceeded: "
                f"batch={batch_size}, d={d}, elements={batch_size*d*d}, "
                f"limit={self.max_kernel_elements}"
            )
        weights = self._leave_one_out_weights(state, memory, neuron)
        local_memory = memory[:, neuron, :]
        return torch.einsum(
            "bp,pd,pe->bde", weights, local_memory, local_memory.conj()
        )

    def local_kernel(
        self,
        states: torch.Tensor,
        neuron: int,
        memories: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the Hermitian kernel ``K_i`` with shape ``[B,d,d]``."""
        memory = self._resolve_memories(memories)
        raw_kernel = self._raw_local_kernel(states, neuron, memory)
        # Round-off can introduce a small anti-Hermitian component. Symmetrize
        # it before eigh. The retrieval result reports the pre-symmetrization
        # residual so this operation cannot hide a construction error.
        return 0.5 * (raw_kernel + raw_kernel.mH)

    @staticmethod
    def _rayleigh(state: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,bde,be->b", state.conj(), kernel, state).real

    def mean_field_energy(
        self, states: torch.Tensor, memories: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the finite-network centered-overlap energy surrogate.

        This is the centered quadratic overlap term used for diagnostics:
        ``E = -N/2 * sum_mu O_mu^2``. It is not a claim that every asynchronous
        finite-precision sweep decreases this value.
        """
        memory = self._resolve_memories(memories)
        state, memory = self._validate_pair(states, memory)
        overlap = torch.einsum("bnd,pnd->bpn", state.conj(), memory).abs().square()
        centered = overlap.mean(dim=-1) - (1.0 / self.qudit_dim)
        n = state.shape[1]
        return -0.5 * n * centered.square().sum(dim=-1)

    @torch.no_grad()
    def retrieve(
        self,
        states: torch.Tensor,
        *,
        memories: Optional[torch.Tensor] = None,
        sweeps: int = 1,
        convergence_tol: float = 1e-6,
    ) -> ProjectiveRetrievalResult:
        """Run asynchronous top-eigenvector sweeps.

        ``sweeps=1`` is the default reference operation. The selected memory is
        the pattern with the largest mean projective fidelity after retrieval.
        """
        if not isinstance(sweeps, int) or sweeps < 1:
            raise ValueError("sweeps must be a positive integer")
        if convergence_tol < 0:
            raise ValueError("convergence_tol must be non-negative")
        memory = self._resolve_memories(memories)
        current, memory = self._validate_pair(states, memory)
        before_energy = self.mean_field_energy(current, memory)
        last_gap = torch.zeros(current.shape[0], current.shape[1], device=current.device)
        last_residual = torch.zeros_like(last_gap)
        rayleigh_before = torch.zeros_like(last_gap)
        rayleigh_after = torch.zeros_like(last_gap)
        previous = current
        actual_sweeps = 0

        for sweep_index in range(sweeps):
            actual_sweeps = sweep_index + 1
            for neuron in range(current.shape[1]):
                raw_kernel = self._raw_local_kernel(current, neuron, memory)
                last_residual[:, neuron] = torch.linalg.matrix_norm(
                    raw_kernel - raw_kernel.mH, ord="fro"
                )
                kernel = 0.5 * (raw_kernel + raw_kernel.mH)
                eigvals, eigvecs = torch.linalg.eigh(kernel)
                top = eigvecs[..., -1]
                rayleigh_before[:, neuron] = self._rayleigh(current[:, neuron, :], kernel)
                rayleigh_after[:, neuron] = eigvals[..., -1]
                last_gap[:, neuron] = eigvals[..., -1] - eigvals[..., -2]
                current[:, neuron, :] = top
                current[:, neuron, :] = self._normalize(current[:, neuron, :].unsqueeze(1)).squeeze(1)

            change = torch.linalg.vector_norm(current - previous, dim=-1).amax(dim=-1)
            previous = current.clone()
            if torch.all(change <= convergence_tol):
                break

        fidelity = self.projective_fidelity(current, memory)
        selected = torch.argmax(fidelity, dim=-1)
        selected_memory = memory[selected]
        energy = self.mean_field_energy(current, memory)
        # Keep the tensor on the active device and expose the finite diagnostic.
        if not torch.isfinite(current).all():
            raise FloatingPointError("projective retrieval produced non-finite state")
        return ProjectiveRetrievalResult(
            state=current,
            selected_memory=selected_memory,
            selected_memory_index=selected,
            projective_fidelity=fidelity,
            spectral_gap=last_gap,
            hermitian_residual=last_residual,
            energy=torch.stack([before_energy, energy], dim=-1),
            rayleigh_before=rayleigh_before,
            rayleigh_after=rayleigh_after,
            sweeps=actual_sweeps,
        )


__all__ = ["ProjectiveHopfieldCleanup", "ProjectiveRetrievalResult"]
