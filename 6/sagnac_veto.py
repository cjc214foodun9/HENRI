"""
sagnac_veto.py  —  PHASE 1 CORRECTION

Replaces the tautological interferometer with a physically meaningful
projection (homodyne) decomposition of the candidate wave against the
target manifold.

------------------------------------------------------------------------
THE BUG (old code):
    reflection_delta   = current_wave - target_manifold
    transmission_truth = current_wave - reflection_delta   # == target_manifold

    => transmission_truth collapses identically to target_manifold for ANY
       candidate. The "truth" that flows downstream is the stored target,
       so Hopfield cleanup matches the target against itself, centroids
       drift toward the target's topology (not the candidate's), and the
       loop reports CONVERGED by construction. The candidate is never tested.

------------------------------------------------------------------------
THE FIX (this file):
    Decompose the candidate wave psi into the component that lies ALONG the
    target direction (the constructively-interfering / transmitted part) and
    the orthogonal residual (the destructively-interfering / reflected part).

    Let t_hat = target / ||target||  (unit target manifold direction).
    Coherent (homodyne) projection coefficient:

        c = <t_hat, psi>            (complex inner product, conj on t_hat)

    Transmission (truth, the part of psi that agrees with the target):

        truth  = c * t_hat

    Reflection (error, the part of psi orthogonal to the target):

        delta  = psi - truth

    Error energy (genuine misalignment, NOT a tautology):

        E = ||delta||^2 = ||psi||^2 - |c|^2 / ||t_hat||^2   (= ||psi||^2 - |c|^2)

    Properties that make this correct:
      * truth depends on psi (it is psi's shadow on the target axis), so a
        candidate that disagrees with the target produces a DIFFERENT truth.
      * <truth, delta> = 0 exactly (orthogonal decomposition), so energy is
        partitioned cleanly:  ||psi||^2 = ||truth||^2 + ||delta||^2.
      * E = 0  iff  psi is parallel to the target (perfect logical agreement).
      * E = ||psi||^2  iff  psi is orthogonal to the target (total contradiction).
      * This IS the balanced coherent homodyne readout the manuscript adopted:
        c is exactly the homodyne inner product, |c|^2 the transmitted
        intensity, E the reflected-port intensity.

This file is drop-in compatible with the old SagnacInterferometer.forward
signature: it returns (transmission_truth, reflection_delta, error_energy).
"""

import torch
import torch.nn as nn


class SagnacInterferometer(nn.Module):
    """
    Physically-grounded coherent projection interferometer.

    forward(current_wave, target_manifold) -> (truth, delta, error_energy)

    The decomposition is an orthogonal (Hilbert-space) projection of the
    candidate wave onto the target manifold direction. It is the discrete
    analogue of a balanced homodyne measurement: the transmitted field is the
    in-phase (constructive) component, the reflected field is the orthogonal
    (destructive) residual.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, current_wave: torch.Tensor, target_manifold: torch.Tensor):
        psi = current_wave
        tgt = target_manifold

        # Flatten to a single complex vector for the inner products; we
        # restore the original shape on the way out so downstream reshape(64,64)
        # etc. keeps working.
        orig_shape = psi.shape
        psi_flat = psi.reshape(-1)
        tgt_flat = tgt.reshape(-1)

        if psi_flat.shape != tgt_flat.shape:
            raise ValueError(
                f"SagnacInterferometer: candidate shape {tuple(psi.shape)} "
                f"!= target shape {tuple(tgt.shape)}. Dimensions must match "
                f"before projection (check L3 router output and target_manifold "
                f"dimensionality)."
            )

        # Promote to complex so the inner products are well defined even if a
        # real wave is passed in.
        if not torch.is_complex(psi_flat):
            psi_flat = psi_flat.to(torch.complex64)
        if not torch.is_complex(tgt_flat):
            tgt_flat = tgt_flat.to(torch.complex64)

        # Target energy and unit direction.
        tgt_energy = torch.real(torch.vdot(tgt_flat, tgt_flat))  # ||tgt||^2, real >= 0
        tgt_energy = torch.clamp(tgt_energy, min=self.eps)

        # Coherent homodyne projection coefficient c = <tgt, psi> / ||tgt||^2.
        # torch.vdot(a, b) conjugates the FIRST argument -> <tgt, psi>.
        inner = torch.vdot(tgt_flat, psi_flat)        # complex scalar
        c = inner / tgt_energy                          # complex scalar coefficient

        # Transmission (truth): projection of psi onto the target axis.
        truth_flat = c * tgt_flat

        # Reflection (error): orthogonal residual.
        delta_flat = psi_flat - truth_flat

        # Genuine error energy = ||delta||^2 (real, >= 0).
        error_energy = torch.real(torch.vdot(delta_flat, delta_flat))

        # Restore shapes.
        truth = truth_flat.reshape(orig_shape)
        delta = delta_flat.reshape(orig_shape)

        return truth, delta, error_energy

    # --- Diagnostic helpers (optional, used by tests / telemetry) -----------

    def alignment(self, current_wave: torch.Tensor, target_manifold: torch.Tensor) -> float:
        """
        Normalized agreement in [0, 1]: |<psi, tgt>|^2 / (||psi||^2 ||tgt||^2).
        1.0 = perfect logical agreement (parallel); 0.0 = total contradiction
        (orthogonal). This is the squared cosine / fringe visibility.
        """
        psi = current_wave.reshape(-1)
        tgt = target_manifold.reshape(-1)
        if not torch.is_complex(psi):
            psi = psi.to(torch.complex64)
        if not torch.is_complex(tgt):
            tgt = tgt.to(torch.complex64)
        num = torch.abs(torch.vdot(tgt, psi)) ** 2
        den = torch.clamp(
            torch.real(torch.vdot(psi, psi)) * torch.real(torch.vdot(tgt, tgt)),
            min=self.eps,
        )
        return float((num / den).item())
