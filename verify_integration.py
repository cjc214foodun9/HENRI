"""
verify_integration.py  —  wiring for Phases 0, 1, 3

Drop-in replacement for the verification decision inside the swarm's
execute_reasoning_loop / verify path. Shows exactly how the three corrected
pieces compose. Copy the body of corrected_verify() into your
ActiveInferenceEngine.verify (or cognitive_swarm verify) and delete the old
Sagnac-tautology + wave-distance decision.

KEY CHANGE OF AUTHORITY:
    OLD: pass/fail came from boundary_validator on a wave that was identically
         the target (tautology) -> always "valid".
    NEW: pass/fail comes from GroundTruthGate (actually running the candidate).
         The interferometer error_energy is computed honestly and used ONLY as
         a *secondary* representational signal (for centroid drift / telemetry),
         never as the gate.

The reward from ground truth is what should drive the LoRA / centroid updates
(that's Phase 2, not done here, but the hook is shown).
"""

import numpy as np
import torch

from ground_truth import GroundTruthGate, GroundTruthResult
from sagnac_veto import SagnacInterferometer
from henri_contract import DIMS, complex_to_db, db_to_complex


class CorrectedVerifier:
    """
    Composes the objective gate (authority) with the honest interferometer
    (representation) and lossless DB codec (persistence).
    """

    def __init__(self, dims=DIMS):
        self.dims = dims
        self.gate = GroundTruthGate()
        self.interferometer = SagnacInterferometer()

    def corrected_verify(
        self,
        candidate_code: str,
        task_examples,          # ARC examples OR unit tests
        repl,                   # the project's universal_repl
        candidate_wave=None,    # optional: psi from the L3 router (complex, hrr_dim)
        target_manifold=None,   # optional: target axiom wave (complex, hrr_dim)
        mode: str = "arc",
    ):
        """
        Returns a dict:
            {
              "passed":        bool,    # <-- THE GATE (objective)
              "reward":        float,   # [0,1] graded score -> drives learning
              "loss":          float,   # 1 - reward
              "error_energy":  float,   # honest interferometer residual (secondary)
              "alignment":     float,   # squared cosine to target (secondary)
              "gt_detail":     dict,    # per-example breakdown
            }
        """
        # 1. OBJECTIVE GATE (Phase 0) — this is the authority.
        if mode == "arc":
            gt: GroundTruthResult = self.gate.evaluate_arc(candidate_code, task_examples, repl)
        else:
            gt = self.gate.evaluate_tests(candidate_code, task_examples, repl)

        result = {
            "passed": gt.passed,
            "reward": gt.reward,
            "loss": gt.error_energy,
            "error_energy": None,
            "alignment": None,
            "gt_detail": gt.detail,
        }

        # 2. HONEST INTERFEROMETER (Phase 1) — secondary representational signal.
        #    Computed only if a candidate wave + target are available. This no
        #    longer decides pass/fail; it informs centroid drift / telemetry.
        if candidate_wave is not None and target_manifold is not None:
            psi = self._to_complex_tensor(candidate_wave)
            tgt = self._to_complex_tensor(target_manifold)
            truth, delta, E = self.interferometer.forward(psi, tgt)
            result["error_energy"] = float(E.item())
            result["alignment"] = self.interferometer.alignment(psi, tgt)
            # truth/delta would be returned to the Phase-2 centroid-drift step;
            # centroids should anchor on the candidate's projected latent g,
            # NOT on the target.
            result["_truth_wave"] = truth
            result["_delta_wave"] = delta

        return result

    @staticmethod
    def _to_complex_tensor(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not torch.is_complex(x):
            x = x.to(torch.complex64)
        return x

    # Persistence helper (Phase 3): store a verified wave losslessly.
    def db_literal(self, psi) -> str:
        return complex_to_db(psi, self.dims.hrr_dim)
