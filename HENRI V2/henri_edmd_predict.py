"""
EDMD Latent Composition Predictor (Carrier G1) - default-OFF.

Fits the RecursiveDualEDMD (R-EDMD) Koopman operator online from public
demonstration pairs (X_i, Y_i) encoded as [num_blocks, 8] real phasor waves,
then predicts the solution wave for an unseen test X wave from:

    hat_PSI_Y = T(PSI_X_test, G)

where T is the learned online operator and G is the goal-prototype action
wave (normalized mean of TRAIN Y waves only; no hold-out leakage).

Pre-registered gate (experiments/verification/g1_edmd_predict_prereg.md):
    EDMD_PREDICT_OK iff, on the LEAVE-ONE-OUT demo pair (X_h, Y_h):
        cos(hat_PSI_Y_h, PSI_Y_h) > 0.30
        AND cos(hat_PSI_Y_h, PSI_Y_h) > cos(PSI_X_h, PSI_Y_h) + 0.10
    else EDMD_PREDICT_UNDERFIT (fail-closed; caller falls through to lower
    goal layers; telemetry records the BLOCKED status).

Zero pretraining. No writes to the repository. Module is CPU-testable and
device-agnostic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from recursive_dual_edmd import RecursiveDualEDMD

STATUS_OK = "EDMD_PREDICT_OK"
STATUS_UNDERFIT = "EDMD_PREDICT_UNDERFIT"
STATUS_NO_DEMOS = "BLOCKED_NO_DEMOS"
STATUS_EMPTY = "BLOCKED_EMPTY_DEMOS"
STATUS_IMPORT = "BLOCKED_IMPORT_FAILED"

# Pre-registered thresholds (see prereg file, Gate G1.2).
RECOVERY_THRESHOLD = 0.30
IDENTITY_MARGIN = 0.10

# Default rank/forgetting for the online operator.
R_RANK = 16
LAMBDA_FORGET = 0.98
REGULARIZATION = 1e-4


@dataclass
class EDMDCompositionResult:
    """Deterministic outcome record for the composition gate."""

    status: str
    reason: str = ""
    held_out_cos: Optional[float] = None
    identity_cos: Optional[float] = None
    improvement: Optional[float] = None
    self_sim: Optional[float] = None
    predicted_wave: Optional[torch.Tensor] = None
    pairs_digest: str = ""
    telemetry: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "held_out_cos": self.held_out_cos,
            "identity_cos": self.identity_cos,
            "improvement": self.improvement,
            "self_sim": self.self_sim,
            "pairs_digest": self.pairs_digest,
            **self.telemetry,
        }


def _unit_rows(t: torch.Tensor) -> torch.Tensor:
    """Per-row normalize [..., nb, 8] phasor rows (scale-invariant blocks)."""
    return t / (t.norm(dim=-1, keepdim=True) + 1e-12)


def _flat_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    fa = F.normalize(a.reshape(-1).float(), p=2, dim=0)
    fb = F.normalize(b.reshape(-1).float(), p=2, dim=0)
    return float(torch.dot(fa, fb).item())


def _pairs_digest(x_waves: torch.Tensor, y_waves: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in (x_waves, y_waves):
        h.update(t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def fit_and_predict(
    x_waves: torch.Tensor,
    y_waves: torch.Tensor,
    test_x_wave: torch.Tensor,
    *,
    r_rank: int = R_RANK,
    lambda_forget: float = LAMBDA_FORGET,
    regularization: float = REGULARIZATION,
    recovery_threshold: float = RECOVERY_THRESHOLD,
    identity_margin: float = IDENTITY_MARGIN,
    hold_out_index: int = -1,
) -> EDMDCompositionResult:
    """Fit R-EDMD online on train pairs, gate on a leave-one-out pair.

    x_waves, y_waves: [m, num_blocks, 8] real phasor waves (per-row unit
    normalized internally). test_x_wave: [num_blocks, 8].

    Returns a typed result; never raises for data-path faults, never
    fabricates a prediction when the gate fails (predicted_wave is None).
    """
    if x_waves.shape[0] != y_waves.shape[0]:
        return EDMDCompositionResult(
            STATUS_IMPORT, f"pair count mismatch: x={x_waves.shape[0]} y={y_waves.shape[0]}")
    m = x_waves.shape[0]

    x_waves = _unit_rows(x_waves)
    y_waves = _unit_rows(y_waves)

    if m == 0:
        return EDMDCompositionResult(STATUS_NO_DEMOS, "no demonstration pairs supplied")
    if m < 2:
        return EDMDCompositionResult(
            STATUS_EMPTY, "no training pairs after leave-one-out hold-out")

    if hold_out_index < 0:
        hold_out_index = m - 1
    hold_out_index = min(hold_out_index, m - 1)
    train_idx = [i for i in range(m) if i != hold_out_index]
    train_x = x_waves[train_idx]
    train_y = y_waves[train_idx]
    hold_x, hold_y = x_waves[hold_out_index], y_waves[hold_out_index]

    # Goal-prototype action from TRAIN outputs only (no hold-out leakage).
    goal_proto = F.normalize(
        train_y.reshape(m - 1, -1).mean(dim=0), p=2, dim=0).reshape(hold_x.shape)

    d_model = int(x_waves.shape[1] * x_waves.shape[2])
    operator = RecursiveDualEDMD(
        d_model=d_model, r_rank=r_rank, lambda_forget=lambda_forget,
        regularization=regularization).to(x_waves.device)
    with torch.no_grad():
        for x_i, y_i in zip(train_x, train_y):
            operator.update_online_step(x_i, goal_proto, y_i)

        # Train self-prediction (diagnostic; not part of the gate).
        self_sims = []
        for x_i, y_i in zip(train_x, train_y):
            self_sims.append(_flat_cos(operator(x_i, goal_proto), y_i))
        self_sim = float(sum(self_sims) / len(self_sims)) if self_sims else 0.0

        # Gate on the leave-one-out pair.
        pred_h = operator(hold_x, goal_proto)
        held_out_cos = _flat_cos(pred_h, hold_y)
        identity_cos = _flat_cos(hold_x, hold_y)
        improvement = held_out_cos - identity_cos

    result = EDMDCompositionResult(
        status=STATUS_UNDERFIT,
        reason="",
        held_out_cos=round(held_out_cos, 6),
        identity_cos=round(identity_cos, 6),
        improvement=round(improvement, 6),
        self_sim=round(self_sim, 6),
        predicted_wave=None,
        pairs_digest=_pairs_digest(x_waves, y_waves),
        telemetry={
            "r_rank": r_rank,
            "lambda_forget": lambda_forget,
            "recovery_threshold": recovery_threshold,
            "identity_margin": identity_margin,
            "demo_pair_count": m,
            "train_pair_count": m - 1,
        },
    )
    if held_out_cos > recovery_threshold and held_out_cos > identity_cos + identity_margin:
        result.status = STATUS_OK
        result.reason = "pre-registered recovery + margin criteria satisfied"
        result.predicted_wave = operator(test_x_wave, goal_proto)
    else:
        result.reason = (
            f"held_out_cos={held_out_cos:.4f} threshold={recovery_threshold:.2f} "
            f"identified_cos={identity_cos:.4f} margin={improvement:.4f} "
            f"< required {identity_margin:.2f}"
        )
    return result


def predict_solution_grids(
    demo_pairs: Sequence[Tuple[Any, Any]],
    tokenizer: Any,
    test_grid: Any,
    device: str = "cpu",
    **kwargs,
) -> EDMDCompositionResult:
    """Production wrapper: encode grids via the live tokenizer, then gate.

    demo_pairs: [(input_grid, output_grid), ...] where each grid is a
    list-of-lists or np.ndarray. tokenizer: object with
    encode_spatial_grid(grid) -> [1, num_blocks, 8]. test_grid: same type
    as a demo input grid (the unseen episode observation frame).
    """
    encode = getattr(tokenizer, "encode_spatial_grid", None)
    if encode is None:
        return EDMDCompositionResult(STATUS_IMPORT, "tokenizer lacks encode_spatial_grid")
    if not demo_pairs:
        return EDMDCompositionResult(STATUS_NO_DEMOS, "no demonstration pairs supplied")

    xs, ys = [], []
    try:
        for x, y in demo_pairs:
            _x = x.tolist() if hasattr(x, "tolist") else x
            _y = y.tolist() if hasattr(y, "tolist") else y
            wx = encode(_x).squeeze(0).float().to(device)
            wy = encode(_y).squeeze(0).float().to(device)
            if wx.dim() != 2 or wy.dim() != 2:
                return EDMDCompositionResult(
                    STATUS_IMPORT,
                    f"encode produced shape {tuple(wx.shape)}/{tuple(wy.shape)}")
            xs.append(wx)
            ys.append(wy)
        tx = encode(test_grid).squeeze(0).float().to(device)
    except Exception as exc:  # data-path fault -> typed fail-closed
        return EDMDCompositionResult(STATUS_IMPORT, f"encode failure: {type(exc).__name__}")

    return fit_and_predict(torch.stack(xs), torch.stack(ys), tx, **kwargs)
