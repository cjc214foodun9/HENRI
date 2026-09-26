"""Preconditioned gradient-alignment reward kernel.

Source method: "Self-Play Pretraining with Zero Data" (arXiv:2609.30063).
Provenance: the paper EXISTS (`orx --paper 2609.30063` -> status=pass). The exact
reward formula below is taken from the HENRI synthesis document
`HENRI-ARCH-2026-SELFPLAY-DREAMING-V1`, whose provenance is HYPOTHESIS until the
primary text is read. Treat the shape of this kernel as an implementation of a
proposed mechanism, not as a reproduction of a measured result.

Definition
----------
    r_i = | < grad_theta L(y_i; theta_e),  P_e @ delta_theta_e > |

    P_e          = diag( eta_lr / (sqrt(v_e) + eps) )        AdamW diagonal operator
    delta_theta  = theta_{p(e)} - theta_e                    lookback displacement
    p(e)         = floor(e / 2)                              growing lookback

Properties this kernel is required to satisfy (see tests):
  * a mastered input (zero gradient) gives zero reward;
  * a gradient orthogonal to P @ delta_theta gives zero reward;
  * a gradient parallel to P @ delta_theta gives the maximal reward;
  * the preconditioner changes the candidate RANKING under a non-uniform v;
    under a uniform v it may only rescale all candidates.

The kernel fails closed. It raises on any input that would make the reward
undefined or non-discriminative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

__all__ = [
    "AlignmentRewardError",
    "AlignmentRewardConfig",
    "lookback_index",
    "parameter_displacement",
    "adamw_preconditioner",
    "alignment_reward",
]


class AlignmentRewardError(ValueError):
    """Raised when the reward kernel receives invalid input. Fail closed."""


@dataclass(frozen=True)
class AlignmentRewardConfig:
    """Static settings for the reward kernel.

    Attributes:
        lr: AdamW learning rate (eta_lr). Must be > 0.
        eps: AdamW numerical floor (epsilon). Must be > 0.
    """

    lr: float = 1e-4
    eps: float = 1e-8

    def validate(self) -> "AlignmentRewardConfig":
        if not isinstance(self.lr, float) or self.lr <= 0.0:
            raise AlignmentRewardError(f"lr must be a positive float, got {self.lr!r}")
        if not isinstance(self.eps, float) or self.eps <= 0.0:
            raise AlignmentRewardError(f"eps must be a positive float, got {self.eps!r}")
        return self


# --------------------------------------------------------------------------------------
# input validation
# --------------------------------------------------------------------------------------


def _require_tensor(name: str, value: object) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise AlignmentRewardError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    if not torch.is_floating_point(value):
        raise AlignmentRewardError(f"{name} must be a floating-point tensor, got dtype {value.dtype}")
    if not torch.isfinite(value).all():
        raise AlignmentRewardError(f"{name} contains non-finite values")
    return value


def _require_same_shape(names: Tuple[str, ...], tensors: Tuple[torch.Tensor, ...]) -> None:
    shapes = {t.shape for t in tensors}
    if len(shapes) != 1:
        raise AlignmentRewardError(
            "shape mismatch: " + ", ".join(f"{n}={tuple(t.shape)}" for n, t in zip(names, tensors))
        )


def _require_rank12(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() not in (1, 2):
        raise AlignmentRewardError(f"{name} must have rank 1 [P] or rank 2 [B, P], got rank {tensor.dim()}")


# --------------------------------------------------------------------------------------
# components
# --------------------------------------------------------------------------------------


def lookback_index(round_e: int) -> int:
    """Return the growing lookback checkpoint index p(e) = floor(e / 2)."""
    if not isinstance(round_e, int) or isinstance(round_e, bool):
        raise AlignmentRewardError(f"round_e must be an int, got {type(round_e).__name__}")
    if round_e < 0:
        raise AlignmentRewardError(f"round_e must be >= 0, got {round_e}")
    return round_e // 2


def parameter_displacement(
    theta_current: torch.Tensor,
    theta_lookback: torch.Tensor,
) -> torch.Tensor:
    """Return delta_theta = theta_lookback - theta_current (the displacement direction)."""
    cur = _require_tensor("theta_current", theta_current)
    look = _require_tensor("theta_lookback", theta_lookback)
    _require_same_shape(("theta_current", "theta_lookback"), (cur, look))
    return look - cur


def adamw_preconditioner(
    exp_avg_sq: torch.Tensor,
    lr: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the diagonal AdamW step operator P = lr / (sqrt(v) + eps).

    The second-moment estimate is biased, so a degree of caution is applied and
    the square root is taken. All three inputs are validated.
    """
    cfg = AlignmentRewardConfig(lr=float(lr), eps=float(eps)).validate()
    v = _require_tensor("exp_avg_sq", exp_avg_sq)
    if (v < 0).any():
        raise AlignmentRewardError("exp_avg_sq (v) must be non-negative")
    return cfg.lr / (torch.sqrt(v) + cfg.eps)


# --------------------------------------------------------------------------------------
# the reward
# --------------------------------------------------------------------------------------


def alignment_reward(
    grad: torch.Tensor,
    displacement: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    lr: float = 1e-4,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the preconditioned gradient-alignment reward.

    Args:
        grad: gradient of the learner loss on the generated sample, shape [P] or [B, P].
        displacement: lookback parameter displacement delta_theta, same shape as grad.
        exp_avg_sq: AdamW second-moment estimate v_e, same shape as grad.
        lr: AdamW learning rate.
        eps: AdamW numerical floor.

    Returns:
        A scalar tensor for rank-1 input; a [B] tensor for rank-2 input.

    The reward is the absolute inner product between the gradient and the
    preconditioned displacement. It is zero when the sample is already mastered
    (gradient -> 0) and zero when the gradient is orthogonal to the coherent
    displacement direction.
    """
    cfg = AlignmentRewardConfig(lr=float(lr), eps=float(eps)).validate()
    g = _require_tensor("grad", grad)
    d = _require_tensor("displacement", displacement)
    v = _require_tensor("exp_avg_sq", exp_avg_sq)
    _require_same_shape(("grad", "displacement", "exp_avg_sq"), (g, d, v))
    _require_rank12("grad", g)
    for name, tensor in (("grad", g), ("displacement", d), ("exp_avg_sq", v)):
        _require_rank12(name, tensor)

    if (v < 0).any():
        raise AlignmentRewardError("exp_avg_sq (v) must be non-negative")

    p = cfg.lr / (torch.sqrt(v) + cfg.eps)
    scaled_delta = p * d

    if g.dim() == 1:
        return torch.abs(torch.sum(g * scaled_delta))

    # [B, P] -> [B]
    return torch.abs(torch.sum(g * scaled_delta, dim=-1))
