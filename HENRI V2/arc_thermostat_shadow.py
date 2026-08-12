"""Phase 7.5 CONN Module B: read-only AdaptiveViscoelasticThermostat shadow.

Corpus grounding (bank ca4bb787, convo 3179135d): the BLIND thermostat
(isotropic heat shock) destroys valid low-frequency structures at D=65,536 —
a labeled falsification. Module B is therefore a DIAGNOSTIC SHADOW ONLY:
it instantiates the real production AdaptiveViscoelasticThermostat (its
constructor is scalar-only: no weight matrices, no VRAM allocation) and
evaluates the production friction / effective-LR math on the live per-step
signals (sagnac_delta, lambda_active). It NEVER mutates weights, NEVER
influences policy, and keeps no memory inside the Zone B planning loop.

FAIL-CLOSED typing: any anomaly returns UNAVAILABLE with no crash. The
shadow emits telemetry only; it cannot change the action path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

THERMO_OK = "THERMO_SHADOW_OK"
THERMO_UNAVAILABLE = "THERMO_SHADOW_UNAVAILABLE"


def evaluate_thermostat_shadow(
    thermostat,
    lambda_active: Optional[float],
    sagnac_delta: Optional[float],
) -> Tuple[Dict[str, Any], str]:
    """Compute the read-only thermostat signals from live per-step values.

    Args:
        thermostat: AdaptiveViscoelasticThermostat instance (or None).
        lambda_active: constraint stiffness from the chosen candidate.
        sagnac_delta: Sagnac phase delta from the relaxation loop.

    Returns:
        (shadow_dict, status). shadow_dict carries the production friction
        and effective-LR math when available; status THERMO_OK on a clean
        evaluation, THERMO_UNAVAILABLE on any anomaly (None inputs,
        exception). Unavailable NEVER mutates anything and never crashes.
    """
    if thermostat is None or lambda_active is None or sagnac_delta is None:
        return {"status": THERMO_UNAVAILABLE}, THERMO_UNAVAILABLE
    try:
        la = float(lambda_active)
        sd = float(sagnac_delta)
        friction = thermostat.compute_anisotropic_friction(la, sd)
        # Production effective-LR math (step_viscoelastic_creep, isotropic
        # branch): effective_lr = (base_lr / friction) * (1 + sagnac_delta).
        effective_lr = (thermostat.base_lr / friction) * (1.0 + sd)
        shadow = {
            "status": THERMO_OK,
            "lambda_active": round(la, 6),
            "sagnac_delta": round(sd, 6),
            "langevin_friction": round(float(friction), 6),
            "effective_lr": round(float(effective_lr), 8),
            "read_only": True,
        }
        return shadow, THERMO_OK
    except Exception:
        return {"status": THERMO_UNAVAILABLE}, THERMO_UNAVAILABLE
