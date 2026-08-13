"""Phase 7.8 P0-A1: production encoder-basis configuration (G1 ACCEPTED).

Promotes the verified incommensurate x/y ramp + CC-OS background-masking
variant (G1 ACCEPTED at D=65,536: max cross-cosine < 0.05, LUT coordinate
recovery 100%) to the DEFAULT production encoder basis. The legacy collinear
basis remains available byte-for-byte via HENRI_ARC_SPATIAL_BASIS=default
with HENRI_ARC_BG_MASK=0 (contract-tested byte identity).

The resolver is a pure function of the environment so contract tests can pin
the production defaults without importing the full runner.
"""
import os
from typing import Tuple

SPATIAL_BASIS_KINDS = ("default", "incommensurate", "random")
DEFAULT_SPATIAL_BASIS = "incommensurate"  # Phase 7.8 P0-A1
DEFAULT_BG_MASK = True                    # Phase 7.8 P0-A1


def resolve_spatial_basis() -> Tuple[str, bool]:
    """Return (spatial_basis_kind, bg_mask) for the production encoder.

    Phase 7.8 P0-A1: production default is the G1-ACCEPTED incommensurate
    basis with CC-OS background masking. Legacy opt-in (byte-identical to
    the pre-7.8 production path) requires BOTH explicit values:
      HENRI_ARC_SPATIAL_BASIS=default HENRI_ARC_BG_MASK=0
    """
    kind = os.environ.get(
        "HENRI_ARC_SPATIAL_BASIS", DEFAULT_SPATIAL_BASIS
    ).strip()
    if kind not in SPATIAL_BASIS_KINDS:
        raise ValueError(
            "HENRI_ARC_SPATIAL_BASIS must be default|incommensurate|random, "
            f"got {kind!r}"
        )
    bg_mask = os.environ.get("HENRI_ARC_BG_MASK", "1") == "1"
    return kind, bg_mask
