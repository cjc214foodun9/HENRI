"""Zone C boundary-axiom loader — the epistemic north-star wire.

Reads the 11 canonical boundary axioms from the Zone C `boundary_axioms`
table (wave_payload BYTEA + semantic_index pgvector + metadata), verifies
integrity (per-block unit norm + projection cosine against the stored
semantic index), and returns real [N, num_blocks, 8] waves for the planner's
boundary_axioms channel. Fail-closed: a missing axiom, decode error, or
integrity violation raises BoundaryAxiomLoadError. The DB is the source of
truth for WHICH axioms exist and their exact wave bytes; the stored 2000-dim
projection is the verifiable digest of the full wave.

Wiring: production_arc_run.py loads this tensor when USE_ZONE_C_AXIOMS=1 and
passes it as the plan_action boundary_axioms channel (the EFE pragmatic
scoring constraint), replacing the per-step prediction residual.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from zone_c_axiom_seeder import semantic_projection

CANONICAL_AXIOM_IDS = [
    "urn:henri:axiom:qfhrr_unitary_norm:v1",
    "urn:henri:axiom:clifford_cl30_causal:v1",
    "urn:henri:axiom:sagnac_homodyne_zero_stress:v1",
    "urn:henri:axiom:stiefel_manifold_orthogonality:v1",
    "urn:henri:axiom:spatial_grid_conservation:v1",
    "urn:henri:axiom:spelke_affine_translation:v1",
    "urn:henri:axiom:spelke_affine_rotation:v1",
    "urn:henri:axiom:spelke_reflection_symmetry:v1",
    "urn:henri:axiom:spelke_color_rebinding:v1",
    "urn:henri:axiom:spelke_gravity_drop:v1",
    "urn:henri:axiom:spelke_contour_fill:v1",
]

AXIOM_NORM_TOL = 1e-4   # per-block unit norm tolerance (| ||w_k|| - 1 |)
AXIOM_PROJ_COS_MIN = 0.999  # stored 2000-dim projection must match the wave


class BoundaryAxiomLoadError(RuntimeError):
    """Fail-closed: the axioms cannot be loaded or verified from Zone C."""


def resolve_dsn(env_file: Optional[str] = None) -> str:
    """Resolve the Zone C DSN from an env file (never prints it)."""
    candidates = [env_file]
    if env_file is None:
        candidates = [os.environ.get("ZONE_C_AXIOM_ENV_FILE"),
                      os.environ.get("ZONE_C_PROD_DSN")]
    for src in candidates:
        if src and os.path.exists(src):
            for line in open(src, "r", encoding="utf-8"):
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip().endswith("DSN"):
                    return v.strip().strip('"').strip("'")
    env = os.environ.get("ZONE_C_PROD_DSN")
    if env:
        return env
    raise BoundaryAxiomLoadError("ZONE_C_PROD_DSN unavailable (no approved secret channel)")


def decode_wave_payload(payload: bytes, num_blocks: int) -> torch.Tensor:
    """Decode the stored float32 wave_payload BYTEA into [num_blocks, 8]."""
    arr = np.frombuffer(payload, dtype=np.float32)
    if arr.size != num_blocks * 8:
        raise BoundaryAxiomLoadError(
            f"wave_payload size {arr.size} != {num_blocks * 8}")
    wave = torch.from_numpy(arr.reshape(num_blocks, 8).copy()).to(torch.float32)
    return wave


def verify_wave_integrity(
    wave: torch.Tensor, stored_semantic_index: torch.Tensor,
) -> dict:
    """Verify per-block unit norm and the stored projection cosine. Returns
    the integrity summary; raises on violation."""
    norms = torch.norm(wave, p=2, dim=-1)
    max_norm_err = float((norms - 1.0).abs().max())
    if max_norm_err > AXIOM_NORM_TOL:
        raise BoundaryAxiomLoadError(f"per-block norm error {max_norm_err:.6f} > {AXIOM_NORM_TOL}")
    proj = semantic_projection(wave.view(-1))
    cos = float(torch.dot(
        torch.nn.functional.normalize(proj, p=2, dim=0),
        torch.nn.functional.normalize(stored_semantic_index.to(torch.float32).view(-1), p=2, dim=0)).item())
    if cos < AXIOM_PROJ_COS_MIN:
        raise BoundaryAxiomLoadError(f"projection cosine {cos:.6f} < {AXIOM_PROJ_COS_MIN}")
    return {"max_norm_err": max_norm_err, "proj_cos": cos}


def load_boundary_axioms(
    dsn: Optional[str] = None,
    env_file: Optional[str] = None,
    num_blocks: int = 8192,
) -> tuple[torch.Tensor, list[dict]]:
    """Load the 11 canonical axioms from Zone C into [11, num_blocks, 8]."""
    if dsn is None:
        dsn = resolve_dsn(env_file)
    import psycopg

    try:
        conn = psycopg.connect(dsn, connect_timeout=15)
    except Exception as exc:
        raise BoundaryAxiomLoadError(f"connect failed: {type(exc).__name__}: {exc}")
    rows = []
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT axiom_id, axiom_kind, wave_payload, semantic_index, num_blocks "
                "FROM boundary_axioms ORDER BY axiom_id")
            rows = cur.fetchall()
    finally:
        conn.close()

    by_id = {r[0]: r for r in rows}
    missing = [aid for aid in CANONICAL_AXIOM_IDS if aid not in by_id]
    if missing:
        raise BoundaryAxiomLoadError(f"missing axioms: {missing}")

    waves = []
    summaries = []
    for aid in CANONICAL_AXIOM_IDS:
        _, kind, payload, sem_index, stored_blocks = by_id[aid]
        if stored_blocks != num_blocks:
            raise BoundaryAxiomLoadError(
                f"{aid}: stored num_blocks {stored_blocks} != {num_blocks}")
        wave = decode_wave_payload(bytes(payload), num_blocks)
        sem_t = torch.as_tensor(
            np.asarray(sem_index, dtype=np.float32)) if not torch.is_tensor(sem_index) else sem_index
        summ = verify_wave_integrity(wave, sem_t)
        summaries.append({"axiom_id": aid, "axiom_kind": kind, **summ})
        waves.append(wave)

    return torch.stack(waves), summaries
