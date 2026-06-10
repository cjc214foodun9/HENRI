"""
henri_contract.py  —  PHASE 3

Single source of truth for (a) all system dimensions and (b) the lossless
encoding of complex HRR waves into TimescaleDB.

------------------------------------------------------------------------
PROBLEMS THIS FIXES

1. SILENT HALF-WAVE TRUNCATION.
   The old write path packed only the first 2048 components:

       db_vec = np.zeros(4096, dtype=np.float32)
       db_vec[:2048] = psi_np[:2048].real
       db_vec[2048:] = psi_np[:2048].imag      # <-- psi[2048:4096] DISCARDED

   A 4096-D complex wave has 4096 real + 4096 imag = 8192 real numbers. Packing
   them into a 4096-float column means HALF the wave is thrown away on every
   write, so every rehydrated memory is corrupted. The read path in
   lookahead_prefetcher then did  vec[:half] + 1j*vec[half:]  on the already
   half-destroyed vector, compounding the loss.

2. DIMENSION DRIFT.
   gemma_dim floats between 2048 (default), the measured embedding size, and
   3840 (docs); hrr_dim is 4096 in some files; rank is 16; boundary_dim is 64.
   These live as magic numbers scattered across modules with silent
   shape-mismatch fallbacks that reinitialize learned weights. This file
   centralizes them so every module imports the SAME values, and provides a
   strict checker that FAILS LOUDLY instead of silently reinitializing.

------------------------------------------------------------------------
THE FIX

* HenriDims: one dataclass, imported everywhere, holds every dimension.
* complex_to_db / db_to_complex: lossless codec. A d-dim complex wave maps to
  a 2*d-dim float vector [real(0..d-1), imag(0..d-1)] and back with no loss.
* The DB column must therefore be vector(2*d). Use migrate_schema_sql() to get
  the corrected CREATE/ALTER statements (vector(8192) for d=4096).
* enforce_shape(): raises HenriShapeError on mismatch. No silent reinit.
"""

from dataclasses import dataclass
import numpy as np


class HenriShapeError(ValueError):
    """Raised when a tensor/array does not match the contracted dimension.
    We raise instead of silently reinitializing so that lost or mismatched
    learned weights surface immediately rather than being discarded."""
    pass


@dataclass(frozen=True)
class HenriDims:
    """The single source of truth for every dimension in the system.

    Import this and read fields off it; never hardcode a dimension again.
    `frozen=True` so it cannot be mutated at runtime.
    """
    hrr_dim: int = 4096        # complex HRR wave dimensionality (Zone B / C)
    gemma_dim: int = 2048      # generator hidden / embedding width (Zone A)
    boundary_dim: int = 64     # CFT boundary projection (Zone B validator)
    lora_rank: int = 16        # LoRA adapter rank
    num_experts: int = 16      # MoE / swarm streams

    @property
    def db_vector_dim(self) -> int:
        """Width of the pgvector column needed to store a full complex wave
        losslessly: real + imag interleaved as 2 * hrr_dim floats."""
        return 2 * self.hrr_dim

    def with_measured_gemma(self, measured: int) -> "HenriDims":
        """Return a copy with gemma_dim set to a runtime-measured value.
        Use this ONCE at boot after probing the embedding model, then pass the
        result everywhere. Because the dataclass is frozen, this returns a new
        instance rather than mutating shared state."""
        return HenriDims(
            hrr_dim=self.hrr_dim,
            gemma_dim=int(measured),
            boundary_dim=self.boundary_dim,
            lora_rank=self.lora_rank,
            num_experts=self.num_experts,
        )


# A module-level default others can import directly.
DIMS = HenriDims()


# ---------------------------------------------------------------------------
# Lossless complex <-> DB float codec
# ---------------------------------------------------------------------------

def complex_to_db(psi, hrr_dim: int) -> str:
    """
    Encode a complex wave of length hrr_dim into a pgvector literal of length
    2*hrr_dim WITHOUT loss.

    Layout:  [ Re(psi[0..d-1]) , Im(psi[0..d-1]) ]   length 2*d

    Accepts numpy arrays or torch tensors (anything with .real/.imag or
    convertible via np.asarray).
    """
    arr = np.asarray(psi)
    if arr.dtype == object:
        arr = arr.astype(np.complex128)
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128)
    arr = arr.reshape(-1)

    if arr.shape[0] != hrr_dim:
        raise HenriShapeError(
            f"complex_to_db: wave length {arr.shape[0]} != hrr_dim {hrr_dim}. "
            f"Refusing to truncate or pad silently."
        )

    db_vec = np.empty(2 * hrr_dim, dtype=np.float32)
    db_vec[:hrr_dim] = arr.real.astype(np.float32)
    db_vec[hrr_dim:] = arr.imag.astype(np.float32)
    return "[" + ",".join(map(str, db_vec.tolist())) + "]"


def db_to_complex(vec, hrr_dim: int) -> np.ndarray:
    """
    Decode a pgvector value (string literal or array) of length 2*hrr_dim back
    into a complex wave of length hrr_dim. Inverse of complex_to_db.
    """
    if isinstance(vec, str):
        cleaned = vec.strip().strip("[]")
        floats = np.fromstring(cleaned, sep=",", dtype=np.float32)
    else:
        floats = np.asarray(vec, dtype=np.float32).reshape(-1)

    if floats.shape[0] != 2 * hrr_dim:
        raise HenriShapeError(
            f"db_to_complex: stored vector length {floats.shape[0]} != "
            f"2*hrr_dim ({2 * hrr_dim}). The column is likely the old, "
            f"half-truncated vector({hrr_dim}) schema. Run migrate_schema_sql() "
            f"and re-seed; do not decode the corrupted vectors."
        )

    real = floats[:hrr_dim]
    imag = floats[hrr_dim:]
    return (real + 1j * imag).astype(np.complex64)


# ---------------------------------------------------------------------------
# Schema migration helpers
# ---------------------------------------------------------------------------

def migrate_schema_sql(dims: HenriDims = DIMS) -> str:
    """
    Returns SQL to migrate the schema to lossless complex storage. The wave
    columns become vector(2*hrr_dim). We create NEW columns and keep the old
    ones around so a migration script can re-encode rather than blindly drop.
    """
    n = dims.db_vector_dim
    return f"""
-- PHASE 3 lossless complex-wave migration (vector({n}) = 2 * hrr_dim {dims.hrr_dim})
ALTER TABLE hrr_canonical_lexicon
    ADD COLUMN IF NOT EXISTS hrr_wavefront_full vector({n});

ALTER TABLE thermodynamic_ledger
    ADD COLUMN IF NOT EXISTS hrr_trajectory_full vector({n});

-- After re-encoding existing rows into the *_full columns (see reencode_rows),
-- you may drop the legacy half-width columns:
--   ALTER TABLE hrr_canonical_lexicon DROP COLUMN hrr_wavefront;
--   ALTER TABLE thermodynamic_ledger DROP COLUMN hrr_trajectory;
""".strip()


def fresh_schema_sql(dims: HenriDims = DIMS) -> str:
    """CREATE TABLE statements for a clean install using the correct width."""
    n = dims.db_vector_dim
    return f"""
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS vector CASCADE;

CREATE TABLE IF NOT EXISTS hrr_canonical_lexicon (
    concept_hash UUID PRIMARY KEY,
    semantic_label TEXT NOT NULL,
    domain_tag TEXT,
    hrr_wavefront vector({n}) NOT NULL,   -- 2 * hrr_dim, lossless complex
    epiplexity_weight FLOAT DEFAULT 1.0,
    last_verified TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    raw_text TEXT
);

CREATE TABLE IF NOT EXISTS thermodynamic_ledger (
    timestamp TIMESTAMPTZ NOT NULL,
    inference_id UUID NOT NULL,
    langevin_heat_injected FLOAT NOT NULL,
    sagnac_error_delta FLOAT NOT NULL,
    attractor_locked BOOLEAN NOT NULL,
    hrr_trajectory vector({n})
);
""".strip()


# ---------------------------------------------------------------------------
# Strict shape enforcement (replaces silent reinit-on-mismatch)
# ---------------------------------------------------------------------------

def enforce_shape(name: str, tensor, expected_shape: tuple):
    """
    Raise HenriShapeError if tensor.shape != expected_shape. Use at every load
    site where the old code silently reinitialized on mismatch.
    """
    actual = tuple(getattr(tensor, "shape", ()))
    if actual != tuple(expected_shape):
        raise HenriShapeError(
            f"{name}: shape {actual} != expected {tuple(expected_shape)}. "
            f"Persisted learned weights do not match the current contract. "
            f"This is a hard failure (not a silent reinit) so the mismatch is "
            f"visible. Either restore the matching checkpoint or run an "
            f"explicit, intentional migration."
        )
    return tensor
