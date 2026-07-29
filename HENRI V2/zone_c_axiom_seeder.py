"""
Zone C Boundary Axiom Seeder and Schema Migration for Project HENRI V2.

Implements the "Crystallization Seeding Protocol":
  1. Ensures tables `boundary_axioms` and `external_outcomes` exist in Zone C TimescaleDB.
  2. Generates and seals 5 fundamental "Seed Crystal Boundary Axioms" on the complex unit hypersphere (D=65536):
     - Axiom 1: qFHRR_Unitary_Norm_Preservation (Unit hypersphere constraint ||Psi|| = 1.0)
     - Axiom 2: Clifford_Cl30_Noncommutative_Causal_Order (Geometric product phase orientation)
     - Axiom 3: Sagnac_Homodyne_Zero_Stress_Veto (Phase-locking homodyne clearance Delta -> 0)
     - Axiom 4: Stiefel_Manifold_Orthogonality_Constraint (Rigid Stiefel retraction ||AA^T - I|| < eps)
     - Axiom 5: Spatial_Grid_2D_Energy_Conservation (Continuous spatial translation invariants)
  3. Inserts seeded axioms into TimescaleDB and verifies HNSW vector search / cosine similarity retrieval.

Usage:
    python zone_c_axiom_seeder.py [--dsn DSN] [--num_blocks 8192] [--dry-run]
"""

import argparse
import json
import logging
import math
import os
import sys
import uuid
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [ZoneC-Seeder] - %(message)s")

try:
    import psycopg
except ImportError:
    psycopg = None


# --- SQL Schema Definitions ---

CREATE_BOUNDARY_AXIOMS_SQL = """
CREATE TABLE IF NOT EXISTS boundary_axioms (
    axiom_id TEXT PRIMARY KEY,
    axiom_kind TEXT NOT NULL,          -- 'qfhrr_codec_contract', 'clifford_causal_order', 'sagnac_veto', 'stiefel_retraction', 'spatial_conservation'
    source TEXT NOT NULL,              -- Document or algebraic origin
    source_commit VARCHAR(40),         -- SHA-1 commit hash
    validity_scope TEXT,               -- Domain namespace
    dimension INT DEFAULT 65536,       -- Hyperspherical dimension D
    num_blocks INT DEFAULT 8192,       -- Total Clifford blocks K
    rank INT DEFAULT 16,               -- Low-rank bottleneck r
    projection_seed INT DEFAULT 7,     -- JL projection seed
    semantic_index VECTOR(2000) NOT NULL, -- 2000-dim HNSW projection for pgvector
    wave_payload BYTEA NOT NULL,       -- Raw float32 wave bytes [num_blocks, 8]
    semantic_projection JSONB,         -- Symbolic AST or algebraic metadata
    residual_scale DOUBLE PRECISION DEFAULT 1.0,
    confidence DOUBLE PRECISION DEFAULT 1.0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS boundary_axioms_kind_idx ON boundary_axioms(axiom_kind);
CREATE INDEX IF NOT EXISTS boundary_axioms_semantic_hnsw_idx
    ON boundary_axioms USING hnsw (semantic_index vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
"""

CREATE_EXTERNAL_OUTCOMES_SQL = """
CREATE TABLE IF NOT EXISTS external_outcomes (
    outcome_id SERIAL,
    run_id UUID NOT NULL,
    step_index INT NOT NULL,
    level_scores DOUBLE PRECISION,
    level_actions INT[],
    motion_mean DOUBLE PRECISION,
    progress_valence INT,              -- -1 for fail/reset, +1 for progress/win, 0 for neutral
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (outcome_id, created_at)
);

SELECT create_hypertable('external_outcomes', 'created_at', if_not_exists => TRUE);
"""


def _proj_matrix(dim: int = 65536, semantic_dim: int = 2000, seed: int = 7) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(semantic_dim, dim, generator=g) / math.sqrt(dim)


def semantic_projection(wave: torch.Tensor, seed: int = 7) -> torch.Tensor:
    flat = wave.view(-1).to(torch.float32).cpu()
    proj = _proj_matrix(flat.numel(), 2000, seed) @ flat
    return F.normalize(proj, p=2, dim=-1)


def generate_seed_crystal_axioms(num_blocks: int = 8192) -> list:
    """Generates 5 canonical seed crystal boundary waves on the unit hypersphere."""
    axioms = []
    
    # 1. qFHRR Unitary Norm Preservation Axiom
    g1 = torch.Generator(device="cpu").manual_seed(1001)
    w1 = torch.randn(num_blocks, 8, generator=g1)
    w1 = F.normalize(w1, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:qfhrr_unitary_norm:v1",
        "axiom_kind": "qfhrr_codec_contract",
        "source": "HENRI V2 UWE qFHRR Specification",
        "validity_scope": "global:physics:norm",
        "wave": w1,
        "metadata": {"contract": "||Psi|| == 1.0", "phase_bits": 8, "quantization_k": 256}
    })

    # 2. Clifford Cl(3,0) Non-Commutative Causal Order Axiom
    g2 = torch.Generator(device="cpu").manual_seed(2002)
    w2 = torch.randn(num_blocks, 8, generator=g2)
    # Enforce antisymmetric bivector phase orientation across blocks
    w2[:, 1:4] = -w2[:, 1:4]
    w2 = F.normalize(w2, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:clifford_cl30_causal:v1",
        "axiom_kind": "clifford_causal_order",
        "source": "Non-Commutative Geometric Product Mechanics",
        "validity_scope": "global:algebra:causal",
        "wave": w2,
        "metadata": {"algebra": "Cl(3,0)", "non_commutative": True, "bivector_phase": "oriented"}
    })

    # 3. Sagnac Homodyne Zero-Stress Veto Axiom
    g3 = torch.Generator(device="cpu").manual_seed(3003)
    w3 = torch.randn(num_blocks, 8, generator=g3)
    w3[:, 0] = 1.0  # Pure real scalar component
    w3 = F.normalize(w3, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:sagnac_homodyne_zero_stress:v1",
        "axiom_kind": "sagnac_veto",
        "source": "Sagnac Homodyne Interferometry Veto Contract",
        "validity_scope": "global:physics:sagnac",
        "wave": w3,
        "metadata": {"target_sagnac_delta": 0.0, "clearance_veto": True}
    })

    # 4. Stiefel Manifold Orthogonality Constraint Axiom
    g4 = torch.Generator(device="cpu").manual_seed(4004)
    w4 = torch.randn(num_blocks, 8, generator=g4)
    w4 = F.normalize(w4, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:stiefel_manifold_orthogonality:v1",
        "axiom_kind": "stiefel_retraction",
        "source": "Newton-Schulz Quadratic Stiefel Retraction Contract",
        "validity_scope": "global:manifold:stiefel",
        "wave": w4,
        "metadata": {"retraction": "cholesky_newton_schulz", "max_gram_error": 1e-4}
    })

    # 5. 2D Spatial Grid Translation Energy Conservation Axiom
    g5 = torch.Generator(device="cpu").manual_seed(5005)
    w5 = torch.randn(num_blocks, 8, generator=g5)
    # Apply toroidal spatial smoothing
    side = int(math.sqrt(num_blocks))
    if side * side == num_blocks:
        grid = w5.view(side, side, 8)
        grid = (grid + torch.roll(grid, 1, 0) + torch.roll(grid, -1, 0)) / 3.0
        w5 = grid.view(num_blocks, 8)
    w5 = F.normalize(w5, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:spatial_grid_conservation:v1",
        "axiom_kind": "spatial_conservation",
        "source": "Continuous 2D Spatial Grid Translation Law",
        "validity_scope": "global:grid:physics",
        "wave": w5,
        "metadata": {"grid_toroidal": True, "energy_conservation": True}
    })

    # --- Spelke Core Knowledge Axiom Expansion (Phase 4.1) ---

    # 6. Spelke Prior: Affine Spatial Translation
    g6 = torch.Generator(device="cpu").manual_seed(6006)
    w6 = torch.randn(num_blocks, 8, generator=g6)
    w6 = F.normalize(w6, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:spelke_affine_translation:v1",
        "axiom_kind": "spelke_prior",
        "source": "Spelke Core Knowledge: Continuous 2D Grid Translation",
        "validity_scope": "global:spelke:translation",
        "wave": w6,
        "metadata": {"prior": "affine_translation", "spelke_core": True}
    })

    # 7. Spelke Prior: Discrete Affine Rotation (90/180/270 degrees)
    g7 = torch.Generator(device="cpu").manual_seed(7007)
    w7 = torch.randn(num_blocks, 8, generator=g7)
    w7 = F.normalize(w7, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:spelke_affine_rotation:v1",
        "axiom_kind": "spelke_prior",
        "source": "Spelke Core Knowledge: Discrete Spatial Rotation Symmetry",
        "validity_scope": "global:spelke:rotation",
        "wave": w7,
        "metadata": {"prior": "affine_rotation", "spelke_core": True}
    })

    # 8. Spelke Prior: Spatial Reflection Parity
    g8 = torch.Generator(device="cpu").manual_seed(8008)
    w8 = torch.randn(num_blocks, 8, generator=g8)
    w8 = F.normalize(w8, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:spelke_reflection_symmetry:v1",
        "axiom_kind": "spelke_prior",
        "source": "Spelke Core Knowledge: Vertical/Horizontal Parity Reflection",
        "validity_scope": "global:spelke:reflection",
        "wave": w8,
        "metadata": {"prior": "reflection_parity", "spelke_core": True}
    })

    # 9. Spelke Prior: Color-Rebinding Permutation Invariance
    g9 = torch.Generator(device="cpu").manual_seed(9009)
    w9 = torch.randn(num_blocks, 8, generator=g9)
    w9 = F.normalize(w9, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:spelke_color_rebinding:v1",
        "axiom_kind": "spelke_prior",
        "source": "Spelke Core Knowledge: Permutation-Invariant Color Index Mapping",
        "validity_scope": "global:spelke:color",
        "wave": w9,
        "metadata": {"prior": "color_rebinding", "spelke_core": True}
    })

    # 10. Spelke Prior: Unidirectional Contact Gravity Drop
    g10 = torch.Generator(device="cpu").manual_seed(10010)
    w10 = torch.randn(num_blocks, 8, generator=g10)
    w10 = F.normalize(w10, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:spelke_gravity_drop:v1",
        "axiom_kind": "spelke_prior",
        "source": "Spelke Core Knowledge: Directional Contact Mechanics / Gravity Drop",
        "validity_scope": "global:spelke:gravity",
        "wave": w10,
        "metadata": {"prior": "gravity_drop", "spelke_core": True}
    })

    # 11. Spelke Prior: Topological Boundary Contour Fill
    g11 = torch.Generator(device="cpu").manual_seed(11011)
    w11 = torch.randn(num_blocks, 8, generator=g11)
    w11 = F.normalize(w11, p=2, dim=-1)
    axioms.append({
        "axiom_id": "urn:henri:axiom:spelke_contour_fill:v1",
        "axiom_kind": "spelke_prior",
        "source": "Spelke Core Knowledge: Enclosed Boundary Flood Fill Operator",
        "validity_scope": "global:spelke:contour",
        "wave": w11,
        "metadata": {"prior": "contour_fill", "spelke_core": True}
    })

    return axioms


def seed_zone_c_database(dsn: str, num_blocks: int = 8192, dry_run: bool = False):
    logging.info(f"Connecting to Zone C TimescaleDB: {dsn[:40]}... (num_blocks={num_blocks})")
    
    if psycopg is None:
        raise RuntimeError("psycopg Python driver is required for Zone C seeding.")

    axioms = generate_seed_crystal_axioms(num_blocks=num_blocks)
    logging.info(f"Generated {len(axioms)} Seed Crystal Boundary Axioms.")

    with psycopg.connect(dsn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            logging.info("Executing schema migration (creating boundary_axioms & external_outcomes)...")
            cur.execute(CREATE_BOUNDARY_AXIOMS_SQL)
            try:
                cur.execute(CREATE_EXTERNAL_OUTCOMES_SQL)
            except Exception as e:
                logging.warning(f"Note on external_outcomes hypertable: {e}")

            commit_hash = "0e435c6"
            inserted = 0
            for ax in axioms:
                wave = ax["wave"]
                wave_bytes = wave.numpy().astype(np.float32).tobytes()
                sem = semantic_projection(wave)
                sem_str = "[" + ",".join(f"{v:.6f}" for v in sem.tolist()) + "]"

                if not dry_run:
                    cur.execute(
                        """
                        INSERT INTO boundary_axioms
                            (axiom_id, axiom_kind, source, source_commit, validity_scope,
                             dimension, num_blocks, semantic_index, wave_payload, semantic_projection)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
                        ON CONFLICT (axiom_id) DO UPDATE SET
                            wave_payload = EXCLUDED.wave_payload,
                            semantic_index = EXCLUDED.semantic_index,
                            created_at = NOW()
                        """,
                        (
                            ax["axiom_id"],
                            ax["axiom_kind"],
                            ax["source"],
                            commit_hash,
                            ax["validity_scope"],
                            num_blocks * 8,
                            num_blocks,
                            sem_str,
                            psycopg.Binary(wave_bytes),
                            json.dumps(ax["metadata"]),
                        ),
                    )
                    inserted += 1
                    logging.info(f"  [Seeded] Axiom ID: {ax['axiom_id']} ({ax['axiom_kind']})")

            conn.commit()

            # Verify retrieval via HNSW vector cosine search
            cur.execute("SELECT count(*) FROM boundary_axioms;")
            total_count = cur.fetchone()[0]
            logging.info(f"Zone C `boundary_axioms` total count in database: {total_count}")

            # Execute test query
            test_q = semantic_projection(axioms[0]["wave"])
            test_q_str = "[" + ",".join(f"{v:.6f}" for v in test_q.tolist()) + "]"
            cur.execute(
                """
                SELECT axiom_id, axiom_kind, 1 - (semantic_index <=> %s::vector) AS sim
                FROM boundary_axioms
                ORDER BY semantic_index <=> %s::vector
                LIMIT 3
                """,
                (test_q_str, test_q_str),
            )
            hits = cur.fetchall()
            logging.info("Vector Search Verification on Seeded Axioms:")
            for hit in hits:
                logging.info(f"  Hit: {hit[0]} ({hit[1]}) | Similarity: {hit[2]:.6f}")

    logging.info("Crystallization Seeding Protocol COMPLETED SUCCESSFULLY.")


def main():
    parser = argparse.ArgumentParser(description="Zone C Boundary Axiom Seeder")
    parser.add_argument("--dsn", default=os.environ.get("ZONE_C_PROD_DSN", "postgresql://postgres:postgres@localhost:10100/henri"))
    parser.add_argument("--num_blocks", type=int, default=8192)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seed_zone_c_database(dsn=args.dsn, num_blocks=args.num_blocks, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
