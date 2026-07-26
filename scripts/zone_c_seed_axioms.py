#!/usr/bin/env python3
"""Seed the explicit HENRI Zone C engineering-axiom catalog.

These records are frozen engineering contracts from the HENRI architecture
ledger. They are not learned world facts, not task outcomes, and not validated
invariant subspaces. The seed is metadata-only and does not inject synthetic
waves into the active GRM checkpoint table.
"""
from __future__ import annotations

import argparse
import json
import psycopg

AXIOMS = [
    {
        "key": "wave_shape_clifford8",
        "domain": "wave_representation",
        "statement": "Planner boundary waves are real Clifford waves with shape [num_blocks, 8].",
        "formal_rule": "psi in R^(num_blocks x 8)",
        "source": "henri-architecture: EFE planner wave boundary",
    },
    {
        "key": "sagnac_delta_bounds",
        "domain": "wave_mechanics",
        "statement": "Normalized Sagnac delta is bounded between 0 and 2.",
        "formal_rule": "delta = 1 - Re(<pred,emp>)/(||pred|| ||emp||), delta in [0,2]",
        "source": "henri-architecture: normalized Sagnac delta",
    },
    {
        "key": "dimension_normalized_residual",
        "domain": "constraints",
        "statement": "Dimension-dependent L2 residual thresholds use sqrt(d) normalization.",
        "formal_rule": "r_norm = ||x||_2 / sqrt(d)",
        "source": "henri-architecture: dimension normalization",
    },
    {
        "key": "candidate_specific_constraint_penalty",
        "domain": "constraints",
        "statement": "Invariant constraints enter action selection as candidate-specific penalties.",
        "formal_rule": "penalty(a) = lambda * ||pred_a - P_inv(pred_a)||_2 / sqrt(d)",
        "source": "henri-architecture: penalty form",
    },
    {
        "key": "no_additive_boundary_attractor",
        "domain": "constraints",
        "statement": "An additive boundary row is not a valid replacement for candidate-specific constraint scoring.",
        "formal_rule": "reject additive boundary-row attractor channel",
        "source": "henri-architecture: falsified boundary mechanism",
    },
    {
        "key": "cholesky_stiefel_retraction",
        "domain": "learned_dynamics",
        "statement": "Cholesky-based retraction is the stable Stiefel path for the current implementation.",
        "formal_rule": "retract(W) = chol-based Stiefel projection",
        "source": "henri-architecture: Cholesky retraction",
    },
    {
        "key": "sgld_noise_scale",
        "domain": "learning",
        "statement": "SGLD noise uses the square root of twice temperature times step size.",
        "formal_rule": "sigma = sqrt(2*T*dt)",
        "source": "henri-architecture: SGLD noise",
    },
    {
        "key": "edmd_dual_thin_svd",
        "domain": "learned_dynamics",
        "statement": "Production EDMD uses dual/thin-SVD methods and avoids d-squared tensors.",
        "formal_rule": "effective_rank = min(requested_rank, N)",
        "source": "henri-architecture: production EDMD",
    },
    {
        "key": "young_edmd_blend",
        "domain": "learned_dynamics",
        "statement": "Young EDMD fits are blended rather than hard-swapped into the active operator.",
        "formal_rule": "K_active = blend(K_old, K_fit, confidence)",
        "source": "henri-architecture: young EDMD fit policy",
    },
    {
        "key": "zone_c_fail_closed",
        "domain": "persistence",
        "statement": "A live Zone C connection failure must not select an in-memory surrogate.",
        "formal_rule": "live failure -> DatabaseConnectionError",
        "source": "henri-architecture: Zone C fail-closed contract",
    },
]


DDL = """
CREATE TABLE IF NOT EXISTS zone_c_axiom_catalog_v1 (
    axiom_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    axiom_key VARCHAR(128) NOT NULL UNIQUE,
    domain_tag VARCHAR(128) NOT NULL,
    statement TEXT NOT NULL,
    formal_rule TEXT NOT NULL,
    evidence_class VARCHAR(64) NOT NULL DEFAULT 'ENGINEERING_CONTRACT',
    status VARCHAR(32) NOT NULL DEFAULT 'FROZEN_CONTRACT',
    source_ref VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (evidence_class = 'ENGINEERING_CONTRACT'),
    CHECK (status IN ('FROZEN_CONTRACT', 'CANDIDATE', 'REJECTED'))
)
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="/var/run/postgresql")
    ap.add_argument("--port", type=int, default=10100)
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--database", default="henri")
    args = ap.parse_args()

    with psycopg.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        dbname=args.database,
        connect_timeout=10,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(DDL)
            for item in AXIOMS:
                cur.execute(
                    """
                    INSERT INTO zone_c_axiom_catalog_v1
                        (axiom_key, domain_tag, statement, formal_rule,
                         evidence_class, status, source_ref)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (axiom_key) DO UPDATE SET
                        domain_tag = EXCLUDED.domain_tag,
                        statement = EXCLUDED.statement,
                        formal_rule = EXCLUDED.formal_rule,
                        evidence_class = EXCLUDED.evidence_class,
                        status = EXCLUDED.status,
                        source_ref = EXCLUDED.source_ref
                    """,
                    (
                        item["key"], item["domain"], item["statement"],
                        item["formal_rule"], "ENGINEERING_CONTRACT",
                        "FROZEN_CONTRACT", item["source"],
                    ),
                )
        conn.commit()

    with psycopg.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        dbname=args.database,
        connect_timeout=10,
    ) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute(
            "SELECT axiom_key, domain_tag, evidence_class, status "
            "FROM zone_c_axiom_catalog_v1 ORDER BY axiom_key"
        )
        rows = [
            {"axiom_key": r[0], "domain": r[1], "evidence_class": r[2], "status": r[3]}
            for r in cur.fetchall()
        ]
    print(json.dumps({"status": "SEEDED", "count": len(rows), "axioms": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
