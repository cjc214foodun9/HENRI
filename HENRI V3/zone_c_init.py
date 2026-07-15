"""
Project HENRI: Zone C Database Schema Initializer
Component: TimescaleDB + pgvector Integration Core
Author: Aletheia (Systems Architect)
Date: 2026-07-14

This module configures the database schema for the permanent, zero-entropy
Holographic Engram Reservoir in Zone C. It creates the pgvector hypersphere index 
required for high-density, O(1) content-addressable pre-fetching.
"""

import os
import sys

def generate_zone_c_schema() -> str:
    """
    Generates the strict PostgreSQL/TimescaleDB migrations file to structure 
    the holographic engram database with index guards.
    """
    schema = """-- Project HENRI: Database Migration V1.0.0
-- Establishes the physical boundaries of the Zone C Engram Reservoir.

-- 1. Enable Vector Extensibility
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- 2. Establish Lexical Hypertable
CREATE TABLE IF NOT EXISTS henri_axiomatic_lexicon (
    id BIGSERIAL,
    token VARCHAR(256) NOT NULL UNIQUE,
    category VARCHAR(128) NOT NULL,
    -- Store phase representation as 4096-dimensional float32 arrays
    phase_wave_real vector(4096) NOT NULL,
    phase_wave_imag vector(4096) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id, created_at)
);

-- 3. Turn into a TimescaleDB Hypertable partitioned by time
SELECT create_hypertable('henri_axiomatic_lexicon', 'created_at', if_not_exists => TRUE);

-- 4. Build Cosine Index for ultra-fast, high-density phase-angle retrieval
CREATE INDEX IF NOT EXISTS idx_axiom_real_cosine 
ON henri_axiomatic_lexicon USING hnsw (phase_wave_real vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_axiom_imag_cosine 
ON henri_axiomatic_lexicon USING hnsw (phase_wave_imag vector_cosine_ops);
"""
    return schema

if __name__ == "__main__":
    print("$$ DATABASE $$ Compiling pgvector / TimescaleDB Zone C migrations...")
    schema_sql = generate_zone_c_schema()
    
    # Save the output directly into the workspace migration file
    output_filename = "001_init_hypertables.sql"
    with open(output_filename, "w") as f:
        f.write(schema_sql)
    
    print(f"$$ DATABASE $$ Complete. Axiomatic schema generated and saved to: {output_filename}")
    print("[Instruction] Execute this migration against your local TimescaleDB instance using: psql -d henri -f 001_init_hypertables.sql")
```
