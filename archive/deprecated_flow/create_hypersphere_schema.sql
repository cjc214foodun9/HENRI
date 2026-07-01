-- Enable pgvector and cryptographic primitives for secure agential routing  
CREATE EXTENSION IF NOT EXISTS vector;  
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Drop staled tracking frames if overriding context configurations  
DROP TABLE IF EXISTS zone_c_resonant_hypersphere CASCADE;

CREATE TABLE zone_c_resonant_hypersphere (  
    id UUID NOT NULL,  
    domain VARCHAR(64) NOT NULL,          -- e.g., Condensed Matter, Topology, Bioelectricity  
    subdomain VARCHAR(64) NOT NULL,       -- e.g., Fractional Quantum Hall, Planarian Telemetry  
    concept_key VARCHAR(128) NOT NULL,     -- Content-addressable semantic identifier  
    turn_index INT NOT NULL,              -- Swarm refinement cycle tracker  
    recorded_at TIMESTAMPTZ NOT NULL,     -- TimescaleDB partitioning anchor  
      
    -- Disaggregated 4096D Holographic Phase Tensor Components  
    -- CXL-aligned Float32 coordinates representing unit-modulus hypersphere projections  
    real_phases REAL[] NOT NULL,          -- Clamped dimension array [4096]  
    imag_phases REAL[] NOT NULL,          -- Clamped dimension array [4096]  
      
    -- Platonic regularizers and complexity metrics  
    epiplexity_floor REAL NOT NULL,       -- Bounded learnable structural index  
    lipschitz_bound REAL NOT NULL,        -- Continuity ceiling constraint for wave stability  
    metadata JSONB,                       -- Out-of-band domain-specific tensor metadata  
    PRIMARY KEY (id, recorded_at)  
);

-- Convert standard table into an optimized time-series Hypertable  
-- Chunks are partitioned into 1-hour temporal buckets to lock step with the overnight distillation sprints  
SELECT create_hypertable('zone_c_resonant_hypersphere', 'recorded_at', chunk_time_interval => INTERVAL '1 hour');

-- Instantiation of high-speed composite B-Tree and GIST indices for associative neighborhood lookups  
CREATE INDEX idx_domain_concept ON zone_c_resonant_hypersphere(domain, concept_key);  
CREATE INDEX idx_trajectory_flow ON zone_c_resonant_hypersphere(concept_key, turn_index DESC);

-- Custom PostgreSQL function to compute Complex Domain Cosine Similarity natively in silicon rows  
CREATE OR REPLACE FUNCTION complex_hypersphere_resonance(  
    r1 REAL[], i1 REAL[], r2 REAL[], i2 REAL[]  
) RETURNS REAL AS $$  
DECLARE  
    dot_product_real REAL := 0.0;  
    norm_1 REAL := 0.0;  
    norm_2 REAL := 0.0;  
    dim CONSTANT INT := 4096;  
BEGIN  
    FOR k IN 1..dim LOOP  
        -- Complex inner product accumulation: (a + jb) * (c - jd) = (ac + bd) + j(bc - ad)  
        dot_product_real := dot_product_real + (r1[k] * r2[k]) + (i1[k] * i2[k]);  
        norm_1 := norm_1 + (r1[k] * r1[k]) + (i1[k] * i1[k]);  
        norm_2 := norm_2 + (r2[k] * r2[k]) + (i2[k] * i2[k]);  
    END LOOP;  
      
    IF norm_1 = 0.0 OR norm_2 = 0.0 THEN  
        RETURN 0.0;  
    END IF;  
      
    RETURN dot_product_real / (sqrt(norm_1) * sqrt(norm_2));  
END;  
$$ LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE;
