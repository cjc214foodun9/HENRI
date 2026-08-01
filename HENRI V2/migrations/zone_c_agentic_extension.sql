-- Project HENRI V2: Zone C Compounding Agentic Engine Extension (zone_c_agentic_extension.sql)
-- Implements:
--   1. Continuous Epistemic Attractor Rollups (TimescaleDB continuous aggregates over Sagnac delta & Kuramoto r)
--   2. Thermodynamic Synaptic Apoptosis (Ebbinghaus decay kinetics S_k(t) = R_k * exp(-(t-t_k)/tau_decay))
--   3. Spatiotemporal Geodesic Routing (pgvector HNSW indexes over time-partitioned hypertables)
--   4. Real-time PL/pgSQL Sagnac Veto Trigger with LISTEN/NOTIFY

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Continuous Epistemic Attractor Rollups
CREATE MATERIALIZED VIEW IF NOT EXISTS zone_c_attractor_basins_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at) AS bucket,
    environment_id,
    AVG(sagnac_delta) AS avg_sagnac_delta,
    MIN(sagnac_delta) AS min_sagnac_delta,
    AVG(coherence) AS avg_kuramoto_r,
    COUNT(*) AS total_samples
FROM zone_c_telemetry
GROUP BY bucket, environment_id;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('zone_c_attractor_basins_1h',
    start_offset => INTERVAL '1 day',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- 2. Thermodynamic Synaptic Apoptosis Table & Function
CREATE TABLE IF NOT EXISTS zone_c_synaptic_engrams (
    engram_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    environment_id VARCHAR(64) NOT NULL,
    qfhrr_phase_vector vector(2000),
    resonance_count INT DEFAULT 1,
    last_accessed_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Function executing Ebbinghaus synaptic apoptosis (S < 0.15 threshold)
CREATE OR REPLACE FUNCTION zone_c_execute_synaptic_apoptosis(tau_decay_hours FLOAT DEFAULT 24.0, threshold FLOAT DEFAULT 0.15)
RETURNS INT AS $$
DECLARE
    deleted_count INT;
BEGIN
    DELETE FROM zone_c_synaptic_engrams
    WHERE (resonance_count * EXP(-EXTRACT(EPOCH FROM (NOW() - last_accessed_at)) / (tau_decay_hours * 3600.0))) < threshold;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- 3. Spatiotemporal Geodesic Routing HNSW Index
CREATE INDEX IF NOT EXISTS idx_synaptic_engrams_hnsw
ON zone_c_synaptic_engrams USING hnsw (qfhrr_phase_vector vector_cosine_ops);

-- 4. Real-time PL/pgSQL Sagnac Veto Trigger with LISTEN/NOTIFY
CREATE OR REPLACE FUNCTION zone_c_sagnac_veto_notify()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.sagnac_delta > 0.35 THEN
        PERFORM pg_notify('sagnac_veto_channel', json_build_object(
            'event_id', NEW.event_id,
            'environment_id', NEW.environment_id,
            'sagnac_delta', NEW.sagnac_delta,
            'action', NEW.selected_action,
            'timestamp', NEW.created_at
        )::text);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_zone_c_sagnac_veto ON zone_c_telemetry;
CREATE TRIGGER trg_zone_c_sagnac_veto
AFTER INSERT ON zone_c_telemetry
FOR EACH ROW EXECUTE FUNCTION zone_c_sagnac_veto_notify();
