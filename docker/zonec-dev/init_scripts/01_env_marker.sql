-- Zone C DEV environment marker — the guardrail anchor.
-- zone_c_env.assert_zone_c_env(conn, 'dev') requires this row to exist.
-- Production Zone C (Vast.ai 5090, db `henri`) must NEVER contain this table:
-- its absence is what makes a production target fail a dev assertion.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS _zonec_environment (
    id SERIAL PRIMARY KEY,
    environment TEXT NOT NULL,
    seeded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO _zonec_environment (environment)
SELECT 'dev'
WHERE NOT EXISTS (SELECT 1 FROM _zonec_environment);
