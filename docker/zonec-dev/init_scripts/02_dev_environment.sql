-- Development-only environment override.
-- The canonical schema seeds 'prod' when no marker exists. Replace that
-- marker inside the disposable dev database so zone_c_env can reject a prod
-- DSN and accept this target as dev.
DELETE FROM _zonec_environment;
INSERT INTO _zonec_environment (environment) VALUES ('dev');
