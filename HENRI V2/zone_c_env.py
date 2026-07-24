"""Zone C environment resolver — structural separation of dev vs production.

WHY THIS EXISTS (load-bearing):
  PRODUCTION Zone C  = Vast.ai 5090 TimescaleDB, reached ONLY via SSH tunnel at
                       localhost:10100, db name `henri`. Holds world-knowledge
                       engrams (zone_c_resonant_hypersphere: 4096-dim phase
                       arrays). Writes here are permanent world-model state.
  DEV Zone C         = local Docker container, localhost:5433, db name
                       `henri_zonec_dev`. Disposable schema sandbox. Any table
                       may be dropped at any time.

The two must never be confusable. This module makes the distinction structural
rather than remembered:

  1. The production DSN is returned ONLY when ZONE_C_ENV=prod is set explicitly.
     Default env is dev; silence never routes to production.
  2. A dev DSN must point at localhost AND use a db name ending in `_dev`.
     Anything else is rejected before a connection is attempted.
  3. assert_zone_c_env() verifies the environment marker table inside the
     connected database before any write/DDL. The dev init script seeds
     `_zonec_environment` with 'dev'; production is asserted by host/port
     heuristics (tunnel port 10100 or db name `henri`).

Usage:
    from zone_c_env import resolve_zone_c_dsn, assert_zone_c_env
    dsn = resolve_zone_c_dsn()            # dev unless ZONE_C_ENV=prod
    conn = psycopg.connect(dsn)
    assert_zone_c_env(conn, "dev")        # raises before any write if wrong DB
"""
from __future__ import annotations

import os
import re

DEV_DB_SUFFIX = "_dev"
DEV_PORT = 5434  # 5433 is occupied by a host service (PID 18260); keep in sync with docker/zonec-dev/docker-compose.yml
DEV_DEFAULT_DSN = (
    f"postgres://zonec_dev_user:zonec_dev@localhost:{DEV_PORT}/henri_zonec_dev"
)
# Production reachability fingerprints (NOT secrets, just topology markers).
_PROD_TUNNEL_PORT = 10100
_PROD_DB_NAME = "henri"


class ZoneCEnvError(RuntimeError):
    """Raised when a Zone C environment selection is ambiguous or unsafe."""


def _parse_dsn(dsn: str) -> dict:
    m = re.match(r"postgres(?:ql)?://([^:]+):([^@]+)@([^:/]+):(\d+)/(\w+)", dsn)
    if not m:
        raise ZoneCEnvError(f"Unparseable DSN shape: {dsn!r}")
    user, _pw, host, port, db = m.groups()
    return {"user": user, "host": host, "port": int(port), "db": db}


def _assert_dev_shape(parts: dict, dsn: str) -> None:
    if parts["host"] not in ("localhost", "127.0.0.1"):
        raise ZoneCEnvError(
            f"DEV DSN must target localhost, got host={parts['host']!r}. "
            "Refusing: a non-local dev target risks being a production box."
        )
    if not parts["db"].endswith(DEV_DB_SUFFIX):
        raise ZoneCEnvError(
            f"DEV database name must end in '{DEV_DB_SUFFIX}', got "
            f"{parts['db']!r}. Refusing: ambiguous target could be production."
        )


def _looks_like_production(parts: dict) -> bool:
    return parts["port"] == _PROD_TUNNEL_PORT or parts["db"] == _PROD_DB_NAME


def resolve_zone_c_dsn() -> str:
    """Return the DSN for the explicitly selected environment.

    ZONE_C_ENV: 'dev' (default) | 'prod'.
      dev  -> ZONE_C_DEV_DSN, else DEV_DEFAULT_DSN. Must pass dev-shape checks.
      prod -> ZONE_C_PROD_DSN (required, never defaulted, never guessed).
    """
    env = os.environ.get("ZONE_C_ENV", "dev").strip().lower()
    if env == "prod":
        dsn = os.environ.get("ZONE_C_PROD_DSN", "").strip()
        if not dsn:
            raise ZoneCEnvError(
                "ZONE_C_ENV=prod requires ZONE_C_PROD_DSN to be set. "
                "Production coordinates are never defaulted."
            )
        parts = _parse_dsn(dsn)
        if not _looks_like_production(parts):
            raise ZoneCEnvError(
                "ZONE_C_ENV=prod but DSN does not match production topology "
                f"(tunnel port {_PROD_TUNNEL_PORT} or db '{_PROD_DB_NAME}'). "
                "Refusing: mislabeled environment."
            )
        return dsn
    if env != "dev":
        raise ZoneCEnvError(f"ZONE_C_ENV must be 'dev' or 'prod', got {env!r}")
    dsn = os.environ.get("ZONE_C_DEV_DSN", "").strip() or DEV_DEFAULT_DSN
    parts = _parse_dsn(dsn)
    _assert_dev_shape(parts, dsn)
    if _looks_like_production(parts):
        raise ZoneCEnvError(
            "ZONE_C_ENV=dev but DSN matches production topology. Refusing."
        )
    return dsn


def assert_zone_c_env(conn, expected: str) -> str:
    """Verify the CONNECTED database's environment marker before writes.

    Returns the observed environment string ('dev' / 'prod' / 'unknown').
    Raises ZoneCEnvError on mismatch. Call this immediately after connecting,
    before any INSERT/DDL.
    """
    expected = expected.strip().lower()
    observed = "unknown"
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT environment FROM _zonec_environment ORDER BY id LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                observed = str(row[0]).strip().lower()
    except Exception:
        # Marker table absent: dev DBs always have it (seeded by init script),
        # so absence means this is NOT a dev database.
        observed = "unknown"

    if expected == "dev":
        if observed != "dev":
            raise ZoneCEnvError(
                f"Expected a DEV database (marker 'dev'), observed "
                f"{observed!r}. Refusing to write: the dev init script seeds "
                "_zonec_environment, and only production lacks it."
            )
    elif expected == "prod":
        # Production assertion is topological (checked at DSN resolution) plus
        # negative marker evidence here; absence of a dev marker is consistent
        # with production but not proof. Documented as a guardrail, not proof.
        if observed == "dev":
            raise ZoneCEnvError(
                "Expected PRODUCTION but connected database carries a 'dev' "
                "marker. Refusing: mislabeled environment."
            )
    else:
        raise ZoneCEnvError(f"expected must be 'dev' or 'prod', got {expected!r}")
    return observed
