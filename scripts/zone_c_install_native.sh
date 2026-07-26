#!/usr/bin/env bash
# Install native Zone C dependencies on an Ubuntu PostgreSQL host.
# This is for the Vast instance. It is not a Docker entrypoint.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: run as root" >&2
  exit 2
fi

PG_MAJOR="${PG_MAJOR:-16}"

apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg apt-transport-https postgresql-common postgresql-client-${PG_MAJOR}

if ! apt-cache show "postgresql-${PG_MAJOR}-pgvector" >/dev/null 2>&1; then
  echo "ERROR: postgresql-${PG_MAJOR}-pgvector is unavailable in the configured apt sources" >&2
  exit 4
fi
apt-get install -y --no-install-recommends "postgresql-${PG_MAJOR}-pgvector"

# TimescaleDB publishes Ubuntu packages through packagecloud. Add the official
# repository only if the package is not already visible to apt.
if ! apt-cache show "timescaledb-2-postgresql-${PG_MAJOR}" >/dev/null 2>&1; then
  tmp="$(mktemp)"
  trap 'rm -f "$tmp"' EXIT
  curl -fsSL https://packagecloud.io/install/repositories/timescale/timescaledb/script.deb.sh -o "$tmp"
  bash "$tmp"
  apt-get update
fi

apt-get install -y --no-install-recommends "timescaledb-2-postgresql-${PG_MAJOR}"

# The Timescale loader requires shared_preload_libraries. Tune only the active
# cluster and restart it through the Debian cluster manager.
if command -v pg_conftool >/dev/null 2>&1; then
  pg_conftool "${PG_MAJOR}" main set shared_preload_libraries timescaledb
fi
pg_ctlcluster "${PG_MAJOR}" main restart

pg_lsclusters
