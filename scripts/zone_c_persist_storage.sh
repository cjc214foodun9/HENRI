#!/usr/bin/env bash
# Move the native PostgreSQL cluster onto a persistent Vast volume.
# Expected volume mount: /workspace. This script changes only the PostgreSQL
# cluster data_directory after validating the mount.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: run as root" >&2
  exit 2
fi

PG_MAJOR="${PG_MAJOR:-16}"
CLUSTER="${PG_CLUSTER:-main}"
PGDATA="${ZONE_C_PGDATA:-/workspace/zone_c/pgdata}"

if [[ ! -d /workspace ]]; then
  echo "ERROR: /workspace is missing" >&2
  exit 2
fi

FSTYPE="$(findmnt -T /workspace -o FSTYPE -n 2>/dev/null || true)"
if [[ "${FSTYPE}" == "overlay" || -z "${FSTYPE}" ]]; then
  echo "BLOCKED: /workspace is not confirmed as a mounted persistent volume" >&2
  exit 3
fi

mkdir -p "$(dirname "$PGDATA")"
if [[ -e "$PGDATA/PG_VERSION" ]]; then
  if [[ "$(cat "$PGDATA/PG_VERSION")" != "$PG_MAJOR" ]]; then
    echo "ERROR: existing PostgreSQL data directory has incompatible version" >&2
    exit 4
  fi
elif [[ -n "$(find "$PGDATA" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: target data directory is non-empty but is not a PostgreSQL cluster: $PGDATA" >&2
  exit 5
else
  mkdir -p "$PGDATA"
  chown postgres:postgres "$PGDATA"
  # Copy the current fresh cluster without deleting anything on the target.
  rsync -a /var/lib/postgresql/${PG_MAJOR}/${CLUSTER}/ "$PGDATA/"
fi

chown -R postgres:postgres "$PGDATA"
chmod 700 "$PGDATA"

pg_ctlcluster "${PG_MAJOR}" "${CLUSTER}" stop || true
pg_conftool "${PG_MAJOR}" "${CLUSTER}" set data_directory "$PGDATA"
pg_ctlcluster "${PG_MAJOR}" "${CLUSTER}" start

actual="$(sudo -u postgres psql -h /var/run/postgresql -p 10100 -d postgres -Atc "SHOW data_directory" 2>/dev/null || true)"
if [[ "$actual" != "$PGDATA" ]]; then
  echo "ERROR: PostgreSQL data_directory verification failed: got ${actual@Q}, expected ${PGDATA@Q}" >&2
  exit 6
fi

echo "Persistent PostgreSQL data directory: $actual"
