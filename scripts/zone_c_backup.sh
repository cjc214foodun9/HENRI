#!/usr/bin/env bash
# Logical backup for production Zone C. Run after a successful bootstrap.
set -euo pipefail

BACKUP_DIR="${ZONE_C_BACKUP_DIR:?ZONE_C_BACKUP_DIR must point to an external or mounted backup path}"
PGHOST="${ZONE_C_PGHOST:-localhost}"
PGPORT="${ZONE_C_PGPORT:-10100}"
PGUSER="${ZONE_C_PGUSER:-postgres}"
DATABASE="henri"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$BACKUP_DIR"

pg_dump --format=custom --no-owner --no-privileges \
  --host="$PGHOST" --port="$PGPORT" --username="$PGUSER" \
  --file="$BACKUP_DIR/henri_zone_c_${STAMP}.dump" "$DATABASE"

find "$BACKUP_DIR" -type f -name 'henri_zone_c_*.dump' -printf '%T@ %p\n' \
  | sort -nr | tail -n +6 | cut -d' ' -f2- | xargs -r rm -f

sha256sum "$BACKUP_DIR/henri_zone_c_${STAMP}.dump" \
  > "$BACKUP_DIR/henri_zone_c_${STAMP}.dump.sha256"
