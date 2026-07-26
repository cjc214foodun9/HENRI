#!/usr/bin/env bash
# HENRI production instance bootstrap.
# Run on the Vast host as root after a persistent volume is attached.
set -euo pipefail

ROOT="${HENRI_ROOT:-/workspace/HENRI}"
PGDATA="${ZONE_C_PGDATA:-/var/lib/postgresql/16/main}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: run as root" >&2
  exit 2
fi

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: HENRI root does not exist: $ROOT" >&2
  exit 2
fi

if command -v vast-capabilities >/dev/null 2>&1; then
  CAP_JSON="$(vast-capabilities)"
  echo "$CAP_JSON" | jq '{instance: {workspace: .instance.workspace, workspace_is_volume: .instance.workspace_is_volume, public_ip: .instance.public_ip}, gpu: .hardware.gpu.summary}'
  if [[ "$(echo "$CAP_JSON" | jq -r '.instance.workspace_is_volume // false')" != "true" && "${ALLOW_EPHEMERAL_ZONE_C:-0}" != "1" ]]; then
    echo "BLOCKED: this instance has no persistent workspace volume." >&2
    echo "Attach a persistent volume before installing production Zone C." >&2
    exit 3
  fi
fi

if [[ "${PERSIST_ZONE_C_STORAGE:-0}" == "1" ]]; then
  "$ROOT/scripts/zone_c_persist_storage.sh"
fi

"$ROOT/scripts/zone_c_install_native.sh"

BOOTSTRAP_ARGS=(
  --host "${ZONE_C_PGHOST:-localhost}"
  --port "${ZONE_C_PGPORT:-10100}"
  --user "${ZONE_C_PGUSER:-postgres}"
  --database henri
  --pgdata "$PGDATA"
  --apply
)

if [[ "${CREATE_ZONE_C_DATABASE:-0}" == "1" ]]; then
  BOOTSTRAP_ARGS+=(--create-database)
fi
if [[ "${ALLOW_EPHEMERAL_ZONE_C:-0}" == "1" ]]; then
  BOOTSTRAP_ARGS+=(--allow-ephemeral)
fi

runuser -u postgres -- python3 "$ROOT/scripts/zone_c_bootstrap.py" "${BOOTSTRAP_ARGS[@]}"

if [[ "${ZONE_C_BACKUP_DIR:-}" != "" ]]; then
  "$ROOT/scripts/zone_c_backup.sh"
fi

echo "HENRI production instance bootstrap complete."
