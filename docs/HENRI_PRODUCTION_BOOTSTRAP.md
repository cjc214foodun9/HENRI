# HENRI Production Bootstrap Protocol

## Scope

This protocol initializes the HENRI CUDA runtime and the Zone C persistence plane on a Vast instance. It separates:

- the CUDA/Python execution image;
- native PostgreSQL/TimescaleDB on the instance;
- persistent storage for PostgreSQL;
- the HENRI schema and verification step.

The Docker image does not create or own the production database. The active Vast instance does not expose Docker to the container, so a Docker Compose file cannot initialize the native PostgreSQL service there.

## Current audit result

Observed on instance `45676286`:

```text
GPU: RTX 5090
Ubuntu: 24.04.4 LTS
PostgreSQL: 16.14 on localhost:10100
Docker CLI inside instance: absent
PostgreSQL cluster: online
PostgreSQL database `henri`: absent
pgvector: not installed
TimescaleDB: not installed
workspace_is_volume: false
```

The old `Dockerfile.vast` only installs CUDA and Python packages. It does not install PostgreSQL, TimescaleDB, pgvector, create `henri`, mount persistent storage, or apply a schema. The old `production_docker_compose_stack.yaml` is not runnable as written: it references missing Dockerfiles and init files, uses a malformed database URL, defaults to a different database name, and mixes optional Hermes services with the database bootstrap.

## Canonical files

```text
Dockerfile.vast
HENRI V2/zone_c_schema.sql
HENRI V2/zone_c_env.py
docker/zonec-dev/docker-compose.yml
docker/zonec-dev/init_scripts/02_dev_environment.sql
scripts/zone_c_install_native.sh
scripts/zone_c_bootstrap.py
scripts/henri_instance_bootstrap.sh
scripts/zone_c_backup.sh
```

## Production procedure

### 1. Attach persistent storage

The current instance reports `workspace_is_volume=false`. Do not initialize production Zone C on the current ephemeral PostgreSQL path.

Create or identify a Vast volume on the same machine, then create/recreate the instance with the volume mounted. Vast volume documentation uses:

```bash
vastai show volumes
vastai create instance <offer-id> --env '-v V.<volume-id>:/workspace' --disk 30 --ssh --direct
```

The current API identity cannot list volumes (`401: api.volumes route access missing`). The owner must enable volume API access or attach the volume in the Vast UI.

### 2. Reconnect and confirm persistence

```bash
vast-capabilities | jq '.instance.workspace_is_volume'
findmnt -T /workspace
findmnt -T /var/lib/postgresql/16/main
```

The value must be `true` for the production workspace or the PostgreSQL data path must be on an independently persistent mount.

### 3. Synchronize the repository

The remote workspace was at `d9d4c28` during the audit. The local branch contains the Zone C fail-closed change and bootstrap implementation. Synchronize only after checking for remote changes:

```bash
cd /workspace/HENRI
 git status --short
 git fetch origin main
 git diff --stat HEAD..origin/main
 git pull --ff-only origin main
 git -C 'HENRI V2' log -1 --oneline
```

Preserve remote telemetry and metadata. Do not use `git reset --hard`.

### 4. Install native dependencies

```bash
cd /workspace/HENRI
bash scripts/zone_c_install_native.sh
```

This installs PostgreSQL pgvector from Ubuntu and TimescaleDB from the official Timescale package repository, then restarts the PostgreSQL 16 cluster.

### 5. Bootstrap the database

First run preflight without DDL:

```bash
python3 scripts/zone_c_bootstrap.py \
  --host localhost --port 10100 --user postgres \
  --database henri --pgdata /var/lib/postgresql/16/main
```

The expected result is `BLOCKED` until the database exists or the operator explicitly authorizes creation.

For a confirmed fresh production target with persistent storage:

```bash
CREATE_ZONE_C_DATABASE=1 \
  bash scripts/henri_instance_bootstrap.sh
```

Equivalent direct command:

```bash
python3 scripts/zone_c_bootstrap.py \
  --host localhost --port 10100 --user postgres \
  --database henri --pgdata /var/lib/postgresql/16/main \
  --create-database --apply
```

The schema creates:

- `phylogenetic_engrams_65536` with `VECTOR(2000)` HNSW projection and full-wave `BYTEA`;
- `zone_c_engrams` hypertable;
- `zone_c_resonant_hypersphere` telemetry hypertable;
- hourly stress continuous aggregate;
- `zone_c_subspace_artifacts_v1` with explicit version, rank, residual, and validation fields;
- production marker `_zonec_environment = 'prod'`;
- schema migration ledger.

No retention policy is enabled by default. Retention requires observed reuse and backup evidence.

### 6. Verify

```bash
python3 scripts/zone_c_bootstrap.py \
  --host localhost --port 10100 --user postgres \
  --database henri --pgdata /var/lib/postgresql/16/main
```

Then run the HENRI remote suite through the canonical CI path. Do not report CUDA benchmark success from bootstrap verification alone.

### 7. Back up

Set `ZONE_C_BACKUP_DIR` to a persistent external path or mounted volume:

```bash
ZONE_C_BACKUP_DIR=/persistent/henri-backups \
  bash scripts/zone_c_backup.sh
```

The backup script retains five custom-format dumps and writes SHA-256 sidecars.

## Development procedure

The dev Compose file uses:

```text
localhost:5434
henri_zonec_dev
zonec_dev_user
_zonec_environment = dev
```

Start it from the repository root:

```bash
docker compose -f docker/zonec-dev/docker-compose.yml up -d
```

The schema file is mounted read-only into the PostgreSQL init directory. PostgreSQL init scripts run only on an empty named volume. To rebuild a disposable dev database, use `docker compose down -v` only when the dev data may be destroyed.

## Rejected protocols

- Do not use `HENRI V2/zone_c_database_initialization.py`; it creates the obsolete `hrr_canonical_lexicon` table with `VECTOR(8192)` and uses psycopg2.
- Do not use `production_docker_compose_stack.yaml` for Vast production. It is not a valid complete deployment and references missing services/files.
- Do not create `henri` on an unmounted ephemeral path and call it persistent.
- Do not install Docker inside the Vast container. The instance is unprivileged and Docker-in-Docker is unsupported.
- Do not enable retention before backup and reuse measurements exist.
- Do not treat schema presence as proof that invariant subspaces are mathematically valid or task-grounded.
