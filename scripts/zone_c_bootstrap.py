#!/usr/bin/env python3
"""Bootstrap and verify native Zone C on the active HENRI host.

The script is deliberately explicit:
- it never silently creates a different database;
- it requires the production database name `henri`;
- it requires a persistent PostgreSQL data path unless overridden for a
  disposable development experiment;
- it executes the canonical schema in a single transaction only after all
  preflight checks pass.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "HENRI V2" / "zone_c_schema.sql"


def run(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def assert_persistent(path: str, allow_ephemeral: bool) -> dict:
    target = Path(path)
    if not target.exists():
        raise RuntimeError(f"PostgreSQL data path does not exist: {target}")
    stat = os.stat(target)
    result = {"path": str(target), "device": stat.st_dev, "filesystem": None}
    try:
        result["filesystem"] = run(["findmnt", "-T", str(target), "-o", "FSTYPE", "-n"])
    except Exception:
        pass
    # Docker overlay and ordinary container paths are not a production
    # persistence guarantee. The operator must explicitly override this check.
    if not allow_ephemeral and result["filesystem"] in {"overlay", "aufs", ""}:
        raise RuntimeError(
            f"PostgreSQL path {target} is not confirmed persistent "
            f"(filesystem={result['filesystem']!r}); provide a mounted volume "
            "or use --allow-ephemeral only for a disposable test"
        )
    return result


def connect(args: argparse.Namespace, database: str = "postgres"):
    return psycopg.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        dbname=database,
        connect_timeout=10,
    )


def enumerate_databases(args: argparse.Namespace) -> list[str]:
    with connect(args) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SELECT datname FROM pg_database WHERE datallowconn ORDER BY 1")
        return [row[0] for row in cur.fetchall()]


def ensure_database(args: argparse.Namespace) -> None:
    if args.database in enumerate_databases(args):
        return
    if not args.create_database:
        raise RuntimeError(
            f"database {args.database!r} is absent. Re-run with --create-database "
            "only after confirming this is the intended empty production target."
        )
    if args.database != "henri":
        raise RuntimeError("production bootstrap only permits database name 'henri'")
    with connect(args) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE DATABASE henri")


def preflight(args: argparse.Namespace) -> dict:
    if args.database != "henri":
        raise RuntimeError("production Zone C database must be named 'henri'")
    persistent = assert_persistent(args.pgdata, args.allow_ephemeral)
    databases = enumerate_databases(args)
    if args.database not in databases and not args.create_database:
        raise RuntimeError(
            f"database {args.database!r} is absent; no schema action was performed"
        )
    return {
        "host": args.host,
        "port": args.port,
        "database": args.database,
        "postgres_user": args.user,
        "databases": databases,
        "persistent_path_check": persistent,
        "schema_file": str(SCHEMA),
    }


def apply_schema(args: argparse.Namespace) -> dict:
    schema = SCHEMA.read_text(encoding="utf-8")
    with connect(args, args.database) as conn:
        with conn.cursor() as cur:
            cur.execute(schema)
        conn.commit()
    return verify(args)


def verify(args: argparse.Namespace) -> dict:
    with connect(args, args.database) as conn, conn.cursor() as cur:
        cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SELECT current_database(), current_user, version()")
        identity = cur.fetchone()
        cur.execute("SELECT extname, extversion FROM pg_extension ORDER BY 1")
        extensions = [dict(name=row[0], version=row[1]) for row in cur.fetchall()]
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN (
                'zone_c_schema_migrations', 'phylogenetic_engrams_65536',
                'zone_c_engrams', 'zone_c_resonant_hypersphere',
                'zone_c_subspace_artifacts_v1'
              )
            ORDER BY 1
        """)
        tables = [row[0] for row in cur.fetchall()]
        cur.execute("""
            SELECT hypertable_name
            FROM timescaledb_information.hypertables
            WHERE hypertable_name IN ('zone_c_engrams', 'zone_c_resonant_hypersphere')
            ORDER BY 1
        """)
        hypertables = [row[0] for row in cur.fetchall()]
        cur.execute("SELECT environment FROM _zonec_environment ORDER BY id LIMIT 1")
        environment = cur.fetchone()[0]
        return {
            "identity": identity,
            "extensions": extensions,
            "tables": tables,
            "hypertables": hypertables,
            "environment": environment,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="/var/run/postgresql")
    parser.add_argument("--port", type=int, default=10100)
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--database", default="henri")
    parser.add_argument("--pgdata", default="/var/lib/postgresql/16/main")
    parser.add_argument("--create-database", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-ephemeral", action="store_true")
    args = parser.parse_args()

    if not SCHEMA.exists():
        raise FileNotFoundError(SCHEMA)
    report = {"status": "PREFLIGHT", "preflight": preflight(args)}
    if args.apply:
        ensure_database(args)
        report["status"] = "APPLIED"
        report["verification"] = apply_schema(args)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error": type(exc).__name__, "message": str(exc)}, indent=2))
        raise SystemExit(2)
