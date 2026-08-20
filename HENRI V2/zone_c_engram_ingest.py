# -*- coding: utf-8 -*-
"""Phase 8.37 — streaming Zone C engram ingest for a sealed trajectory bank.

Component B of 8.37 (HENRI-ANALYSIS-SOTA-BOTTLENECKS-2026 §3.2):
populate phylogenetic_engrams_65536 (+ zone_c_engrams stress rollup)
from an authorized henri.arc-trajectory-bank.v1 npz + jsonl + manifest.

Properties:
- Streaming: rows are read from the npz memmap in bounded batches; no
  giant host arrays beyond the memmap; commits every BATCH rows.
- Deterministic ids: uuid from sha256(psi_f16 || onehot_u8 || nxt_f16) →
  INSERT ... ON CONFLICT (id) DO NOTHING RETURNING id → re-running the
  same bank inserts 0 rows (idempotent, verified live in G2).
- Authorized-only: manifest data_source must equal 'authorized'.
- No DDL: reuses the live schema; fails closed on missing tables.

Usage (remote, as root with the runner DSN):
  /venv/main/bin/python zone_c_engram_ingest.py \
      --bank /root/henri-837-bank/trajectories_harvest_837_20260819.npz \
      --manifest .../trajectories_harvest_837_20260819_manifest.json \
      --jsonl .../trajectories_harvest_837_20260819.jsonl \
      --dsn-env /workspace/zonec_prod.env [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional, Tuple

import numpy as np

from henri_trajectory_bank import TrajectoryBank
from zone_c_segment_cache import semantic_projection

BATCH_ROWS = 500


def parse_dsn_env(path: str) -> str:
    """Parse a key=value env file into a psycopg conninfo string.

    Accepts a literal postgresql:// DSN in any value, or key=value pairs
    (host=..., port=..., dbname=..., user=..., password=...).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dsn env file missing: {path}")
    pairs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if not v:
                continue
            if k in ("dsn", "database_url", "url") or k.endswith("_dsn"):
                if v.startswith("postgres"):
                    return v
                continue
            pairs.append(f"{k}={v}")
    if not pairs:
        raise ValueError(f"no usable DSN/conninfo keys in {path}")
    return " ".join(pairs)


def record_id(psi_f16: bytes, onehot_u8: bytes, nxt_f16: Optional[bytes]) -> uuid.UUID:
    """Deterministic id: sha256 over the record's raw stored bytes."""
    h = hashlib.sha256()
    h.update(psi_f16)
    h.update(onehot_u8)
    if nxt_f16 is not None:
        h.update(nxt_f16)
    return uuid.UUID(bytes=h.digest()[:16])


def _sem_list(wave: np.ndarray) -> str:
    """semantic_projection expects a tensor; returns the pgvector literal."""
    import torch
    t = torch.from_numpy(wave.astype(np.float32)).view(-1)
    sem = semantic_projection(t)
    return "[" + ",".join(f"{v:.6f}" for v in sem.tolist()) + "]"


def ingest(
    bank_npz: str,
    manifest_path: str,
    jsonl_path: str,
    dsn_env: str,
    batch_rows: int = BATCH_ROWS,
    dry_run: bool = False,
) -> Dict[str, object]:
    data = TrajectoryBank.load(bank_npz, manifest_path, verify_digest=True)
    manifest = data["manifest"]
    if manifest.get("data_source") != "authorized":
        raise RuntimeError(
            f"refusing ingest: data_source={manifest.get('data_source')} "
            f"(authorized required)")
    psi = data["psi"]            # [M, D] float32 (npz float16 preserved below)
    onehot = data["actions_onehot"]
    nxt = data["next_wave"]      # [M, D] | None
    vocab = data["action_vocab"]

    # jsonl meta alignment (env per record, same append order as npz rows).
    metas: List[Dict[str, object]] = []
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    metas.append(json.loads(line))

    M = psi.shape[0]
    sem_cap = 2000
    inserted = 0
    skipped = 0

    conn = None
    if not dry_run:
        import psycopg
        conninfo = parse_dsn_env(dsn_env)
        conn = psycopg.connect(conninfo, connect_timeout=8)
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM phylogenetic_engrams_65536")
        pre_count = int(cur.fetchone()[0])
    else:
        pre_count = None

    t0 = time.time()
    for start in range(0, M, batch_rows):
        stop = min(start + batch_rows, M)
        batch_psi = psi[start:stop].astype(np.float16)
        batch_one = onehot[start:stop].astype(np.uint8)
        batch_nxt = nxt[start:stop].astype(np.float16) if nxt is not None else None

        if dry_run:
            inserted += stop - start
            continue

        new_ids: List[uuid.UUID] = []
        sem_strs: List[str] = []
        wave_bytes: List[bytes] = []
        ctxs: List[str] = []
        rows = []
        for i in range(stop - start):
            rid = record_id(
                batch_psi[i].tobytes(), batch_one[i].tobytes(),
                batch_nxt[i].tobytes() if batch_nxt is not None else None)
            ctx = "unknown"
            meta = metas[start + i] if start + i < len(metas) else {}
            env = str(meta.get("env", "unknown"))
            act = vocab[int(batch_one[i].argmax())] if batch_one[i].sum() else "?"
            ctx = f"{env}:{act}"
            rows.append((rid, ctx, _sem_list(batch_psi[i].astype(np.float32)),
                         batch_psi[i].astype(np.float32).tobytes()))
        if rows:
            for rid, ctx, sem, wb in rows:
                cur.execute(
                    """INSERT INTO phylogenetic_engrams_65536
                       (id, timestamp, environmental_context_hash,
                        semantic_index, engram_wave_bytes)
                       VALUES (%s, now(), %s, %s::vector, %s)
                       ON CONFLICT (id) DO NOTHING RETURNING id""",
                    (rid, ctx, sem, psycopg.Binary(wb)))
                row = cur.fetchone()
                if row is not None:
                    new_ids.append(row[0])
                    # Telemetry rollup row only for genuinely new ids.
                    cur.execute(
                        """INSERT INTO zone_c_engrams
                           (time, axiom_id, domain_tag, phase_vector, sagnac_stress)
                           VALUES (now(), %s, %s, %s::vector, %s)""",
                        (rid, ctx, sem, 0.0))
        conn.commit()
        inserted += len(new_ids)
        skipped += len(rows) - len(new_ids)
        if (start // batch_rows) % 20 == 0:
            print(f"[ingest] rows {start}/{M} new={inserted} dup={skipped}")

    if conn is not None:
        cur.execute("SELECT count(*) FROM phylogenetic_engrams_65536")
        post_count = int(cur.fetchone()[0])
        conn.close()
    else:
        post_count = None

    out = {
        "schema": "henri.phase837.ingest.v1",
        "status": "OK",
        "records_read": int(M),
        "inserted": inserted,
        "duplicates_skipped": skipped,
        "pre_count": pre_count,
        "post_count": post_count,
        "delta": (post_count - pre_count) if (pre_count is not None
                                              and post_count is not None) else None,
        "dry_run": bool(dry_run),
        "elapsed_s": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=1))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Zone C trajectory bank ingest")
    p.add_argument("--bank", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--jsonl", required=True)
    p.add_argument("--dsn-env", required=True)
    p.add_argument("--batch", type=int, default=BATCH_ROWS)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    ingest(args.bank, args.manifest, args.jsonl, args.dsn_env,
           batch_rows=args.batch, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
