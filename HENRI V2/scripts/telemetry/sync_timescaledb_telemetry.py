"""Export compact Zone C telemetry through the guarded production boundary.

This is a Zone C latent-space adapter. It does not read or write the local
Obsidian agentic graph. Output defaults outside the production repository.
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import psycopg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zone_c_env import assert_zone_c_env, resolve_zone_c_dsn  # noqa: E402

def sync_telemetry(output_dir: str | None = None, limit: int = 500):
    if os.environ.get("ZONE_C_ENV", "dev").strip().lower() != "prod":
        raise RuntimeError("telemetry export requires explicit ZONE_C_ENV=prod")
    db_url = resolve_zone_c_dsn()
    log_dir = Path(output_dir or os.environ.get(
        "HENRI_TELEMETRY_EXPORT_DIR",
        str(Path.home() / "HENRI_telemetry_exports"),
    )).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = log_dir / f"vast_ai_sync_{timestamp}.jsonl"
    
    print("[ZONE_C] Connecting through guarded production DSN")
    
    try:
        with psycopg.connect(db_url, connect_timeout=8) as conn:
            assert_zone_c_env(conn, "prod")
            with conn.cursor() as cur:
                print("[ALETHEIA] Querying Zone C Resonant Hypersphere...")
                
                # Fetch the latest 500 records to prevent massive downloads (4096-dim arrays are large)
                cur.execute("""
                    SELECT id, domain, subdomain, concept_key, recorded_at, 
                           real_phases, imag_phases, phase_delta, sagnac_clearance 
                    FROM zone_c_resonant_hypersphere 
                    ORDER BY recorded_at DESC
                    LIMIT %s
                """, (int(limit),))
                
                rows = cur.fetchall()
                if not rows:
                    print("[ALETHEIA] No telemetry records found in the hypertable yet.")
                    return
                
                print(f"[ALETHEIA] Successfully fetched {len(rows)} records. Writing to {out_file}...")
                
                with open(out_file, 'w', encoding='utf-8') as f:
                    for row in rows:
                        record = {
                            "id": str(row[0]),
                            "domain": row[1],
                            "subdomain": row[2],
                            "concept_key": row[3],
                            "recorded_at": row[4].isoformat() if row[4] else None,
                            "real_phases": row[5],
                            "imag_phases": row[6],
                            "phase_delta": float(row[7]) if row[7] is not None else None,
                            "sagnac_clearance": row[8]
                        }
                        f.write(json.dumps(record) + '\n')
                        
                print(f"[ZONE_C] Export complete: {out_file.absolute()}")
                
    except Exception as e:
        print(f"[ZONE_C BLOCKED] Export failed: {type(e).__name__}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    sync_telemetry(args.output_dir, args.limit)
