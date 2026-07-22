import psycopg
import json
import datetime
from pathlib import Path

def sync_telemetry():
    # Localhost connection for execution natively on Vast.ai
    db_url = "postgres://postgres:password@localhost:10100/henri"
    log_dir = Path("./telemetry_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = log_dir / f"vast_ai_sync_{timestamp}.jsonl"
    
    print(f"[ALETHEIA] Connecting to Vast.ai TimescaleDB at 62.107.25.198:53468...")
    
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                print("[ALETHEIA] Querying Zone C Resonant Hypersphere...")
                
                # Fetch the latest 500 records to prevent massive downloads (4096-dim arrays are large)
                cur.execute("""
                    SELECT id, domain, subdomain, concept_key, recorded_at, 
                           real_phases, imag_phases, phase_delta, sagnac_clearance 
                    FROM zone_c_resonant_hypersphere 
                    ORDER BY recorded_at DESC
                    LIMIT 500
                """)
                
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
                        
                print(f"[ALETHEIA] Sync Complete! Telemetry dumped to {out_file.absolute()}")
                
    except Exception as e:
        print(f"[ALETHEIA FATAL] Sync failed: {e}")

if __name__ == "__main__":
    sync_telemetry()
