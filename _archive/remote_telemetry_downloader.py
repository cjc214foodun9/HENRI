import os
import json
import psycopg
import datetime
from decimal import Decimal

# Connection string mapped to the vast.ai assigned proxy port or local SSH tunnel
VAST_AI_DB_URL = "postgresql://postgres:password@localhost:10100/henri"
OUTPUT_DIR = "HENRI V2/telemetry_logs"

def sync_telemetry():
    """
    Connects to the remote Zone C hypertable on the Vast.ai cluster.
    Extracts the latest continuous phase states (Real/Imaginary) and Sagnac validation metrics.
    Dumps the results into a structured local .jsonl file for thermodynamic plotting.
    """
    print(f"[*] Initiating secure telemetry sync with {VAST_AI_DB_URL}...")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filepath = os.path.join(OUTPUT_DIR, f"vast_ai_sync_{timestamp_str}.jsonl")
    
    # Query the 500 most recent wave states
    query = """
    SELECT 
        recorded_at, 
        domain, 
        subdomain, 
        concept_key, 
        real_phases, 
        imag_phases, 
        phase_delta, 
        sagnac_clearance 
    FROM zone_c_resonant_hypersphere
    ORDER BY recorded_at DESC
    LIMIT 500;
    """
    
    try:
        with psycopg.connect(VAST_AI_DB_URL) as conn:
            with conn.cursor() as cur:
                print("[*] Connection established. Executing hypertable extraction...")
                cur.execute(query)
                records = cur.fetchall()
                
                print(f"[*] Downloaded {len(records)} continuous wave states.")
                
                with open(output_filepath, 'w') as f:
                    for row in records:
                        # Unpack the tuple based on the SELECT query
                        recorded_at, domain, subdomain, concept_key, real_phases, imag_phases, phase_delta, sagnac_clearance = row
                        
                        # Serialize the record into a JSON object
                        log_entry = {
                            "timestamp": recorded_at.isoformat() if isinstance(recorded_at, datetime.datetime) else str(recorded_at),
                            "domain": domain,
                            "subdomain": subdomain,
                            "concept_key": concept_key,
                            "real_phases": real_phases,
                            "imag_phases": imag_phases,
                            "phase_delta": float(phase_delta) if isinstance(phase_delta, Decimal) else phase_delta,
                            "sagnac_clearance": sagnac_clearance
                        }
                        f.write(json.dumps(log_entry) + '\n')
                        
                print(f"[SUCCESS] Telemetry successfully saved to {output_filepath}")
                
    except Exception as e:
        print(f"[FATAL] Failed to synchronize with Vast.ai Zone C storage: {e}")

if __name__ == "__main__":
    sync_telemetry()