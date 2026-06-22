#!/usr/bin/env python3
import os  
import sys
import json  
import uuid  
import argparse
import psycopg  
import numpy as np  
from pathlib import Path

# Add paths to make sure we can resolve henri_contract
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir / "6"))
sys.path.append(str(current_dir / "henri_core"))

try:
    from henri_contract import complex_to_db, DIMS
except ImportError:
    # If not found directly, add current directory
    sys.path.append(str(current_dir))
    from henri_contract import complex_to_db, DIMS

# Database connection URL: check environment variable, then fallback to 5432, then 5433
DB_URL = os.environ.get("DATABASE_URL")
if not DB_URL:
    DB_URL = "postgresql://postgres:password@localhost:5432/henri"

def decode_tensor_data(tensor_data):  
    """Stage 1: Filters trailing zeros and reconstructs text from ASCII arrays."""  
    # Convert floats to raw integer bytes, skipping padding blocks  
    byte_sequence = [int(x) for x in tensor_data if x != 0.0]  
    try:  
        return bytes(byte_sequence).decode('utf-8', errors='ignore')  
    except Exception as e:  
        return f"Decoding Error: {e}"

def generate_deterministic_wavefront(text_content, valence_mode="attractor"):  
    """Stage stage 2: Seeds text uniformly into a 4096D complex hypervector."""  
    # Enforce deterministic seeding using text hash to keep the axiom stable  
    seed_value = int(uuid.uuid5(uuid.NAMESPACE_DNS, text_content).int & 0xFFFFFFFF)  
    rng = np.random.default_rng(seed_value)  
      
    # Generate phases uniformly bounded across the unit circle (-pi to +pi)  
    phases = (rng.random(DIMS.hrr_dim) * 2.0 * np.pi) - np.pi
    
    # Store inverse phase coordinate for repellers if requested
    if valence_mode == "repeller":
        phases = -phases
        
    complex_vector = np.exp(1j * phases)  
    return complex_vector

def ingest_directory_packets(target_dir_path, valence_mode="attractor", db_url=DB_URL):  
    """Walks through directories, processes data foundry packets, and bulk inserts."""  
    path = Path(target_dir_path)  
    if not path.exists():  
        print(f"[ERROR] Specified path does not exist: {target_dir_path}")  
        return  
  
    json_files = list(path.glob("*.json")) + list(path.glob("**/*.json"))  
    if not json_files:  
        print(f"[-] No raw tensor packet JSON files discovered in {target_dir_path}")  
        return  
  
    print(f"[*] Discovered {len(json_files)} packets in {target_dir_path}. Connecting to TimescaleDB...")  
      
    # Establish weight polarity depending on attractor/repeller usage boundaries  
    weight_modifier = 1.0 if valence_mode == "attractor" else -1.0  
  
    try:
        with psycopg.connect(db_url) as conn:  
            with conn.cursor() as cur:  
                conn.autocommit = True  
                records_loaded = 0  
                  
                for file_path in json_files:  
                    try:  
                        with open(file_path, 'r', encoding='utf-8') as f:  
                            data = json.load(f)  
                          
                        # Safety validation for standard format compatibility  
                        if "tensor_data" not in data:  
                            continue  
                              
                        metadata = data.get("metadata", {})  
                        packet_id = metadata.get("id", file_path.stem)  
                        domain_tag = data.get("boundary_type", metadata.get("quadrant", "general_axiom"))  
                          
                        # Execute Stage 1 Translation  
                        decoded_text = decode_tensor_data(data["tensor_data"])  
                        if not decoded_text.strip():  
                            continue  
                          
                        # Execute Stage 2 Mapping (with phase coordinate adjustment for repellers)
                        complex_wave = generate_deterministic_wavefront(decoded_text, valence_mode=valence_mode)  
                          
                        # Execute Stage 3 Serialization  
                        flat_vector_str = complex_to_db(complex_wave, DIMS.hrr_dim)  
                          
                        # Generate permanent primary key hash  
                        concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, decoded_text))  
                          
                        # Execute Stage 4 Database Upload  
                        cur.execute(  
                            """  
                            INSERT INTO hrr_canonical_lexicon   
                            (concept_hash, semantic_label, domain_tag, hrr_wavefront, epiplexity_weight, raw_text)  
                            VALUES (%s, %s, %s, %s::vector, %s, %s)  
                            ON CONFLICT (concept_hash) DO UPDATE   
                            SET epiplexity_weight = EXCLUDED.epiplexity_weight, raw_text = EXCLUDED.raw_text;  
                            """,  
                            (concept_hash, packet_id[:128], domain_tag, flat_vector_str, weight_modifier, decoded_text)  
                        )  
                        records_loaded += 1  
                        print(f"[+] Loaded: {packet_id} as [{valence_mode}] -> Tag: {domain_tag}")  
                          
                    except Exception as file_err:  
                        print(f"[WARNING] Skipping faulty transaction on {file_path.name}: {file_err}")  
                          
                print(f"\n[SUCCESS] Pipeline Complete. Loaded {records_loaded} boundary fields into the lexicon.")  
    except Exception as db_err:
        print(f"[FATAL ERROR] Database connection or query failed: {db_err}")

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Ingest boundary token packages into TimescaleDB.")
    parser.add_argument("--dir", default="esc_compiled_dataset", help="Target directory containing packet JSON files")
    parser.add_argument("--valence", choices=["attractor", "repeller"], default="attractor", help="Valence mode of the axioms")
    parser.add_argument("--db-url", help="Database connection URL")
    args = parser.parse_args()
    
    db_conn_str = args.db_url if args.db_url else DB_URL
    
    # Try local folder first, then absolute target folder
    target_dir = args.dir
    if not Path(target_dir).exists():
        # Try relative to archive if it exists
        archive_path = Path("archive") / target_dir
        if archive_path.exists():
            target_dir = str(archive_path)
            
    ingest_directory_packets(target_dir, valence_mode=args.valence, db_url=db_conn_str)
