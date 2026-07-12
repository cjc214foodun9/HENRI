import torch
import psycopg2
import numpy as np

def upload_zone_c():
    db_path = "zone_c_timescaledb.pt"
    db_url = "postgresql://postgres:password@127.0.0.1:5432/henri"
    
    print(f"[*] Loading Zone C data from {db_path}...")
    try:
        zone_c_db = torch.load(db_path, weights_only=True)
    except:
        zone_c_db = torch.load(db_path)
        
    print("[*] Connecting to PostgreSQL...")
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    
    # Create extension and table if not exists
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS hrr_canonical_lexicon (
            id SERIAL PRIMARY KEY,
            quadrant VARCHAR(50),
            vector vector(8192)
        );
    """)
    conn.commit()
    
    # Truncate to ensure fresh
    cursor.execute("TRUNCATE hrr_canonical_lexicon RESTART IDENTITY;")
    conn.commit()
    
    print("[*] Uploading axioms to pgvector...")
    total_uploaded = 0
    for quad_key, tensor_data in zone_c_db.items():
        if tensor_data.size(0) == 0:
            continue
            
        print(f"   -> Processing Quadrant {quad_key.upper()} ({tensor_data.size(0)} vectors)...")
        # Extract Real and Imaginary, concat to 8192-D real vector
        # tensor_data is [N, 4096] complex64
        real_parts = tensor_data.real.cpu().numpy()
        imag_parts = tensor_data.imag.cpu().numpy()
        
        for i in range(tensor_data.size(0)):
            vector_array = np.concatenate([real_parts[i], imag_parts[i]]).tolist()
            cursor.execute("INSERT INTO hrr_canonical_lexicon (quadrant, vector) VALUES (%s, %s)", (quad_key, vector_array))
            total_uploaded += 1
            if total_uploaded % 500 == 0:
                print(f"      {total_uploaded} uploaded...")
                conn.commit()
                
    conn.commit()
    
    print("[*] Creating HNSW Index for Sagnac Fast-Path...")
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_hrr_vector 
        ON hrr_canonical_lexicon USING hnsw (vector vector_cosine_ops);
    """)
    conn.commit()
    
    cursor.close()
    conn.close()
    print(f"[*] Successfully uploaded {total_uploaded} axiomatic wavefronts to TimescaleDB.")

if __name__ == "__main__":
    upload_zone_c()
