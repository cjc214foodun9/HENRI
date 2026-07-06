import os
import glob
import uuid
import torch
import torch.nn.functional as F
import numpy as np
import psycopg

from holographic_vector_lifter import HolographicVectorLifter

class SimpleTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        
    def encode(self, text):
        words = text.split()
        return [hash(w) % self.vocab_size for w in words]

def execute_epistemic_seeding():
    print("=== Launching Zone C Epistemic Axiom Seeding ===")
    
    db_params = {
        "dbname": "henri",
        "user": "postgres",
        "password": "password",
        "host": "localhost",
        "port": "5432"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = HolographicVectorLifter(vocab_size=32000, dim=4096).to(device)
    tokenizer = SimpleTokenizer(vocab_size=32000)
    
    # 1. Gather all MD blueprints
    md_files = glob.glob("*.md")
    if not md_files:
        print("[!] No blueprint MD files found.")
        return
        
    print(f"[*] Found {len(md_files)} blueprints to transduce into axioms.")
    
    conn = psycopg.connect(**db_params)
    cur = conn.cursor()
    
    total_seeded = 0
    
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
            
        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 20]
        
        for p in paragraphs:
            # Tokenize and transduce paragraph into Ontological Phase Wave
            tokens = tokenizer.encode(p)
            # Ensure sequence length > 0
            if len(tokens) == 0:
                continue
            
            # Truncate or pad to length 128 for the encoder? 
            # HolographicVectorLifter handles arbitrary lengths along seq_len, 
            # we just need to add batch dimension.
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            with torch.no_grad():
                wave_complex = encoder(input_ids)[0]
                
            # Convert to 8192D representation
            real_part = wave_complex.real.cpu().numpy()
            imag_part = wave_complex.imag.cpu().numpy()
            db_vec = np.concatenate([real_part, imag_part])
            wavefront_str = "[" + ",".join(map(str, db_vec.tolist())) + "]"
            
            concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, p[:100]))
            label = f"Epistemic_Axiom_from_{os.path.basename(md_file)}"
            
            try:
                cur.execute(
                    """
                    INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront, epiplexity_weight, raw_text)
                    VALUES (%s, %s, %s, %s::vector, %s, %s)
                    ON CONFLICT (concept_hash) DO NOTHING;
                    """,
                    (concept_hash, label, "epistemic_blueprint", wavefront_str, 1.0, p)
                )
                total_seeded += cur.rowcount
            except Exception as e:
                print(f"[!] DB Insert Error: {e}")
                conn.rollback()
        
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"[SUCCESS] Holographically injected {total_seeded} boundaries into Zone C.")

if __name__ == "__main__":
    execute_epistemic_seeding()
