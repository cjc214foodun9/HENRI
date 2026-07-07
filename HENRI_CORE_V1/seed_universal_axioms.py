import os
import json
import uuid
import torch
import torch.nn.functional as F
import numpy as np
import psycopg
from datasets import load_dataset
from transformers import AutoTokenizer
from holographic_vector_lifter import HolographicVectorLifter

class UniversalAxiomSeeder:
    def __init__(self, vocab_size=32000, dim=4096, max_samples=500):
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_samples = max_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("[FOUNDRY] Initializing hf-internal-testing/llama-tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        
        print("[FOUNDRY] Initializing HolographicVectorLifter (Zone C Prism)...")
        torch.manual_seed(42)
        self.lifter = HolographicVectorLifter(vocab_size=vocab_size, dim=dim).to(self.device)
        
        print("[FOUNDRY] Connecting to PostgreSQL TimescaleDB on localhost...")
        self.conn = psycopg.connect(
            dbname="henri", user="postgres", password="password", host="localhost", port="5432"
        )
        
        print("[FOUNDRY] Initializing TimescaleDB Schema (hrr_canonical_lexicon)...")
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS hrr_canonical_lexicon (
                    concept_hash TEXT PRIMARY KEY,
                    semantic_label TEXT,
                    domain_tag TEXT,
                    hrr_wavefront vector(8192),
                    epiplexity_weight FLOAT,
                    raw_text TEXT
                );
            """)
        self.conn.commit()
        
    def _insert_wavefront(self, quadrant_key: str, text: str, wavefront: torch.Tensor):
        real_part = wavefront.real.cpu().numpy()
        imag_part = wavefront.imag.cpu().numpy()
        db_vec = np.concatenate([real_part, imag_part])
        wavefront_str = "[" + ",".join(map(str, db_vec.tolist())) + "]"
        
        concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, text[:100] + quadrant_key))
        label = f"Axiom_{quadrant_key}_{uuid.uuid4().hex[:8]}"
        
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront, epiplexity_weight, raw_text)
                    VALUES (%s, %s, %s, %s::vector, %s, %s)
                    ON CONFLICT (concept_hash) DO NOTHING;
                    """,
                    (concept_hash, label, quadrant_key, wavefront_str, 1.0, text)
                )
            self.conn.commit()
            return True
        except Exception as e:
            print(f"[!] DB Insert Error: {e}")
            self.conn.rollback()
            return False

    def process_hf_stream(self, hf_path: str, split: str, text_column: str, quadrant_key: str):
        print(f"[*] Streaming {hf_path} for Quadrant {quadrant_key.upper()}...")
        try:
            ds = load_dataset(hf_path, split=split, streaming=True)
            samples = 0
            for row in ds:
                if samples >= self.max_samples:
                    break
                text = row.get(text_column, "")
                if not text:
                    continue
                
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                seq_len = 128
                for i in range(0, len(tokens) - seq_len + 1, seq_len):
                    if samples >= self.max_samples:
                        break
                        
                    chunk = tokens[i : i + seq_len]
                    chunk = [min(t, self.vocab_size - 1) for t in chunk]
                    
                    token_tensor = torch.tensor(chunk, dtype=torch.long, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        phasors = self.lifter(token_tensor)
                        bound_wavefront = torch.prod(phasors, dim=1).squeeze(0)
                        bound_wavefront = F.normalize(bound_wavefront, p=2, dim=-1)
                    
                    if self._insert_wavefront(quadrant_key, text[:200], bound_wavefront):
                        samples += 1
            print(f"[SUCCESS] Seeded {samples} axioms for {quadrant_key.upper()}.")
        except Exception as e:
            print(f"[!] Error pulling {hf_path}: {e}")

    def process_arc_data(self, arc_dir: str):
        print(f"[*] Processing ARC data from {arc_dir} for Quadrant EPSILON...")
        if not os.path.exists(arc_dir):
            print(f"[!] ARC directory {arc_dir} not found.")
            return
            
        samples = 0
        for fname in os.listdir(arc_dir):
            if samples >= self.max_samples:
                break
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(arc_dir, fname), 'r') as f:
                task = json.load(f)
            
            text_repr = json.dumps(task)
            tokens = self.tokenizer.encode(text_repr, add_special_tokens=False)
            seq_len = 128
            for i in range(0, len(tokens) - seq_len + 1, seq_len):
                if samples >= self.max_samples:
                    break
                    
                chunk = tokens[i : i + seq_len]
                chunk = [min(t, self.vocab_size - 1) for t in chunk]
                
                token_tensor = torch.tensor(chunk, dtype=torch.long, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    phasors = self.lifter(token_tensor)
                    bound_wavefront = torch.prod(phasors, dim=1).squeeze(0)
                    bound_wavefront = F.normalize(bound_wavefront, p=2, dim=-1)
                
                if self._insert_wavefront("epsilon", text_repr[:200], bound_wavefront):
                    samples += 1
                    
        print(f"[SUCCESS] Seeded {samples} axioms for EPSILON (ARC Constraints).")

if __name__ == "__main__":
    seeder = UniversalAxiomSeeder(max_samples=250)
    seeder.process_hf_stream("open-web-math/open-web-math", "train", "text", "alpha")
    seeder.process_hf_stream("HuggingFaceH4/CodeAlpaca_20K", "train", "completion", "beta")
    seeder.process_hf_stream("ag_news", "train", "text", "gamma") 
    seeder.process_hf_stream("HuggingFaceFW/fineweb-edu", "train", "text", "delta")
    seeder.process_arc_data("/root/HENRI/ARC-AGI-2-main/data/training")
    print("=== UNIVERSAL AXIOM SEEDING COMPLETE ===")
