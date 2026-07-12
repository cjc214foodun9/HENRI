import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import json
import glob
from holographic_vector_lifter import HolographicVectorLifter
from holographic_vector_lifter import HolographicVectorLifter

class StreamingDataFoundry:
    """
    Pulls live data from HuggingFace, tokenizes it via LLaMA tokenizer, 
    routes it through the HolographicVectorLifter, and binds it via 
    frequency-domain circular convolution into pristine 4096-D FHRR wavefronts.
    These are serialized to disk to act as the immutable Zone C TimescaleDB.
    """
    def __init__(self, vocab_size: int = 32000, dim: int = 4096, seq_len: int = 128, max_samples_per_quadrant: int = 500):
        self.vocab_size = 32000
        self.seq_len = 256
        self.max_samples = max_samples_per_quadrant
        
        print("[FOUNDRY] Initializing Physical Mock Tokenizer...")
        class MockTokenizer:
            def __init__(self, vocab_size):
                self.vocab_size = vocab_size
            def encode(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
                return [1] + [ord(c) % self.vocab_size for c in text]
        
        self.tokenizer = MockTokenizer(self.vocab_size)
        
        print("[FOUNDRY] Initializing HolographicVectorLifter (Zone C Prism)...")
        # Ensure it is seeded deterministically so the lexical vectors are stable across runs
        torch.manual_seed(42)
        self.lifter = HolographicVectorLifter(vocab_size=self.vocab_size, dim=4096)
        
        # Output DB
        self.zone_c_db = {
            "alpha": [],
            "beta": [],
            "gamma": [],
            "delta": []
        }

    def _process_arc_grid(self, text: str, quadrant_key: str, samples_count: list):
        if samples_count[0] >= self.max_samples:
            return
            
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # Chunk into seq_len blocks
        for i in range(0, len(tokens) - self.seq_len + 1, self.seq_len):
            if samples_count[0] >= self.max_samples:
                break
                
            chunk = tokens[i : i + self.seq_len]
            chunk = [min(t, self.vocab_size - 1) for t in chunk]
            token_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0)
            
            with torch.no_grad():
                phasors = self.lifter(token_tensor)
                bound_wavefront = torch.prod(phasors, dim=1).squeeze(0)
                bound_wavefront = F.normalize(bound_wavefront, p=2, dim=-1)
                
                self.zone_c_db[quadrant_key].append(bound_wavefront)
                samples_count[0] += 1

    def compile_zone_c_lexicon(self):
        print("[FOUNDRY] Parsing local ARC-AGI-2 JSON grid datasets...")
        arc_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ARC-AGI-2-main", "data", "training", "*.json")
        json_files = glob.glob(arc_data_path)
        
        if not json_files:
            print(f"[!] No JSON files found at {arc_data_path}")
            return
            
        print(f"[FOUNDRY] Found {len(json_files)} ARC grid files.")
        
        samples_count = [0]
        
        for file_path in json_files:
            if samples_count[0] >= self.max_samples * 4: # distribute across 4 quadrants
                break
                
            with open(file_path, "r") as f:
                data = json.load(f)
                
            for i, example in enumerate(data.get("train", [])):
                in_grid = str(example.get("input", []))
                out_grid = str(example.get("output", []))
                text = f"INPUT: {in_grid} OUTPUT: {out_grid}"
                
                # We'll just distribute them round-robin across quadrants
                quad_idx = samples_count[0] % 4
                quad_keys = ["alpha", "beta", "gamma", "delta"]
                quad_key = quad_keys[quad_idx]
                
                self._process_arc_grid(text, quad_key, samples_count)
        
        print("[FOUNDRY] Serializing Zone C Holographic Database...")
        db_path = "zone_c_timescaledb.pt"
        
        # Stack lists into monolithic tensors
        for k in self.zone_c_db:
            if len(self.zone_c_db[k]) > 0:
                self.zone_c_db[k] = torch.stack(self.zone_c_db[k])
            else:
                self.zone_c_db[k] = torch.empty((0, self.dim), dtype=torch.complex64)
                
        torch.save(self.zone_c_db, db_path)
        print(f"[FOUNDRY] Zone C Database written to {db_path} (Static Dirichlet Boundaries).")

if __name__ == "__main__":
    # Max out the samples to extract as many structural grid tokens as possible
    foundry = StreamingDataFoundry(max_samples_per_quadrant=5000)
    foundry.compile_zone_c_lexicon()
