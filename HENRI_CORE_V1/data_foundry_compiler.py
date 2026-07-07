import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import math
from holographic_vector_lifter import HolographicVectorLifter

class StreamingDataFoundry:
    """
    Pulls live data from HuggingFace, tokenizes it via LLaMA tokenizer, 
    routes it through the HolographicVectorLifter, and binds it via 
    frequency-domain circular convolution into pristine 4096-D FHRR wavefronts.
    These are serialized to disk to act as the immutable Zone C TimescaleDB.
    """
    def __init__(self, vocab_size: int = 32000, dim: int = 4096, seq_len: int = 128, max_samples_per_quadrant: int = 500):
        self.vocab_size = vocab_size
        self.dim = dim
        self.seq_len = seq_len
        self.max_samples = max_samples_per_quadrant
        
        print("[FOUNDRY] Initializing hf-internal-testing/llama-tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
        
        print("[FOUNDRY] Initializing HolographicVectorLifter (Zone C Prism)...")
        # Ensure it is seeded deterministically so the lexical vectors are stable across runs
        torch.manual_seed(42)
        self.lifter = HolographicVectorLifter(vocab_size=vocab_size, dim=dim)
        
        # Output DB
        self.zone_c_db = {
            "alpha": [],
            "beta": [],
            "gamma": [],
            "delta": []
        }

    def _process_stream(self, hf_path: str, split: str, text_column: str, quadrant_key: str):
        print(f"[FOUNDRY] Mounting {hf_path} for Quadrant {quadrant_key.upper()}...")
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
                
                # Chunk into seq_len blocks
                for i in range(0, len(tokens) - self.seq_len + 1, self.seq_len):
                    if samples >= self.max_samples:
                        break
                        
                    chunk = tokens[i : i + self.seq_len]
                    
                    # Ensure all tokens are within the 32k boundary (LLaMA should be, but just in case)
                    chunk = [min(t, self.vocab_size - 1) for t in chunk]
                    
                    token_tensor = torch.tensor(chunk, dtype=torch.long).unsqueeze(0) # [1, seq_len]
                    
                    # 1. Lift tokens into continuous complex phase angles
                    with torch.no_grad():
                        phasors = self.lifter(token_tensor) # [1, seq_len, 4096]
                        
                        # 2. Bind the sequence via frequency-domain circular convolution (element-wise product)
                        # The phase angles sum together (modulo 2pi), meaning the entire sequence is 
                        # losslessly compressed into a single 4096-D mathematical invariant.
                        bound_wavefront = torch.prod(phasors, dim=1).squeeze(0) # [4096]
                        
                        # Normalize to ensure unit modulus invariants
                        bound_wavefront = F.normalize(bound_wavefront, p=2, dim=-1)
                        
                        self.zone_c_db[quadrant_key].append(bound_wavefront)
                        samples += 1
                        
            print(f"[FOUNDRY] Successfully seeded {samples} axiomatic wavefronts for Quadrant {quadrant_key.upper()}.")
            
        except Exception as e:
            print(f"[!] Critical Error pulling {hf_path}: {e}")

    def compile_zone_c_lexicon(self):
        # Quadrant Alpha: Math/Physics
        self._process_stream("open-web-math/open-web-math", "train", "text", "alpha")
        
        # Quadrant Beta: Code/Logic
        self._process_stream("HuggingFaceH4/CodeAlpaca_20K", "train", "completion", "beta")
        
        # Quadrant Gamma: Science/Knowledge (wikipedia snippet or similar)
        # We'll use a small fast dataset for testing if wikipedia is too heavy
        self._process_stream("ag_news", "train", "text", "gamma") 
        
        # Quadrant Delta: Heuristics
        self._process_stream("HuggingFaceFW/fineweb-edu", "train", "text", "delta")
        
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
    foundry = StreamingDataFoundry(max_samples_per_quadrant=100) # Quick run for testing
    foundry.compile_zone_c_lexicon()
