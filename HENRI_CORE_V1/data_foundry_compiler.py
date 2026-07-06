import torch
import ast
from torch.utils.data import IterableDataset
import itertools

try:
    from datasets import load_dataset
except ImportError:
    pass

class StreamingDataFoundry(IterableDataset):
    """
    On-the-fly streaming pipeline for compiling the 4 distinct Data Quadrants 
    (Physics, Code ASTs, Biophysics, Heuristics) directly into the Swarm Core 
    without requiring 11 Terabytes of local HDF5 storage.
    """
    def __init__(self, seq_len: int = 128, vocab_size: int = 32000, max_samples: int = None):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.max_samples = max_samples

    def _parse_python_ast(self, code_str: str) -> list:
        """ Parses raw code into a deterministic sequence of AST node types. """
        nodes = []
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                # Hash the AST node type into our 32000 vocabulary space
                node_hash = hash(type(node).__name__) % self.vocab_size
                nodes.append(node_hash)
        except Exception:
            pass
        return nodes

    def quadrant_alpha_stream(self):
        """ Quadrant Alpha: Formal Physics & Mathematics """
        # Simulating Lean 4 Mathlib ASTs / Dirichlet boundary matrices.
        # In production, connects to OpenFOAM & FEniCSx datasets.
        print("[FOUNDRY] Initiating Quadrant Alpha (Physics/Math) Stream...")
        while True:
            # Synthetic structural invariant tokens for now until full dataset mounts
            yield torch.randint(0, self.vocab_size, (self.seq_len,))

    def quadrant_beta_stream(self):
        """ Quadrant Beta: Structural Coding Graphs """
        print("[FOUNDRY] Initiating Quadrant Beta (Code ASTs) Stream...")
        try:
            # Stream Python ASTs from an ungated public dataset to avoid HF token requirements
            ds = load_dataset("flytech/python-codes-25k", split="train", streaming=True)
            for row in ds:
                code = row.get("output", row.get("text", row.get("content", "")))
                ast_tokens = self._parse_python_ast(code)
                # If the AST has enough depth, pad or slice to exact sequence length
                if len(ast_tokens) >= self.seq_len:
                    yield torch.tensor(ast_tokens[:self.seq_len], dtype=torch.long)
                elif len(ast_tokens) > 0:
                    padded = ast_tokens + [0] * (self.seq_len - len(ast_tokens))
                    yield torch.tensor(padded, dtype=torch.long)
        except Exception as e:
            print(f"[!] Quadrant Beta Stream failed (are `datasets` installed? Falling back): {e}")
            while True:
                yield torch.randint(0, self.vocab_size, (self.seq_len,))

    def quadrant_gamma_stream(self):
        """ Quadrant Gamma: Biophysical Dynamics & Scale-Free Cognition """
        # Simulating biological telemetry / Planarian voltage gradients
        print("[FOUNDRY] Initiating Quadrant Gamma (Biophysics) Stream...")
        while True:
            yield torch.randint(0, self.vocab_size, (self.seq_len,))

    def quadrant_delta_stream(self):
        """ Quadrant Delta: Human Heuristics & Cognitive Aesthetics """
        # Simulating JEPA Expert tool-use trajectories
        print("[FOUNDRY] Initiating Quadrant Delta (Heuristics) Stream...")
        while True:
            yield torch.randint(0, self.vocab_size, (self.seq_len,))

    def __iter__(self):
        """ Round-robin stream from all 4 Quadrants symmetrically. """
        streams = [
            self.quadrant_alpha_stream(),
            self.quadrant_beta_stream(),
            self.quadrant_gamma_stream(),
            self.quadrant_delta_stream()
        ]
        
        samples_yielded = 0
        for stream_idx in itertools.cycle(range(len(streams))):
            if self.max_samples and samples_yielded >= self.max_samples:
                break
                
            seq = next(streams[stream_idx])
            # Return sequence as both input and target (for auto-encoding the continuous geometry)
            yield seq, seq
            samples_yielded += 1

if __name__ == "__main__":
    # Test the pipeline
    foundry = StreamingDataFoundry(seq_len=64, max_samples=4)
    for i, (inp, tgt) in enumerate(foundry):
        print(f"Batch {i}: {inp.shape}")
