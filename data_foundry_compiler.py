import os
import sys
import json
import re
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import ast

# Ensure output writes properly in Unicode
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Strict parameters matching the 8.59B Swarm model footprint
HRR_DIM = 4096
SEQUENCE_LENGTH = 512

class HolographicVectorLifter(nn.Module):
    """
    Transforms discrete programmatic tokens and continuous PDE scalar fields 
    into unit-modulus complex phase vectors on the S^4095 hypersphere.
    """
    def __init__(self, input_dim: int = SEQUENCE_LENGTH, bulk_dim: int = HRR_DIM):
        super().__init__()
        self.bulk_dim = bulk_dim
        # Instantiates an invariant orthogonal projection manifold
        self.projection_gate = nn.Linear(input_dim, bulk_dim, bias=False)
        nn.init.orthogonal_(self.projection_gate.weight)
        self.projection_gate.weight.requires_grad = False

    @torch.no_grad()
    def lift_to_hypersphere(self, numerical_array: np.ndarray) -> torch.Tensor:
        """
        Maps numeric structures to phase coordinates via FFT or projection.
        """
        device = self.projection_gate.weight.device
        x_raw = torch.from_numpy(numerical_array).float().to(device)
        
        # analytical FFT mapping for continuous physics invariants
        if x_raw.ndim == 1:
            x_fft = torch.fft.fft(x_raw, n=self.bulk_dim)
            phases = torch.angle(x_fft)
        else:
            lifted_real = self.projection_gate(x_raw)
            phases = torch.remainder(lifted_real, 2.0 * math.pi)
            
        complex_wave = torch.polar(torch.ones_like(phases), phases)
        return complex_wave

class MultiDomainCorpusCompiler:
    """
    Drives data compilation pipelines across text, code repositories, and mathematical matrices.
    """
    def __init__(self, output_hdf5_path: str):
        self.output_path = output_hdf5_path
        self.lifter = HolographicVectorLifter()
        
        # Load GPT-2 tokenizer as our reference BPE tokenizer
        try:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            print("[FOUNDRY] Reference BPE Tokenizer (GPT-2) loaded successfully.")
        except Exception as e:
            print(f"[FOUNDRY WARNING] Failed to load BPE tokenizer from Hugging Face: {e}. Falling back to mock character BPE.")
            self.tokenizer = None

        # Seed random dictionaries for AST node types and roles for UWE
        np.random.seed(42)
        self.node_type_vectors = {}
        self.role_vectors = {}

    def get_node_type_vector(self, node_type: str) -> np.ndarray:
        if node_type not in self.node_type_vectors:
            phases = np.random.uniform(-np.pi, np.pi, HRR_DIM)
            self.node_type_vectors[node_type] = np.cos(phases) + 1j * np.sin(phases)
        return self.node_type_vectors[node_type]

    def get_role_vector(self, role: str) -> np.ndarray:
        if role not in self.role_vectors:
            phases = np.random.uniform(-np.pi, np.pi, HRR_DIM)
            self.role_vectors[role] = np.cos(phases) + 1j * np.sin(phases)
        return self.role_vectors[role]

    def circular_convolution(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Computes frequency-domain circular convolution binding: u * v
        """
        u_fft = np.fft.fft(u)
        v_fft = np.fft.fft(v)
        res = np.fft.ifft(u_fft * v_fft)
        # Normalize
        norm = np.linalg.norm(res)
        if norm > 1e-8:
            res = res / norm
        return res

    def subword_token_framer(self, text_body: str, tokenizer_reference) -> np.ndarray:
        """
        Parses natural language and code into high-rank token arrays,
        aligning them to uniform target sequence windows.
        """
        if tokenizer_reference is not None:
            token_ids = tokenizer_reference.encode(text_body, add_special_tokens=True)
        else:
            # Fallback mock BPE character/word tokenization
            token_ids = [ord(char) % 32000 for char in text_body if ord(char) < 128]

        # Enforce static array chunk alignment
        if len(token_ids) < SEQUENCE_LENGTH:
            token_ids += [0] * (SEQUENCE_LENGTH - len(token_ids))
        else:
            token_ids = token_ids[:SEQUENCE_LENGTH]
            
        return np.array(token_ids, dtype=np.float32)

    def compute_ast_uwe_vector(self, source_code: str) -> np.ndarray:
        """
        Parses python source code into a clean AST, strips comments, and
        recursively builds a single Unitary Wave Embedding (UWE) 4096-D complex vector.
        """
        # Strip text comments
        clean_code = re.sub(r"#.*", "", source_code)
        clean_code = re.sub(r'""".*?"""', "", clean_code, flags=re.DOTALL)
        clean_code = re.sub(r"'''.*?'''", "", clean_code, flags=re.DOTALL)
        
        try:
            tree = ast.parse(clean_code)
        except Exception:
            # If parsing fails, fall back to subword-framer phase vector mapping
            tokens = self.subword_token_framer(clean_code, self.tokenizer)
            wave = self.lifter.lift_to_hypersphere(tokens)
            return wave.cpu().numpy()

        def recurse_node(node, depth=0) -> np.ndarray:
            if depth > 10:
                return self.get_node_type_vector("MaxDepth")
            node_type = type(node).__name__
            v_type = self.get_node_type_vector(node_type)
            
            # Superpose children bindings
            child_vectors = []
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    for idx, item in enumerate(value):
                        if isinstance(item, ast.AST):
                            v_child = recurse_node(item, depth + 1)
                            v_role = self.get_role_vector(f"{field}_{idx}")
                            child_vectors.append(self.circular_convolution(v_role, v_child))
                elif isinstance(value, ast.AST):
                    v_child = recurse_node(value, depth + 1)
                    v_role = self.get_role_vector(field)
                    child_vectors.append(self.circular_convolution(v_role, v_child))
            
            if child_vectors:
                superposed = v_type + sum(child_vectors)
                norm = np.linalg.norm(superposed)
                if norm > 1e-8:
                    superposed = superposed / norm
                return superposed
            return v_type

        return recurse_node(tree)

    def parse_cpp_block_structure(self, file_content: str) -> np.ndarray:
        """
        Parses non-python or custom coding blocks based on structural indentation.
        """
        lines = [line.strip() for line in file_content.split("\n") if line.strip() and not line.strip().startswith("//")]
        if not lines:
            return self.get_node_type_vector("EmptyCode")
            
        block_vectors = []
        for line in lines[:SEQUENCE_LENGTH]:
            # Each statement gets a role binding based on length/structure
            statement_vec = self.get_node_type_vector(hash(line) % 1000)
            role_vec = self.get_role_vector(f"line_{len(line) % 16}")
            block_vectors.append(self.circular_convolution(role_vec, statement_vec))
            
        superposed = sum(block_vectors)
        norm = np.linalg.norm(superposed)
        if norm > 1e-8:
            superposed = superposed / norm
        return superposed

    def compile_physics_packet(self, json_packet_path: str) -> np.ndarray:
        """
        Ingests continuous PDE matrix entries from data foundry packets.
        """
        with open(json_packet_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        raw_field = np.array(data.get("field_data", []), dtype=np.float32)
        flattened_field = raw_field.flatten()
        
        if len(flattened_field) < SEQUENCE_LENGTH:
            padded = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
            padded[:len(flattened_field)] = flattened_field
            return padded
        else:
            return flattened_field[:SEQUENCE_LENGTH]

    def process_and_shard_corpus(self, physics_folder: str, code_folder: str, heuristics_folder: str):
        """
        Assembles continuous physics, code syntax trees, and human heuristics into HDF5 datasets.
        """
        print(f"[FOUNDRY] Instantiating multi-domain corpus build at: {self.output_path}")
        
        with h5py.File(self.output_path, "w") as hf:
            physics_group = hf.create_group("continuous_physics")
            coding_group = hf.create_group("structural_code")
            heuristic_group = hf.create_group("human_heuristics")
            
            # 1. Processing Physics Domain
            idx = 0
            if os.path.exists(physics_folder):
                print("[FOUNDRY] Compiling continuous physics PDE vectors...")
                for file in os.listdir(physics_folder):
                    if file.endswith(".json"):
                        try:
                            vec_array = self.compile_physics_packet(os.path.join(physics_folder, file))
                            wave_tensor = self.lifter.lift_to_hypersphere(vec_array)
                            wave_data = np.stack([wave_tensor.real.cpu().numpy(), wave_tensor.imag.cpu().numpy()], axis=-1)
                            physics_group.create_dataset(f"wave_{idx}", data=wave_data, compression="gzip", compression_opts=4)
                            idx += 1
                        except Exception as e:
                            print(f"[WARN] Failed to process physics file {file}: {e}")
                print(f"  - Successfully encoded {idx} continuous physics wave configurations.")

            # 2. Processing Coding Domain
            idx = 0
            if os.path.exists(code_folder):
                print("[FOUNDRY] Compiling structural abstract syntax tree code waves...")
                for root, _, files in os.walk(code_folder):
                    for file in files:
                        if file.endswith((".py", ".cpp", ".txt", ".conf", ".md")):
                            try:
                                with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                                    content = f.read()
                                if file.endswith(".py"):
                                    wave_np = self.compute_ast_uwe_vector(content)
                                else:
                                    wave_np = self.parse_cpp_block_structure(content)
                                wave_data = np.stack([wave_np.real, wave_np.imag], axis=-1)
                                coding_group.create_dataset(f"wave_{idx}", data=wave_data, compression="gzip", compression_opts=4)
                                idx += 1
                            except Exception as e:
                                print(f"[WARN] Failed to process code file {file}: {e}")
                print(f"  - Successfully encoded {idx} structural programming structures.")

            # 3. Processing Heuristics Domain (Context-isolated 512-token envelopes)
            idx = 0
            if os.path.exists(heuristics_folder):
                print("[FOUNDRY] Compiling human intent narrow playbook arrays...")
                for file in os.listdir(heuristics_folder):
                    if file.endswith((".md", ".json")):
                        try:
                            with open(os.path.join(heuristics_folder, file), "r", encoding="utf-8", errors="ignore") as f:
                                content = f.read()
                            
                            # Shard into 512-token context envelopes
                            vec_array = self.subword_token_framer(content, self.tokenizer)
                            wave_tensor = self.lifter.lift_to_hypersphere(vec_array)
                            wave_data = np.stack([wave_tensor.real.cpu().numpy(), wave_tensor.imag.cpu().numpy()], axis=-1)
                            heuristic_group.create_dataset(f"wave_{idx}", data=wave_data, compression="gzip", compression_opts=4)
                            idx += 1
                        except Exception as e:
                            print(f"[WARN] Failed to process heuristic file {file}: {e}")
                print(f"  - Successfully encoded {idx} intent playbook wavefronts.")

        print("[SUCCESS] Multi-domain HDF5 data compilation complete. Pipeline ready for loop injection.")

if __name__ == "__main__":
    # Test execution stub using raw sources
    physics_dir = "foundry_physics"
    code_dir = "6"
    heuristics_dir = "archive/raw_sources"
    
    os.makedirs(physics_dir, exist_ok=True)
    
    # Generate mock physics packets if empty
    if not os.listdir(physics_dir):
        mock_pde_data = {"field_data": [[0.124, -0.412, 1.414], [0.992, -1.212, 0.041]]}
        with open(os.path.join(physics_dir, "packet_mock_poisson.json"), "w") as f:
            json.dump(mock_pde_data, f)
            
    compiler = MultiDomainCorpusCompiler(output_hdf5_path="henri_corpus_4096.h5")
    compiler.process_and_shard_corpus(
        physics_folder=physics_dir,
        code_folder=code_dir,
        heuristics_folder=heuristics_dir
    )
