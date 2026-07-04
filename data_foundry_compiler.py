#!/usr/bin/env python3
"""
Project HENRI: Core Data Foundry Compiler
Parses mathematical proofs (Domain A), structural execution graphs (Domain B),
and active inference trajectories (Domain C), translating them into unrolled
complex wave vectors on the unit hypersphere, and serializing them to HDF5 format.
"""

import os
import sys
import re
import json
import torch
import numpy as np
import h5py
import shutil

# Resolve project directories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "6"))
sys.path.insert(0, PROJECT_DIR)

from transformers import AutoTokenizer
from cognitive_swarm import HenriCognitiveSwarmOrchestrator

def clean_python_code(code_str):
    """Strips comments, docstrings, and literal strings to isolate AST structures."""
    # Strip single line comments
    code_str = re.sub(r'#.*', '', code_str)
    # Strip triple-quote docstrings
    code_str = re.sub(r'\"\"\"[\s\S]*?\"\"\"', '', code_str)
    code_str = re.sub(r'\'\'\'[\s\S]*?\'\'\'', '', code_str)
    # Strip literal string noise but keep structure delimiters
    code_str = re.sub(r'\"[^\"]*\"', '""', code_str)
    code_str = re.sub(r'\'[^\']*\'', "''", code_str)
    # Normalize whitespaces
    lines = [line.rstrip() for line in code_str.splitlines() if line.strip()]
    return "\n".join(lines)

def complex_to_unrolled(tensor):
    """Converts a complex tensor to a float32 numpy array with trailing dimension 2 (real, imag)."""
    # tensor shape: [seq, 4096] complex or [4096] complex
    real = tensor.real.cpu().numpy().astype(np.float32)
    imag = tensor.imag.cpu().numpy().astype(np.float32)
    return np.stack([real, imag], axis=-1)

def main():
    print("=====================================================================")
    print("                PROJECT HENRI: DATA FOUNDRY COMPILER                 ")
    print("=====================================================================")
    
    # 1. Initialize tokenizer and swarm model
    print("[INIT] Loading Llama local fast tokenizer...")
    local_tok_dir = os.path.join(PROJECT_DIR, "llama_tokenizer_local")
    tokenizer = AutoTokenizer.from_pretrained(local_tok_dir)
    
    print("[INIT] Booting Orchestrator to access UWE phase router...")
    # Seed torch to ensure deterministic quasi-orthogonal basis projection
    torch.manual_seed(42)
    orchestrator = HenriCognitiveSwarmOrchestrator(num_streams=16)
    router = orchestrator.l3_router
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router.to(device)
    router.eval()
    
    # --- DOMAIN A: Formal Logical Invariants (Lean 4 ASTs / Proofs) ---
    print("\n[DOMAIN A] Curating Lean 4 ASTs / Mathematical Proofs...")
    proofs_file = os.path.join(PROJECT_DIR, "Project HENRI Mathematical and Physical Proofs.md")
    if not os.path.exists(proofs_file):
        proofs_file = os.path.join(os.path.dirname(PROJECT_DIR), "Project HENRI Mathematical and Physical Proofs.md")
    
    domain_a_wavefronts = []
    domain_a_targets = []
    
    if os.path.exists(proofs_file):
        with open(proofs_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split proofs file into logical paragraph/equation chunks
        chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        print(f" [+] Found {len(chunks)} formal proof text blocks.")
        
        for idx, chunk in enumerate(chunks):
            # Tokenize chunk
            token_ids = tokenizer.encode(chunk)
            # Pad/truncate to seq_len 128
            if len(token_ids) > 128:
                token_ids_in = token_ids[:128]
                token_ids_tgt = token_ids[1:129]
            else:
                token_ids_in = token_ids + [0] * (128 - len(token_ids))
                token_ids_tgt = token_ids[1:] + [0] * (129 - len(token_ids))
            
            with torch.no_grad():
                tensor_in = torch.tensor(token_ids_in, dtype=torch.long, device=device).unsqueeze(0)
                tensor_tgt = torch.tensor(token_ids_tgt, dtype=torch.long, device=device).unsqueeze(0)
                
                wave_in = router.text_to_wave(tensor_in).squeeze(0)
                wave_tgt = router.text_to_wave(tensor_tgt).squeeze(0)
                
                # Global norm preservation onto S^4095
                wave_in = torch.nn.functional.normalize(wave_in, p=2, dim=-1)
                wave_tgt = torch.nn.functional.normalize(wave_tgt, p=2, dim=-1)
                
                domain_a_wavefronts.append(complex_to_unrolled(wave_in))
                domain_a_targets.append(complex_to_unrolled(wave_tgt))
    else:
        print(" [!] Warning: Mathematical proofs file not found. Generating mock Domain A data.")
        # Generates fallback mock entries
        for _ in range(50):
            mock_in = torch.randn(4096, dtype=torch.complex64, device=device)
            mock_tgt = torch.randn(4096, dtype=torch.complex64, device=device)
            domain_a_wavefronts.append(complex_to_unrolled(torch.nn.functional.normalize(mock_in, p=2, dim=-1)))
            domain_a_targets.append(complex_to_unrolled(torch.nn.functional.normalize(mock_tgt, p=2, dim=-1)))
            
    # --- DOMAIN B: Structural Execution Graphs (Python AST Code) ---
    print("\n[DOMAIN B] Stripping Python Source Code AST Graphs...")
    code_dir = os.path.join(PROJECT_DIR, "6")
    
    domain_b_wavefronts = []
    domain_b_targets = []
    
    code_files = []
    if os.path.exists(code_dir):
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                if file.endswith(".py"):
                    code_files.append(os.path.join(root, file))
                    
    print(f" [+] Found {len(code_files)} local Python files in core package registry.")
    
    for c_file in code_files:
        with open(c_file, "r", encoding="utf-8") as f:
            code_content = f.read()
        
        # Strip literal variable name/docstring noise (AST abstraction)
        ast_code = clean_python_code(code_content)
        
        # Split by classes/defs or chunks of lines
        code_chunks = [chunk.strip() for chunk in ast_code.split("\n\n") if chunk.strip()]
        
        for chunk in code_chunks:
            token_ids = tokenizer.encode(chunk)
            if len(token_ids) > 128:
                token_ids_in = token_ids[:128]
                token_ids_tgt = token_ids[1:129]
            else:
                token_ids_in = token_ids + [0] * (128 - len(token_ids))
                token_ids_tgt = token_ids[1:] + [0] * (129 - len(token_ids))
                
            with torch.no_grad():
                tensor_in = torch.tensor(token_ids_in, dtype=torch.long, device=device).unsqueeze(0)
                tensor_tgt = torch.tensor(token_ids_tgt, dtype=torch.long, device=device).unsqueeze(0)
                
                wave_in = router.text_to_wave(tensor_in).squeeze(0)
                wave_tgt = router.text_to_wave(tensor_tgt).squeeze(0)
                
                wave_in = torch.nn.functional.normalize(wave_in, p=2, dim=-1)
                wave_tgt = torch.nn.functional.normalize(wave_tgt, p=2, dim=-1)
                
                domain_b_wavefronts.append(complex_to_unrolled(wave_in))
                domain_b_targets.append(complex_to_unrolled(wave_tgt))
                
    # --- DOMAIN C: Active Inference Trajectories (UniversalREPL / JEPA logs) ---
    print("\n[DOMAIN C] Packaging Active Inference Trajectories...")
    summary_file = os.path.join(PROJECT_DIR, "results", "distillation_summary.json")
    if not os.path.exists(summary_file):
        summary_file = os.path.join(os.path.dirname(PROJECT_DIR), "results", "distillation_summary.json")
    
    domain_c_trajectories = []
    domain_c_belief_states = []
    
    seq_length = 16
    
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            run_stats = json.load(f)
            
        print(f" [+] Found {len(run_stats)} historical task traces in distillation summary ledger.")
        
        for task_id, epochs in run_stats.items():
            # Build execution sequence: prompt -> revision steps -> final verification
            steps_text = []
            steps_text.append(f"Prompt: Solve ARC task {task_id}")
            
            for epoch_name in ["epoch1", "epoch2", "epoch3"]:
                epoch_data = epochs.get(epoch_name)
                if epoch_data:
                    steps_text.append(f"Execution {epoch_name}: Status={epoch_data.get('status')}, Resonance={epoch_data.get('resonance')}")
            
            # Pad steps list to seq_length
            if len(steps_text) < seq_length:
                steps_text = steps_text + [steps_text[-1]] * (seq_length - len(steps_text))
            else:
                steps_text = steps_text[:seq_length]
                
            trajectory_waves = []
            belief_waves = []
            
            for step_txt in steps_text:
                token_ids = tokenizer.encode(step_txt)
                if len(token_ids) > 128:
                    token_ids = token_ids[:128]
                else:
                    token_ids = token_ids + [0] * (128 - len(token_ids))
                    
                with torch.no_grad():
                    tokens_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
                    wave = router.text_to_wave(tokens_tensor).squeeze(0)
                    wave = torch.nn.functional.normalize(wave, p=2, dim=-1)
                    
                    trajectory_waves.append(complex_to_unrolled(wave))
                    # Future/Target belief state: shifted step or correct task invariant wave
                    belief_waves.append(complex_to_unrolled(wave))
            
            domain_c_trajectories.append(np.stack(trajectory_waves, axis=0))
            domain_c_belief_states.append(np.stack(belief_waves, axis=0))
    else:
        print(" [!] Warning: Distillation summary not found. Synthesizing mock Domain C trajectories.")
        for _ in range(30):
            trajectory_waves = []
            belief_waves = []
            for _ in range(seq_length):
                mock_t = torch.randn(4096, dtype=torch.complex64, device=device)
                mock_b = torch.randn(4096, dtype=torch.complex64, device=device)
                trajectory_waves.append(complex_to_unrolled(torch.nn.functional.normalize(mock_t, p=2, dim=-1)))
                belief_waves.append(complex_to_unrolled(torch.nn.functional.normalize(mock_b, p=2, dim=-1)))
            domain_c_trajectories.append(np.stack(trajectory_waves, axis=0))
            domain_c_belief_states.append(np.stack(belief_waves, axis=0))
            
    # Convert lists to numpy arrays
    A_wave = np.array(domain_a_wavefronts, dtype=np.float32)
    A_tgt = np.array(domain_a_targets, dtype=np.float32)
    B_wave = np.array(domain_b_wavefronts, dtype=np.float32)
    B_tgt = np.array(domain_b_targets, dtype=np.float32)
    C_traj = np.array(domain_c_trajectories, dtype=np.float32)
    C_belief = np.array(domain_c_belief_states, dtype=np.float32)
    
    # 2. Serialize to chunked HDF5 File
    output_dir = os.path.join(PROJECT_DIR, "data")
    os.makedirs(output_dir, exist_ok=True)
    
    h5_path = os.path.join(output_dir, "henri_corpus_4096.h5")
    print(f"\n[HDF5] Writing compiled corpus to {h5_path}...")
    
    with h5py.File(h5_path, "w") as h5_file:
        # Domain A Group
        grp_a = h5_file.create_group("domain_a")
        grp_a.create_dataset("wavefronts", data=A_wave, chunks=(min(32, len(A_wave)), 4096, 2), compression="gzip")
        grp_a.create_dataset("targets", data=A_tgt, chunks=(min(32, len(A_tgt)), 4096, 2), compression="gzip")
        print(f" [+] Domain A datasets compiled: {A_wave.shape} wavefronts.")
        
        # Domain B Group
        grp_b = h5_file.create_group("domain_b")
        grp_b.create_dataset("wavefronts", data=B_wave, chunks=(min(32, len(B_wave)), 4096, 2), compression="gzip")
        grp_b.create_dataset("targets", data=B_tgt, chunks=(min(32, len(B_tgt)), 4096, 2), compression="gzip")
        print(f" [+] Domain B datasets compiled: {B_wave.shape} AST graphs.")
        
        # Domain C Group
        grp_c = h5_file.create_group("domain_c")
        grp_c.create_dataset("trajectories", data=C_traj, chunks=(min(16, len(C_traj)), seq_length, 4096, 2), compression="gzip")
        grp_c.create_dataset("belief_states", data=C_belief, chunks=(min(16, len(C_belief)), seq_length, 4096, 2), compression="gzip")
        print(f" [+] Domain C datasets compiled: {C_traj.shape} JEPA trajectories.")
        
    # Copy to target training file names and target paths
    h5_target = os.path.join(output_dir, "henri_structural_corpus.h5")
    shutil.copy(h5_path, h5_target)
    print(f"[HDF5] Copied to structural target: {h5_target}")
    
    # Also create the parent 'data' directory if needed and copy there
    parent_data_dir = os.path.join(os.path.dirname(PROJECT_DIR), "data")
    os.makedirs(parent_data_dir, exist_ok=True)
    parent_h5_target = os.path.join(parent_data_dir, "henri_structural_corpus.h5")
    shutil.copy(h5_path, parent_h5_target)
    print(f"[HDF5] Copied to parent workspace path: {parent_h5_target}")
    
    print("\n[SUCCESS] HENRI HDF5 Core Data Foundry compilation successfully completed!")
    print("=====================================================================")

if __name__ == "__main__":
    main()
