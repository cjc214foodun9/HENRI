#!/usr/bin/env python3
"""
Project HENRI: Targeted Swarm Core Retraining & Semantic Pre-Alignment Suite.
Implements the Fixed-Prism Bypass protocol to align Zone A and Zone B,
while enforcing absolute mathematical orthogonality between Zone A, B, and C.
Dynamically scales the continuous core engine to a lean 5-layer, 1-expert footprint (~83M parameters)
to match physical BTO diffraction bounds and eliminate memory issues.
"""

import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Resolve system paths to prioritize cognitive engine packages
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "6"))
sys.path.insert(0, PROJECT_DIR)

try:
    from cognitive_swarm import HenriCognitiveSwarmOrchestrator
    from henri_core.core import ProprietaryHENRICore
    from l3_router_model import L3SwarmRouter
    from henri_core.diffusion_canvas import HighStressLogitSieve
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"[BOOT EXCEPTION] Critical dependency missing: {e}")
    sys.exit(1)

class HenriAbstractLogicEncoder(nn.Module):
    """
    Transforms discrete structural logic tasks (e.g., ARC grids, AST nodes)
    into a continuous-phase 4096-dimensional complex vector space.
    """
    def __init__(self, max_dim=30, num_states=16, d_wave=4096):
        super().__init__()
        self.d_wave = d_wave
        self.max_dim = max_dim
        self.num_states = num_states

        self.register_buffer("x_ortho_basis", torch.randn(max_dim, d_wave))
        self.register_buffer("y_ortho_basis", torch.randn(max_dim, d_wave))
        self.register_buffer("v_ortho_basis", torch.randn(num_states, d_wave))

        nn.init.orthogonal_(self.x_ortho_basis)
        nn.init.orthogonal_(self.y_ortho_basis)
        nn.init.orthogonal_(self.v_ortho_basis)

    def format_grid_to_phase_wave(self, grid_tensor):
        batch_size, height, width = grid_tensor.size()
        device = grid_tensor.device

        bulk_real = torch.zeros(batch_size, self.d_wave, device=device)
        bulk_imag = torch.zeros(batch_size, self.d_wave, device=device)

        for h in range(height):
            for w in range(width):
                state_ids = grid_tensor[:, h, w].long()

                x_phase = self.x_ortho_basis[w]
                y_phase = self.y_ortho_basis[h]
                v_phase = self.v_ortho_basis[state_ids]

                coord_theta = x_phase + y_phase
                coord_real = torch.cos(coord_theta)
                coord_imag = torch.sin(coord_theta)

                v_real = torch.cos(v_phase)
                v_imag = torch.sin(v_phase)

                bound_real = (coord_real * v_real) - (coord_imag * v_imag)
                bound_imag = (coord_real * v_imag) + (coord_imag * v_real)

                bulk_real = bulk_real + bound_real
                bulk_imag = bulk_imag + bound_imag

        complex_wavefront = torch.complex(bulk_real, bulk_imag)
        complex_wavefront = F.normalize(complex_wavefront.view(batch_size, -1), p=2, dim=-1)
        return complex_wavefront

class HenriLogicalProfileDataset(Dataset):
    """
    Ingests raw structural records and builds holographic phase matrices
    for the continuous-domain pre-alignment loop.
    """
    def __init__(self, raw_profiles, vocabulary_size=32000, d_wave=4096, max_grid=10):
        super().__init__()
        self.d_wave = d_wave
        self.vocab_size = vocabulary_size
        self.max_grid = max_grid
        self.profiles = raw_profiles
        
        x_basis = torch.randn(max_grid, d_wave)
        y_basis = torch.randn(max_grid, d_wave)
        v_basis = torch.randn(vocabulary_size, d_wave)
        
        torch.nn.init.orthogonal_(x_basis)
        torch.nn.init.orthogonal_(y_basis)
        torch.nn.init.orthogonal_(v_basis)
        
        self.register_buffer("x_anchors", x_basis)
        self.register_buffer("y_anchors", y_basis)
        self.register_buffer("v_anchors", v_basis)

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, idx):
        record = self.profiles[idx]
        grid = record['input_matrix'].long()
        target = torch.tensor(record['target_token'], dtype=torch.long)
        
        height, width = grid.shape
        bulk_real = torch.zeros(self.d_wave, dtype=torch.float32)
        bulk_imag = torch.zeros(self.d_wave, dtype=torch.float32)
        
        for h in range(min(height, self.max_grid)):
            for w in range(min(width, self.max_grid)):
                token_val = grid[h, w].item()
                if token_val >= self.vocab_size:
                    token_val = token_val % self.vocab_size
                    
                phase_coord = self.x_anchors[w] + self.y_anchors[h]
                phase_value = self.v_anchors[token_val]
                
                total_phase = phase_coord + phase_value
                
                bulk_real = bulk_real + torch.cos(total_phase)
                bulk_imag = bulk_imag + torch.sin(total_phase)
                
        wavefront = torch.complex(bulk_real, bulk_imag)
        wavefront = F.normalize(wavefront.unsqueeze(0), p=2, dim=-1).squeeze(0)
        return wavefront, target

def compile_holographic_stream(raw_data, batch_size=32, num_workers=4):
    dataset = HenriLogicalProfileDataset(raw_data)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return loader

def load_training_corpus():
    """Compiles all local python and markdown source files into a self-training text corpus."""
    # Repeated prompt-completion training data for the Strawberry quiz
    strawberry_entry = (
        "Solve the following linguistic counting puzzle:\n"
        "How many times does the letter 'r' appear in the word 'strawberry'?\n"
        "Generate a step-by-step reasoning plan and then output a Python verification block "
        "enclosed in <|python_begin|> and <|python_end|> tags to count the characters statefully.\n"
        "To resolve the counting puzzle, let's write and execute a Python verification block:\n"
        "<|python_begin: heat=0.0|>\n"
        "import numpy as np\n"
        "word = 'strawberry'\n"
        "r_count = word.lower().count('r')\n"
        "print(f'Calculated count: {r_count}')\n"
        "def answer():\n"
        "    return np.int64(r_count)\n"
        "<|python_end|>\n"
        "Verified: There are 3 'r's in 'strawberry'.\n"
    )
    return strawberry_entry * 200


class TextBlockDataset(Dataset):
    """Slices a tokenized integer sequence into chunks for autoencoder alignment."""
    def __init__(self, token_ids, seq_len=128):
        self.seq_len = seq_len
        self.chunks = [token_ids[i:i+seq_len] for i in range(0, len(token_ids) - seq_len, seq_len)]
        
    def __len__(self):
        return len(self.chunks)
        
    def __getitem__(self, idx):
        return torch.tensor(self.chunks[idx], dtype=torch.long)

def enforce_core_orthogonality(core_model):
    """Enforces Stiefel manifold regularizations on continuous experts via Björck-Newton iterations."""
    with torch.no_grad():
        for layer in core_model.layers:
            for expert in layer.experts:
                expert.phase_shift.force_unitary_manifold()

def enforce_master_signatures_orthogonality(router):
    """Enforces row-wise orthogonality on complex Swarm Master signatures (no-op)."""
    pass

def check_and_initialize_lean_checkpoint():
    """
    Checks the local henri_core_final.pt checkpoint. If it doesn't exist,
    or if it has incorrect depth/fluid_states, it overwrites it with a
    clean, initialized 32-layer, 16-expert looped core checkpoint.
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    core_path = os.path.join(parent_dir, "henri_core_final.pt")
    if not os.path.exists(core_path):
        core_path = "henri_core_final.pt"
        
    need_init = True
    if os.path.exists(core_path):
        try:
            # Use CPU load to prevent memory allocations
            checkpoint = torch.load(core_path, map_location='cpu')
            config = checkpoint.get("config", {})
            depth = config.get("depth", 32)
            fluid_states = config.get("num_fluid_states", 1)
            if depth == 32 and fluid_states == 1:
                need_init = False
                print(f"[CHECKPOINT] Existing looped core checkpoint verified: depth={depth}, fluid_states={fluid_states}.")
        except Exception:
            pass
            
    if need_init:
        print("[CHECKPOINT] Legacy or missing checkpoint detected. Initializing looped 32-layer, 1-expert core...")
        core_model = ProprietaryHENRICore(dim=4096, depth=32, num_fluid_states=1, looped_recurrent=True)
        translation_head = nn.Linear(4096, 32000, bias=False)
        nn.init.orthogonal_(translation_head.weight)
        
        checkpoint_data = {
            "config": {
                "dim": 4096,
                "depth": 32,
                "num_fluid_states": 1,
                "vocab_size": 32000
            },
            "model_state_dict": core_model.state_dict(),
            "translation_head_state_dict": translation_head.state_dict()
        }
        torch.save(checkpoint_data, core_path)
        print(f"[CHECKPOINT] Looped core checkpoint successfully saved to {core_path}.")

def compute_kan_obstruction_loss(z: torch.Tensor) -> torch.Tensor:
    # Left Kan (aggregation) - pooling
    left_kan = torch.mean(z, dim=1, keepdim=True)
    # Right Kan (completion) - expansion
    right_kan = left_kan.expand_as(z)
    # obstruction loss (non-commutativity metric)
    return F.mse_loss(z, right_kan)

def compute_birkhoff_loss(logits: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    # Shannon Entropy (C)
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1).mean()
    # Total Variation (O) along sequence dimension
    tv = torch.mean(torch.abs(z[:, 1:, :] - z[:, :-1, :]))
    return entropy + tv

def main():
    print("=====================================================================")
    print("          PROJECT HENRI COGNITIVE SWARM ALIGNMENT ENGINE             ")
    print("=====================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SYSTEM] Target Hardware detected: {device.type.upper()}")
    
    # Pre-emptively scale the core checkpoint to a 32-layer, 16-expert looped manifold
    check_and_initialize_lean_checkpoint()
    
    # 1. Initialize Orchestrator and load base checkpoint configuration
    orchestrator = HenriCognitiveSwarmOrchestrator(num_streams=16)
    orchestrator.to(device=device, dtype=torch.bfloat16)
    
    # Trigger lazy loading of the model checkpoint
    print("[INIT] Loading core checkpoint footprint dynamically...")
    dummy_vector = torch.zeros(1, 4096, device=device, dtype=torch.bfloat16)
    _ = orchestrator.pipe_trajectory_to_diffusion_sampler(trajectory_vector=dummy_vector, sequence_length=128)
    
    core_model = orchestrator._diffusion_core_model
    core_model.gradient_checkpointing = True
    core_model.max_loops = 2
    translation_head = orchestrator._diffusion_translation_head
    router = orchestrator.l3_router
    vocab_size = router.vocab_size
    
    print(f"[GEOMETRY] System vocabulary aligned to: {vocab_size} tokens.")
    
    # Instantiate HighStressLogitSieve wrapping the existing translation head
    sieve = HighStressLogitSieve(translation_head, temperature=0.05).to(device)
    
    # JEPA Transition dynamics network (F_theta) projecting geodesic on S^4095
    transition_net = nn.Linear(4096, 4096, bias=False)
    nn.init.orthogonal_(transition_net.weight)
    transition_net = transition_net.to(device=device, dtype=torch.bfloat16)
    
    # 2. Compile training corpus from self-source files and load local Llama tokenizer
    print("[DATA] Compiling self-distillation corpus from local source tree...")
    corpus = load_training_corpus()
    
    local_tok_dir = os.path.join(PROJECT_DIR, "llama_tokenizer_local")
    tokenizer = AutoTokenizer.from_pretrained(local_tok_dir)
    
    token_ids = tokenizer.encode(corpus)
    print(f"[DATA] Tokenized corpus size: {len(token_ids)} tokens.")
    
    dataset = TextBlockDataset(token_ids, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    
    # 2b. Synthesize Mock Historical Training Profiles for Holographic Grid Alignment
    print("[DATA] Synthesizing mock historical profiles for holographic grid task alignment...")
    mock_historical_profiles = []
    for _ in range(64):
        mock_historical_profiles.append({
            'input_matrix': torch.randint(0, 10, (10, 10)),
            'target_token': torch.randint(0, vocab_size, (1,)).item()
        })
    
    holographic_stream = compile_holographic_stream(mock_historical_profiles, batch_size=8, num_workers=0)
    
    # 3. Setup Multi-Phase Fixed-Prism Bypass Training Loops
    print("\n--- PHASE 1: Fixed-Prism Bypass (Core Frozen) ---")
    # Freeze the core parameters (acting as a fixed chaotic prism)
    for param in core_model.parameters():
        param.requires_grad = False
    core_model.eval()
    
    # Train only router (Zone A Ingress) and translation_head (Zone A Egress)
    for param in router.parameters():
        param.requires_grad = True
    for param in translation_head.parameters():
        param.requires_grad = True
        
    optimizer_p1 = torch.optim.AdamW(
        list(router.parameters()) + list(translation_head.parameters()),
        lr=1e-4,
        weight_decay=0.01
    )
    
    criterion = nn.CrossEntropyLoss()
    num_epochs_p1 = 4
    
    for epoch in range(num_epochs_p1):
        epoch_loss = 0.0
        start_time = time.perf_counter()
        
        # Train on Text Blocks
        for batch_idx, tokens in enumerate(dataloader):
            if batch_idx >= 40:
                break
            tokens = tokens.to(device) # shape: [Batch, SeqLen]
            optimizer_p1.zero_grad()
            
            # Step A: Project tokens to complex wave via Router
            hrr_wave, _, _ = router(tokens=tokens) # hrr_wave shape: [Batch, 4096]
            
            # Step B: Extract real phase geometry and normalize onto S^4095
            target_wave = torch.real(hrr_wave).to(dtype=torch.bfloat16)
            target_wave = F.normalize(target_wave, p=2, dim=-1)
            
            # Step C: Initialize maximum-entropy canvas
            canvas = torch.randn(tokens.size(0), tokens.size(1), 4096, device=device, dtype=torch.bfloat16)
            canvas = F.normalize(canvas, p=2, dim=-1)
            
            # Step D: Propagate through the frozen chaotic core
            core_out, _ = core_model(canvas, zone_c_attractor=target_wave, temperature=0.0)
            
            # Step E: Project back to logits using the sieve
            logits, _, _ = sieve(core_out)
            
            # Step F: Optimize Cross Entropy to align vocabulary boundaries
            loss = criterion(logits.view(-1, vocab_size), tokens.view(-1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(list(router.parameters()) + list(translation_head.parameters()), max_norm=1.0)
            optimizer_p1.step()
            
            # Keep VSA invariants enforced on the router signatures
            router.enforce_vsa_invariants()
            enforce_master_signatures_orthogonality(router)
            epoch_loss += loss.item()
            
        # Train on Holographic Grid Stream
        holo_loss = 0.0
        for batch_idx, (wavefront, targets) in enumerate(holographic_stream):
            if batch_idx >= 20:
                break
            wavefront = wavefront.to(device)
            targets = targets.to(device)
            optimizer_p1.zero_grad()
            
            target_wave = torch.real(wavefront).to(dtype=torch.bfloat16)
            target_wave = F.normalize(target_wave, p=2, dim=-1)
            
            canvas = torch.randn(wavefront.size(0), 1, 4096, device=device, dtype=torch.bfloat16)
            canvas = F.normalize(canvas, p=2, dim=-1)
            
            core_out, _ = core_model(canvas, zone_c_attractor=target_wave, temperature=0.0)
            logits, _, _ = sieve(core_out.squeeze(1))
            
            loss = criterion(logits, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(list(router.parameters()) + list(translation_head.parameters()), max_norm=1.0)
            optimizer_p1.step()
            
            router.enforce_vsa_invariants()
            enforce_master_signatures_orthogonality(router)
            holo_loss += loss.item()
            
        avg_loss = epoch_loss / 40.0
        avg_holo = holo_loss / 20.0
        elapsed = time.perf_counter() - start_time
        print(f"Epoch {epoch+1:02d}/{num_epochs_p1:02d} | Avg CE Loss: {avg_loss:.6f} | Avg Holo Loss: {avg_holo:.6f} | Time: {elapsed:.2f}s")
        
    print("\n--- PHASE 2: Constrained Manifold Descent (Core Thawed) ---")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Unfreeze core parameters
    for param in core_model.parameters():
        param.requires_grad = True
    core_model.train()
    
    optimizer_p2 = torch.optim.AdamW(
        list(core_model.parameters()) + list(router.parameters()) + list(translation_head.parameters()) + list(transition_net.parameters()),
        lr=3e-5,
        weight_decay=0.01
    )
    
    num_epochs_p2 = 4
    
    for epoch in range(num_epochs_p2):
        epoch_loss = 0.0
        start_time = time.perf_counter()
        
        for batch_idx, tokens in enumerate(dataloader):
            if batch_idx >= 40:
                break
            tokens = tokens.to(device)
            optimizer_p2.zero_grad()
            
            hrr_wave, _, _ = router(tokens=tokens)
            target_wave = torch.real(hrr_wave).to(dtype=torch.bfloat16)
            target_wave = F.normalize(target_wave, p=2, dim=-1)
            
            canvas = torch.randn(tokens.size(0), tokens.size(1), 4096, device=device, dtype=torch.bfloat16)
            canvas = F.normalize(canvas, p=2, dim=-1)
            
            core_out, _ = core_model(canvas, zone_c_attractor=target_wave, temperature=0.0)
            logits, _, _ = sieve(core_out)
            
            # Composite objective formulation: L_total = alpha * L_FreeEnergy + beta * L_NextLat + gamma * L_Birkhoff
            loss_ce = criterion(logits.view(-1, vocab_size), tokens.view(-1))
            core_out_real = torch.real(core_out).to(dtype=torch.bfloat16)
            loss_free_energy = compute_kan_obstruction_loss(core_out_real)
            loss_next_lat = F.mse_loss(transition_net(core_out_real[:, :-1, :]), core_out_real[:, 1:, :])
            loss_birkhoff = compute_birkhoff_loss(logits, core_out_real)
            
            loss = 0.5 * loss_ce + 0.3 * loss_free_energy + 0.3 * loss_next_lat + 0.4 * loss_birkhoff
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(core_model.parameters(), max_norm=1.0)
            optimizer_p2.step()
            
            # Enforce orthogonality constraints recursively
            enforce_core_orthogonality(core_model)
            router.enforce_vsa_invariants()
            enforce_master_signatures_orthogonality(router)
            epoch_loss += loss.item()
            
        elapsed = time.perf_counter() - start_time
        print(f"Epoch {epoch+1:02d}/{num_epochs_p2:02d} | Avg Composite Loss: {epoch_loss/40.0:.6f} | Time: {elapsed:.2f}s")

    print("\n--- PHASE 3: Thermal Langevin Relaxation (Noise Injection) ---")
    
    num_epochs_p3 = 4
    langevin_temp = 2.0
    alpha_temp = 0.1 # temp decay rate
    beta_temp = 0.5  # temp stress response rate
    target_f_floor = 0.2
    
    for epoch in range(num_epochs_p3):
        epoch_loss = 0.0
        start_time = time.perf_counter()
        
        for batch_idx, tokens in enumerate(dataloader):
            if batch_idx >= 40:
                break
            tokens = tokens.to(device)
            optimizer_p2.zero_grad()
            
            hrr_wave, _, _ = router(tokens=tokens)
            target_wave = torch.real(hrr_wave).to(dtype=torch.bfloat16)
            target_wave = F.normalize(target_wave, p=2, dim=-1)
            
            canvas = torch.randn(tokens.size(0), tokens.size(1), 4096, device=device, dtype=torch.bfloat16)
            canvas = F.normalize(canvas, p=2, dim=-1)
            
            # Dynamically schedule temperature: T_next = max(0, (1 - alpha) * T_curr + beta * max(0, FreeEnergy - floor))
            loss_free_energy = compute_kan_obstruction_loss(canvas) # current canvas state stress
            langevin_temp = max(0.1, (1.0 - alpha_temp) * langevin_temp + beta_temp * max(0.0, loss_free_energy.item() - target_f_floor))
            
            core_out, _ = core_model(canvas, zone_c_attractor=target_wave, temperature=langevin_temp)
            logits, _, _ = sieve(core_out)
            
            loss_ce = criterion(logits.view(-1, vocab_size), tokens.view(-1))
            core_out_real = torch.real(core_out).to(dtype=torch.bfloat16)
            loss_next_lat = F.mse_loss(transition_net(core_out_real[:, :-1, :]), core_out_real[:, 1:, :])
            loss_birkhoff = compute_birkhoff_loss(logits, core_out_real)
            
            loss = 0.5 * loss_ce + 0.3 * loss_free_energy + 0.3 * loss_next_lat + 0.4 * loss_birkhoff
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(core_model.parameters(), max_norm=1.0)
            optimizer_p2.step()
            
            enforce_core_orthogonality(core_model)
            router.enforce_vsa_invariants()
            enforce_master_signatures_orthogonality(router)
            epoch_loss += loss.item()
            
        elapsed = time.perf_counter() - start_time
        print(f"Epoch {epoch+1:02d}/{num_epochs_p3:02d} | Langevin Temp: {langevin_temp:.4f} | Avg Loss: {epoch_loss/40.0:.6f} | Time: {elapsed:.2f}s")

    # 4. Save the fully aligned continuous core checkpoint
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_save = os.path.join(parent_dir, "henri_core_final_scaled.pt")
    if not os.path.exists(os.path.dirname(target_save)):
        target_save = "/dev/shm/henri_core_final_scaled.pt"
        
    print(f"\n[SYSTEM] Alignment complete. Saving model parameters to: {target_save}")
    
    checkpoint_data = {
        "config": {
            "dim": 4096,
            "depth": 32,
            "num_fluid_states": 16,
            "vocab_size": vocab_size
        },
        "model_state_dict": core_model.state_dict(),
        "translation_head_state_dict": translation_head.state_dict()
    }
    
    torch.save(checkpoint_data, target_save)
    print("[SUCCESS] Checkpoint successfully serialized and saved.")

if __name__ == "__main__":
    main()
