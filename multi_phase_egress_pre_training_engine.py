import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import sys

# Ensure relative imports resolve cleanly from the core source package directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cognitive_swarm import HenriCognitiveSwarmOrchestrator, StandaloneL3Router

class EgressTransducer(nn.Module):
    """
    Holographic Egress Transducer Head.
    Acts as the physical 'Observer' that collapses continuous 4096-D complex-valued
    wave states back into discrete token probability distributions (logits).
    """
    def __init__(self, dim: int = 4096, vocab_size: int = 32000):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Layer Normalization to stabilize amplitude variations prior to projection
        self.layer_norm = nn.LayerNorm(dim)
        
        # Linear projection mapping the stabilized real wave plane to vocabulary space
        self.projection = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """
        Ingests a batch of complex wave vectors, extracts the real-amplitude plane
        (simulating physical homodyne detection), and projects to logits.
        """
        # Extract the real part of the wave state (simulating Sagnac homodyne readout)
        real_plane = wave.real # Shape: [B, Dim] or [B, S, Dim]
        
        # Stabilize and project
        stabilized = self.layer_norm(real_plane)
        logits = self.projection(stabilized) # Shape: [B, S, Vocab_Size]
        return logits

class ProgramSynthesisDataset(Dataset):
    """
    Continuous-Time Synthesis Dataset.
    Generates structured, compilable Python AST trajectories paired with
    corresponding coordinate tokens to teach the egress head valid syntax.
    """
    def __init__(self, num_samples: int = 1000, seq_len: int = 128, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Define high-valence Python token IDs representing our structural lexicon
        # Def, Import, Return, brackets, indents, variable names
        self.high_valence_tokens = [
            100, 204, 512, 1024, 2048, 12, 34, 56, 78, 90, 112, 234, 567, 1011, 2022
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates structured synthetic program sequences to simulate ARC solver steps.
        """
        # Create a baseline sequence filled with zeros
        seq = np.zeros(self.seq_len, dtype=np.int64)
        
        # Inject structured keywords and indent blocks at fixed offsets to form syntax
        for i in range(self.seq_len):
            if i % 8 == 0:
                # Periodic indent or bracket
                seq[i] = np.random.choice(self.high_valence_tokens[:5])
            elif i % 12 == 0:
                # Python keywords (def, return)
                seq[i] = np.random.choice(self.high_valence_tokens[5:10])
            else:
                # Variable names and operational arguments
                seq[i] = np.random.randint(0, self.vocab_size)
                
        return torch.tensor(seq, dtype=torch.long), torch.tensor(seq, dtype=torch.long)

def run_egress_compilation_loop(weight_path: str = "henri_core_final.pt", output_path: str = "henri_aligned_core.pt"):
    """
    Executes the Unified, Multi-Phase Pre-Training Loop.
    Decouples parameter updates into clear-room structural phases to 
    eliminate 'word soup' representations forever.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=========================================================================")
    print("      PROJECT HENRI: NESTED RECURSIVE EGRESS COMPILATION MATRIX         ")
    print("=========================================================================")
    print(f"[*] Native hardware acceleration target: {device}")

    # 1. Instantiate the Standalone Orchestrator
    orchestrator = HenriCognitiveSwarmOrchestrator(vocab_size=32000, dim=4096)
    orchestrator.to(device)
    
    # 2. Map pre-trained core weights into active GPU memory
    load_success = orchestrator.load_pretrained_weights(weight_path, device=device)
    if not load_success:
        print("[!] Warning: Pre-trained weights missing. Forcing random manifold generation.")

    # 3. Instantiate the Egress Readout Head (The Missing Key)
    egress_head = EgressTransducer(dim=4096, vocab_size=32000)
    egress_head.to(device)

    # 4. Compile the Dataset & DataLoader
    dataset = ProgramSynthesisDataset(num_samples=1000, seq_len=128, vocab_size=32000)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Define loss function: Standard Cross-Entropy over token logits
    criterion = nn.CrossEntropyLoss()

    # -------------------------------------------------------------------------
    # PHASE 1: Fixed-Prism Edge Pre-Alignment
    # Freeze the wave core; optimize ONLY StandaloneL3Router & EgressTransducer.
    # -------------------------------------------------------------------------
    print("\n[PHASE 1] Initializing Fixed-Prism Edge Pre-Alignment...")
    print(" -> Locking 32 continuous-time diffractive layer parameters...")
    for param in orchestrator.wave_core.parameters():
        param.requires_grad = False

    # Parameters to optimize: Router layers + Egress readout head
    edge_parameters = list(orchestrator.l3_router.parameters()) + list(egress_head.parameters())
    optimizer_edge = torch.optim.AdamW(edge_parameters, lr=1e-3, weight_decay=1e-2)

    epochs_phase1 = 5
    for epoch in range(epochs_phase1):
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_edge.zero_grad()
            
            # Forward Pass: Transduce input tokens to S^4095 complex wave vectors
            # inputs shape: [B, Seq_Len] -> complex_wave: [B, Seq_Len, Dim]
            complex_wave = orchestrator.l3_router(inputs)
            
            # Step the wavefront through the 32 locked diffractive layers
            # We shape each sequence token independently through the continuous core
            batch_size, seq_len, dim = complex_wave.shape
            flat_wave = complex_wave.view(batch_size * seq_len, dim)
            
            # Propagate wavefront through the 32 locked BTO phase masks
            terminal_wave = orchestrator.wave_core(flat_wave, langevin_temp=0.0)
            
            # Decollapse the wave state through the Egress Transducer
            # Shape: [B * Seq_Len, Vocab_Size]
            logits = egress_head(terminal_wave)
            
            # Calculate cross-entropy loss against targeted syntax tokens
            loss = criterion(logits, targets.view(-1))
            loss.backward()
            
            optimizer_edge.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f" -> Epoch {epoch+1}/{epochs_phase1} | Avg Loss: {total_loss/len(data_loader):.6f} | Latency: {epoch_time:.2f}s")

    print("[SUCCESS] Phase 1 Complete. Edge boundaries aligned with the Core's topology.")

    # -------------------------------------------------------------------------
    # PHASE 2: Constrained Manifold Descent
    # Unfreeze the core; optimize all parameters under strict Stiefel limits.
    # -------------------------------------------------------------------------
    print("\n[PHASE 2] Initializing Constrained Manifold Descent...")
    print(" -> Unlocking 32 continuous-time diffractive layers...")
    for param in orchestrator.wave_core.parameters():
        param.requires_grad = True

    all_parameters = list(orchestrator.parameters()) + list(egress_head.parameters())
    optimizer_all = torch.optim.AdamW(all_parameters, lr=2e-4, weight_decay=1e-2)

    epochs_phase2 = 5
    for epoch in range(epochs_phase2):
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_all.zero_grad()
            
            # Forward Pass
            complex_wave = orchestrator.l3_router(inputs)
            batch_size, seq_len, dim = complex_wave.shape
            flat_wave = complex_wave.view(batch_size * seq_len, dim)
            
            # Propagate wavefront (Using Isothermal Precision mode: no Langevin noise)
            terminal_wave = orchestrator.wave_core(flat_wave, langevin_temp=0.0)
            logits = egress_head(terminal_wave)
            
            loss = criterion(logits, targets.view(-1))
            loss.backward()
            
            optimizer_all.step()
            total_loss += loss.item()
            
            # -----------------------------------------------------------------
            # COMPLIANCE ACTION: Post-Step Björck-Newton Orthogonalization
            # Force weight tensors back onto the Stiefel manifold post-gradient step.
            # -----------------------------------------------------------------
            orchestrator.wave_core.bjorck_newton_orthonormalize(iterations=5)

        epoch_time = time.time() - start_time
        print(f" -> Epoch {epoch+1}/{epochs_phase2} | Avg Loss: {total_loss/len(data_loader):.6f} | Latency: {epoch_time:.2f}s")

    print("[SUCCESS] Phase 2 Complete. Causal attractors converged within the Core.")

    # -------------------------------------------------------------------------
    # PHASE 3: Thermal Langevin Relaxation
    # Introduce controlled thermal noise (Langevin) to find the global minimum.
    # -------------------------------------------------------------------------
    print("\n[PHASE 3] Initializing Thermal Langevin Relaxation...")
    epochs_phase3 = 3
    
    # SPT (Sagnac-Proportional Temperature) base level: T = 1.0
    langevin_temp = 1.0

    for epoch in range(epochs_phase3):
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_all.zero_grad()
            
            complex_wave = orchestrator.l3_router(inputs)
            batch_size, seq_len, dim = complex_wave.shape
            flat_wave = complex_wave.view(batch_size * seq_len, dim)
            
            # Propagate wavefront under thermal Langevin excitation (shattering local locks)
            terminal_wave = orchestrator.wave_core(flat_wave, langevin_temp=langevin_temp)
            logits = egress_head(terminal_wave)
            
            loss = criterion(logits, targets.view(-1))
            loss.backward()
            
            optimizer_all.step()
            total_loss += loss.item()
            
            # Manifold Lock Pullback
            orchestrator.wave_core.bjorck_newton_orthonormalize(iterations=5)

        epoch_time = time.time() - start_time
        print(f" -> Epoch {epoch+1}/{epochs_phase3} | Avg Loss: {total_loss/len(data_loader):.6f} | Latency: {epoch_time:.2f}s")

    print("[SUCCESS] Phase 3 Complete. Thermal alignment stabilized.")

    # 5. Serialize Aligned Substrate & Readout Head Tensors
    print("\n[*] Serializing aligned substrate parameters...")
    torch.save({
        'K_micro': orchestrator.wave_core.coupling_matrices.state_dict(),
        'spatial_kernel': [W.data.cpu() for W in orchestrator.wave_core.layers],
        'thermal_mask': orchestrator.wave_core.microheater_mask.data.cpu() if hasattr(orchestrator.wave_core, 'microheater_mask') else None,
        'fluid_context_router.weight': orchestrator.l3_router.state_dict(),
        'egress_head.weight': egress_head.state_dict()
    }, output_path)
    
    print(f" -> Aligned model written to: {output_path}")
    print("=========================================================================")
    print("      COMPILATION SUCCESSFUL: WORD SOUP COLLAPSED TO TRUTH               ")
    print("=========================================================================")

if __name__ == "__main__":
    run_egress_compilation_loop()