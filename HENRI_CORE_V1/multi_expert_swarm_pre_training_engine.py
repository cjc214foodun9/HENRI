import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from viscoelastic_swarm_core_shared_baseplate import ProprietaryHENRICore

class ZoneCDataset(Dataset):
    """
    Loads the immutable Zone C Canonical Lexicon (TimescaleDB simulation).
    These are pristine 4096-D FHRR wavefronts containing world knowledge.
    """
    def __init__(self, db_path="zone_c_timescaledb.pt"):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Zone C Database not found at {db_path}. Please run data_foundry_compiler.py first.")
            
        db = torch.load(db_path)
        self.vectors = []
        for k in db:
            if len(db[k]) > 0:
                self.vectors.append(db[k])
                
        if len(self.vectors) == 0:
            raise ValueError("Zone C Database is empty!")
            
        self.vectors = torch.cat(self.vectors, dim=0) # [N, 4096] complex64

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return self.vectors[idx]

class FreshHENRIOrchestrator(nn.Module):
    def __init__(self, dim=4096, num_experts=16):
        super().__init__()
        # The core is now purely a Prism of Logic. No Tokenizer. No Egress Head.
        self.core = ProprietaryHENRICore(dim=dim, num_experts=num_experts)

    def apply_viscoelastic_gradient_updates(self, lr=1e-3):
        for expert in self.core.experts:
            if hasattr(expert, 'apply_viscoelastic_update'):
                expert.apply_viscoelastic_update(lr=lr)

def inject_langevin_heat(wavefront: torch.Tensor, temperature: float = 0.8) -> torch.Tensor:
    """Perturbs the pristine target wave into a chaotic initial state."""
    noise_real = torch.randn_like(wavefront.real) * temperature
    noise_imag = torch.randn_like(wavefront.imag) * temperature
    shaken = torch.complex(wavefront.real + noise_real, wavefront.imag + noise_imag)
    return F.normalize(shaken, p=2, dim=-1)

def infonce_phase_resonance_loss(output_wave: torch.Tensor, target_wave: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Supervised Contrastive Phase-Resonance Loss (InfoNCE).
    Maximizes cosine similarity between the rotated output wave and the true Zone C vector,
    while minimizing similarity with all other batch vectors.
    """
    # Extract structural real plane for similarity matrix
    out_real = F.normalize(output_wave.real, p=2, dim=-1)
    tgt_real = F.normalize(target_wave.real, p=2, dim=-1)
    
    # [Batch, Batch] similarity matrix
    logits = torch.matmul(out_real, tgt_real.T) / temperature
    
    # The true target is along the diagonal
    labels = torch.arange(logits.size(0), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def run_swarm_egress_compilation(output_path: str = "henri_fresh_core.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=========================================================================")
    print("      PROJECT HENRI: EPISTEMIC PHASE-RESONANCE TRAINING LOOP           ")
    print("=========================================================================")
    print(f"[*] Native hardware acceleration target: {device}")

    orchestrator = FreshHENRIOrchestrator(dim=4096, num_experts=16)
    orchestrator.to(device)
    
    try:
        dataset = ZoneCDataset("zone_c_timescaledb.pt")
        data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    except Exception as e:
        print(f"[FATAL] {e}")
        return

    # ONLY optimize the core experts to rotate continuous vectors.
    # No language modeling parameters are updated.
    optimizer = torch.optim.AdamW(orchestrator.core.parameters(), lr=1e-3, weight_decay=1e-2)

    epochs = 5
    print("\n[PHASE 1] Initiating Holographic Contrastive Learning (InfoNCE)...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        start_time = time.time()
        
        for batch_idx, target_waves in enumerate(data_loader):
            target_waves = target_waves.to(device)
            optimizer.zero_grad()
            
            # 1. Generate chaotic input state via Langevin Heat
            chaotic_input = inject_langevin_heat(target_waves, temperature=1.0)
            
            # 2. Rotate wavefront through the Swarm Prism (which handles internal routing and superposition)
            telemetry = orchestrator.core(chaotic_input, temperature=1.0)
            
            # 3. Extract the superposed consensus wave
            consensus_wave = telemetry["resolved_wave"]
            
            # 5. Compute Phase-Resonance Loss (InfoNCE)
            loss = infonce_phase_resonance_loss(consensus_wave, target_waves)
            
            loss.backward()
            
            # Prevent cascading phase inversions
            torch.nn.utils.clip_grad_norm_(orchestrator.core.parameters(), 1.0)
            
            optimizer.step()
            orchestrator.apply_viscoelastic_gradient_updates(lr=1e-3)
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(data_loader)} | InfoNCE Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        epoch_time = time.time() - start_time
        print(f"[*] Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")
        
    print("\n[SUCCESS] Absolute Epiplexity Rotational Mastery Achieved.")
    torch.save(orchestrator.core.state_dict(), output_path)
    print(f"[*] Pure Physics Matrix saved to {output_path}")

if __name__ == "__main__":
    run_swarm_egress_compilation()