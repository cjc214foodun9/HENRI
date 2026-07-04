import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import os
import sys
import time
from typing import Tuple

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from viscoelastic_swarm_core_shared_baseplate import ProprietaryHENRICore

from universal_thermodynamic_harness import OntologicalPhaseEncoder

class SwarmEgressTransducer(nn.Module):
    def __init__(self, dim: int = 4096, vocab_size: int = 32000):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.layer_norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, vocab_size, bias=False)
        
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        real_plane = wave.real
        stabilized = self.layer_norm(real_plane)
        logits = self.projection(stabilized)
        return logits

class SwarmProgramDataset(Dataset):
    def __init__(self, num_samples: int = 1000, seq_len: int = 128, vocab_size: int = 32000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.high_valence_tokens = [100, 204, 512, 1024, 2048, 12, 34, 56, 78, 90, 112, 234, 567, 1011, 2022]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = np.zeros(self.seq_len, dtype=np.int64)
        for i in range(self.seq_len):
            if i % 8 == 0:
                seq[i] = np.random.choice(self.high_valence_tokens[:5])
            elif i % 12 == 0:
                seq[i] = np.random.choice(self.high_valence_tokens[5:10])
            else:
                seq[i] = np.random.randint(0, self.vocab_size)
        return torch.tensor(seq, dtype=torch.long), torch.tensor(seq, dtype=torch.long)

class FreshHENRIOrchestrator(nn.Module):
    def __init__(self, vocab_size=32000, dim=4096, num_experts=16):
        super().__init__()
        self.l3_router = OntologicalPhaseEncoder(vocab_size=vocab_size, dim=dim)
        self.core = ProprietaryHENRICore(dim=dim, num_layers=32, num_experts=num_experts)

    def encode_phase(self, inputs):
        wave_real = self.l3_router(inputs)
        return wave_real + 1j * torch.zeros_like(wave_real)

    def apply_viscoelastic_gradient_updates(self, lr=1e-3):
        for layer in self.core.swarm_adapters:
            for expert in layer:
                expert.apply_viscoelastic_update(lr=lr)

def run_swarm_egress_compilation(output_path: str = "henri_fresh_core.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=========================================================================")
    print("      PROJECT HENRI: NESTED RECURSIVE SWARM COMPILATION MATRIX         ")
    print("=========================================================================")
    print(f"[*] Native hardware acceleration target: {device}")

    orchestrator = FreshHENRIOrchestrator(vocab_size=32000, dim=4096, num_experts=16)
    orchestrator.to(device)
    
    egress_head = SwarmEgressTransducer(dim=4096, vocab_size=32000)
    egress_head.to(device)

    dataset = SwarmProgramDataset(num_samples=1000, seq_len=128, vocab_size=32000)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    # PHASE 1: Fixed-Prism Swarm Pre-Alignment
    print("\n[PHASE 1] Initializing Fixed-Prism Swarm Pre-Alignment...")
    for param in orchestrator.core.parameters():
        param.requires_grad = False

    edge_parameters = list(orchestrator.l3_router.parameters()) + list(egress_head.parameters())
    optimizer_edge = torch.optim.AdamW(edge_parameters, lr=1e-3, weight_decay=1e-2)

    epochs_phase1 = 5
    for epoch in range(epochs_phase1):
        total_loss = 0.0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_edge.zero_grad()
            
            complex_wave = orchestrator.encode_phase(inputs)
            batch_size, seq_len, dim = complex_wave.shape
            flat_wave = complex_wave.view(batch_size * seq_len, dim)
            
            # Create 16 parallel initial wavefronts [16, B, Dim]
            swarm_wavefronts = flat_wave.unsqueeze(0).repeat(16, 1, 1)
            
            # Forward through ProprietaryHENRICore
            consensus_wave = orchestrator.core(swarm_wavefronts)
            
            # Colimit
            consensus_wave = consensus_wave.sum(dim=0)
            consensus_wave = F.normalize(consensus_wave.real, p=2, dim=-1) + 1j * F.normalize(consensus_wave.imag, p=2, dim=-1)
            
            logits = egress_head(consensus_wave)
            loss = criterion(logits, targets.view(-1))
            loss.backward()
            optimizer_edge.step()
            total_loss += loss.item()
        epoch_time = time.time() - start_time
        print(f" -> Epoch {epoch+1}/{epochs_phase1} | Avg Loss: {total_loss/len(data_loader):.6f} | Latency: {epoch_time:.2f}s")

    # PHASE 2: Constrained Swarm descent
    print("\n[PHASE 2] Initializing Constrained Swarm Descent...")
    for param in orchestrator.core.parameters():
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
            
            complex_wave = orchestrator.encode_phase(inputs)
            batch_size, seq_len, dim = complex_wave.shape
            flat_wave = complex_wave.view(batch_size * seq_len, dim)
            swarm_wavefronts = flat_wave.unsqueeze(0).repeat(16, 1, 1)
            
            consensus_wave = orchestrator.core(swarm_wavefronts).sum(dim=0)
            consensus_wave = F.normalize(consensus_wave.real, p=2, dim=-1) + 1j * F.normalize(consensus_wave.imag, p=2, dim=-1)
            
            # Sagnac Thermodynamic Veto Penalty
            target_wave = orchestrator.encode_phase(targets).view(batch_size * seq_len, dim)
            inner_product = torch.sum(consensus_wave * target_wave.conj(), dim=-1)
            sagnac_penalty = (1.0 - torch.abs(inner_product)).mean()
            
            logits = egress_head(consensus_wave)
            loss = criterion(logits, targets.view(-1)) + sagnac_penalty
            loss.backward()
            optimizer_all.step()
            total_loss += loss.item()
            
            orchestrator.core.bjorck_newton_orthonormalize(iterations=5)

        epoch_time = time.time() - start_time
        print(f" -> Epoch {epoch+1}/{epochs_phase2} | Avg Loss: {total_loss/len(data_loader):.6f} | Latency: {epoch_time:.2f}s")

    # PHASE 3: Viscoelastic Swarm Relaxation
    print("\n[PHASE 3] Initializing Viscoelastic Swarm Relaxation...")
    epochs_phase3 = 3
    for epoch in range(epochs_phase3):
        total_loss = 0.0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_all.zero_grad()
            
            complex_wave = orchestrator.encode_phase(inputs)
            batch_size, seq_len, dim = complex_wave.shape
            flat_wave = complex_wave.view(batch_size * seq_len, dim)
            swarm_wavefronts = flat_wave.unsqueeze(0).repeat(16, 1, 1)
            
            consensus_wave = orchestrator.core(swarm_wavefronts).sum(dim=0)
            consensus_wave = F.normalize(consensus_wave.real, p=2, dim=-1) + 1j * F.normalize(consensus_wave.imag, p=2, dim=-1)
            
            logits = egress_head(consensus_wave)
            loss = criterion(logits, targets.view(-1))
            loss.backward()
            optimizer_all.step()
            
            orchestrator.apply_viscoelastic_gradient_updates(lr=1e-3)
            orchestrator.core.bjorck_newton_orthonormalize(iterations=5)
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        print(f" -> Epoch {epoch+1}/{epochs_phase3} | Avg Loss: {total_loss/len(data_loader):.6f} | Latency: {epoch_time:.2f}s")

    print("\n[*] Serializing fresh core parameters...")
    torch.save({
        'core': orchestrator.core.state_dict(),
        'l3_router.weight': orchestrator.l3_router.state_dict(),
        'egress_head.weight': egress_head.state_dict(),
    }, output_path)
    print(f" -> Aligned swarm model written to: {output_path}")

if __name__ == "__main__":
    run_swarm_egress_compilation()