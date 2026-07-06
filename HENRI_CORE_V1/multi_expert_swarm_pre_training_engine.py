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
from triton_fused_physics import triton_fused_superposition, triton_fused_sagnac_veto_penalty
from triton_fused_physics import triton_fused_superposition, triton_fused_sagnac_veto_penalty
from holographic_egress_high_stress_logit_sieve import HolographicPhaseTransducer
from holographic_vector_lifter import HolographicVectorLifter
from data_foundry_compiler import StreamingDataFoundry

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

# SwarmProgramDataset removed in favor of StreamingDataFoundry

class FreshHENRIOrchestrator(nn.Module):
    def __init__(self, vocab_size=32000, dim=4096, num_experts=16):
        super().__init__()
        self.l3_router = HolographicVectorLifter(vocab_size=vocab_size, dim=dim)
        self.core = ProprietaryHENRICore(dim=dim, num_layers=32, num_experts=num_experts)

    def encode_phase(self, inputs):
        wave_real = self.l3_router(inputs, relational_edges=None)
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
    
    egress_head = HolographicPhaseTransducer(d_wave=4096, vocab_size=32000)
    egress_head.to(device)

    dataset = StreamingDataFoundry(seq_len=64, vocab_size=32000, max_samples=1000)
    data_loader = DataLoader(dataset, batch_size=1)
    criterion = nn.CrossEntropyLoss()
    
    steps_per_epoch = 1000 # Fix for IterableDataset length

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
            if complex_wave.dim() == 2:
                complex_wave = complex_wave.unsqueeze(1).repeat(1, inputs.size(1), 1)
            batch_size, seq_len, dim = complex_wave.shape
            flat_wave = complex_wave.view(batch_size * seq_len, dim)
            
            # Create 16 parallel initial wavefronts [16, B, Dim]
            swarm_wavefronts = flat_wave.unsqueeze(0).repeat(16, 1, 1)
            
            # Forward through ProprietaryHENRICore and Triton Fused Superposition
            expert_waves = orchestrator.core(swarm_wavefronts)
            consensus_wave = triton_fused_superposition(expert_waves)
            
            logits = egress_head(consensus_wave)
            loss = criterion(logits, targets.view(-1))
            loss.backward()
            optimizer_edge.step()
            total_loss += loss.item()
            if batch_idx >= steps_per_epoch - 1: break
            
        epoch_time = time.time() - start_time
        total_loss /= steps_per_epoch
        print(f" -> Epoch {epoch+1}/{epochs_phase1} | Avg Loss: {total_loss:.6f} | Latency: {epoch_time:.2f}s")

    # PHASE 2: Constrained Swarm descent
    print("\n[PHASE 2] Initializing Constrained Swarm Descent...")
    del optimizer_edge
    torch.cuda.empty_cache()
    
    for param in orchestrator.core.parameters():
        param.requires_grad = True

    all_parameters = list(orchestrator.parameters()) + list(egress_head.parameters())
    optimizer_all = torch.optim.AdamW(all_parameters, lr=2e-4, weight_decay=1e-2, foreach=False)

    epochs_phase2 = 5
    for epoch in range(epochs_phase2):
        total_loss = 0.0
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer_all.zero_grad()
            
            complex_wave = orchestrator.encode_phase(inputs)
            if complex_wave.dim() == 2:
                complex_wave = complex_wave.unsqueeze(1).repeat(1, inputs.size(1), 1)
            batch_size, seq_len, dim = complex_wave.shape
            flat_wave = complex_wave.view(batch_size * seq_len, dim)
            swarm_wavefronts = flat_wave.unsqueeze(0).repeat(16, 1, 1)
            
            expert_waves = orchestrator.core(swarm_wavefronts)
            consensus_wave = triton_fused_superposition(expert_waves)
            
            # Sagnac Thermodynamic Veto Penalty
            target_wave_raw = orchestrator.encode_phase(targets)
            if target_wave_raw.dim() == 2:
                target_wave_raw = target_wave_raw.unsqueeze(1).repeat(1, targets.size(1), 1)
            target_wave = target_wave_raw.view(batch_size * seq_len, dim)
            
            sagnac_penalty = triton_fused_sagnac_veto_penalty(consensus_wave, target_wave)
            
            logits = egress_head(consensus_wave)
            loss = criterion(logits, targets.view(-1)) + sagnac_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
            optimizer_all.step()
            total_loss += loss.item()
            if batch_idx >= steps_per_epoch - 1: break

        orchestrator.core.bjorck_newton_orthonormalize(iterations=5)
        orchestrator.l3_router.project_to_unit_modulus()
        epoch_time = time.time() - start_time
        total_loss /= steps_per_epoch
        print(f" -> Epoch {epoch+1}/{epochs_phase2} | Avg Loss: {total_loss:.6f} | Latency: {epoch_time:.2f}s")

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
            if complex_wave.dim() == 2:
                complex_wave = complex_wave.unsqueeze(1).repeat(1, inputs.size(1), 1)
            batch_size, seq_len, dim = complex_wave.shape
            flat_wave = complex_wave.view(batch_size * seq_len, dim)
            swarm_wavefronts = flat_wave.unsqueeze(0).repeat(16, 1, 1)
            
            expert_waves = orchestrator.core(swarm_wavefronts)
            consensus_wave = triton_fused_superposition(expert_waves)
            
            logits = egress_head(consensus_wave)
            loss = criterion(logits, targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_parameters, 1.0)
            optimizer_all.step()
            
            orchestrator.apply_viscoelastic_gradient_updates(lr=1e-3)
            total_loss += loss.item()
            if batch_idx >= steps_per_epoch - 1: break

        orchestrator.core.bjorck_newton_orthonormalize(iterations=5)
        orchestrator.l3_router.project_to_unit_modulus()
        epoch_time = time.time() - start_time
        total_loss /= steps_per_epoch
        print(f" -> Epoch {epoch+1}/{epochs_phase3} | Avg Loss: {total_loss:.6f} | Latency: {epoch_time:.2f}s")

    print("\n[*] Serializing fresh core parameters...")
    torch.save({
        'core': orchestrator.core.state_dict(),
        'l3_router.weight': orchestrator.l3_router.state_dict(),
        'egress_head.weight': egress_head.state_dict(),
    }, output_path)
    print(f" -> Aligned swarm model written to: {output_path}")

if __name__ == "__main__":
    run_swarm_egress_compilation()