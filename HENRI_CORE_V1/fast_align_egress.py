import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multi_expert_swarm_pre_training_engine import FreshHENRIOrchestrator, SwarmProgramDataset
from holographic_egress_high_stress_logit_sieve import HolographicPhaseTransducer

def fast_align():
    print("[*] Initializing Fast Phase Alignment...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    orchestrator = FreshHENRIOrchestrator(vocab_size=32000, dim=4096, num_experts=16)
    try:
        orchestrator.core.load_state_dict(torch.load("henri_fresh_core.pt", map_location=device, weights_only=True))
        print(" -> Successfully loaded pristine core weights.")
    except Exception as e:
        print(f" -> Could not load weights: {e}")
        return
        
    orchestrator.to(device)
    for param in orchestrator.parameters():
        param.requires_grad = False
        
    new_transducer = HolographicPhaseTransducer(d_wave=4096, vocab_size=32000).to(device)
    dataset = SwarmProgramDataset(num_samples=200, seq_len=64, vocab_size=32000)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    optimizer = torch.optim.AdamW(new_transducer.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    print("[*] Freezing core physics. Training ONLY the Phase Transducer projection...")
    for epoch in range(3):
        total_loss = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.no_grad():
                wave = orchestrator.l3_router(inputs, None)
                wave = wave + 1j * torch.zeros_like(wave)
                swarm_waves = wave.unsqueeze(0).repeat(16, 1, 1).view(16, -1, 4096)
                expert_waves = orchestrator.core(swarm_waves)
                consensus = expert_waves.sum(dim=0)
                norm = torch.linalg.vector_norm(consensus, dim=-1, keepdim=True)
                consensus = consensus / norm.clamp(min=1e-8)
                
            logits = new_transducer(consensus)
            loss = criterion(logits, targets.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f" -> Epoch {epoch+1}/3 | Transducer Alignment Loss: {total_loss/len(loader):.4f}")
        
    torch.save(new_transducer.state_dict(), "aligned_phase_transducer.pt")
    print("[*] Alignment complete! Saved to aligned_phase_transducer.pt")

if __name__ == "__main__":
    fast_align()
