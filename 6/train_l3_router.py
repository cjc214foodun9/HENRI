import torch
import torch.nn as nn
import os
import time

class PhaseResonanceInfoNCE(nn.Module):
    """
    Phase-Resonance InfoNCE (Supervised Contrastive) Loss in the complex domain.
    Sharpened by temperature tau = 0.05 to force massive high-dimensional distance
    and ensure mathematical orthogonality between distinct Swarm Masters.
    """
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, resonance_scores, target_master_ids):
        """
        resonance_scores: [Batch, Num_Masters] - Cosine similarity resonance metrics.
        target_master_ids: [Batch] - Integer IDs of correct Swarm Masters.
        """
        # Scale similarities by temperature to sharpen the distribution
        scaled_logits = resonance_scores / self.temperature
        
        # Cross-entropy inherently computes log-softmax over similarities
        loss = self.criterion(scaled_logits, target_master_ids)
        return loss


def train_swarm_router(router_model, train_dataloader, epochs=10, lr=3e-5):
    """
    V-Cache pinned training pipeline for the L3SwarmRouter.
    Binds the Python process to cores sharing the high-speed L3 3D V-Cache die.
    """
    print("--- Booting Swarm Router Training Pipeline ---")
    
    # 1. Lock execution thread to EPYC cores 0-15 (sharing primary V-Cache die)
    try:
        os.sched_setaffinity(0, set(range(min(16, os.cpu_count()))))
        print(f"Process successfully pinned to Cores 0-{min(15, os.cpu_count() - 1)}. L3 SRAM execution locked.")
    except AttributeError:
        print("[!] Core pinning failed: os.sched_setaffinity not supported on this platform/OS. Continuing on default cores.")
    except Exception as e:
        print(f"[!] Warning setting core affinity: {e}. Continuing on default cores.")

    # 2. Initialize Optimizer & Loss Function
    optimizer = torch.optim.AdamW(router_model.parameters(), lr=lr, weight_decay=0.01)
    contrastive_loss_fn = PhaseResonanceInfoNCE(temperature=0.05)
    
    router_model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct_routings = 0
        total_samples = 0
        start_time = time.perf_counter()
        
        for batch_idx, (inputs, target_master_ids) in enumerate(train_dataloader):
            # Move tensors to CPU (locked in L3 Cache)
            target_master_ids = target_master_ids.to('cpu')
            
            optimizer.zero_grad()
            
            # Forward Pass: determine if inputs are tokens (LongTensor) or activation hidden states (FloatTensor)
            if inputs.dtype in (torch.int32, torch.int64):
                hrr_wave, winning_master_id, resonance_scores = router_model(tokens=inputs.to('cpu'))
            else:
                hrr_wave, winning_master_id, resonance_scores = router_model(activations=inputs.to('cpu'))
                
            # Calculate Contrastive Loss
            loss = contrastive_loss_fn(resonance_scores, target_master_ids)
            
            # Backward Pass
            loss.backward()
            
            # Gradient clipping to protect continuous phase-amplitude geometry mapping
            torch.nn.utils.clip_grad_norm_(router_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # CRITICAL: Re-normalize the trainable Master Signatures back to the unit circle
            router_model.enforce_vsa_invariants()
            
            # Telemetry Metrics
            total_loss += loss.item()
            correct_routings += (winning_master_id == target_master_ids).sum().item()
            total_samples += target_master_ids.size(0)
            
        epoch_time = time.perf_counter() - start_time
        accuracy = (correct_routings / total_samples) * 100
        avg_loss = total_loss / len(train_dataloader)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | "
              f"Loss: {avg_loss:.4f} | "
              f"Routing Accuracy: {accuracy:.2f}% | "
              f"Time: {epoch_time:.2f}s")
              
    print("--- Swarm Router Training Complete ---")
    return router_model
