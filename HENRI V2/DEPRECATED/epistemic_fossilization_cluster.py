import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
import asyncio
import os
from morphogenetic_syncytium import SyncytiumCore
from phylogenetic_memory import EngramStore
from darwinian_selection_loop import SagnacEvolutionaryFilter

torch.set_default_dtype(torch.float32)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

async def fetch_axioms(store: EngramStore):
    axioms = {}
    hashes = ["arc_spatial_translation", "arc_color_invariance", "arc_symmetry", "arc_flood_fill"]
    await store._connect_with_backoff()
    async with store.pool.acquire() as conn:
        for h in hashes:
            row = await conn.fetchrow(
                "SELECT engram_wave::text FROM phylogenetic_engrams WHERE environmental_context_hash = $1", h
            )
            if row:
                engram = store._deserialize_wave(row["engram_wave"])
                axioms[h] = engram
    return axioms

async def fossilize():
    print("=" * 80)
    print("   PROJECT HENRI: EPISTEMIC FOSSILIZATION CLUSTER")
    print("=" * 80)
    
    DIMENSION = 4096
    DEPTH = 8
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Fossilizing on device: {device}")
    
    # 1. Initialize the Syncytium (The Bone)
    syncytium = SyncytiumCore(dimension=DIMENSION, depth=DEPTH).to(device)
    
    # 2. Connect to Zone C and fetch axioms
    store = EngramStore("postgresql://user:pass@localhost:5432/henri")
    await store.initialize_schema()
    axioms = await fetch_axioms(store)
    await store.close()
    
    if not axioms:
        logging.warning("No axioms found in Zone C! Ensure ARC-AGIzonecseed.py ran successfully.")
        return
        
    logging.info(f"Loaded {len(axioms)} axioms from Zone C.")
    
    # 3. Fossilization (Pre-Training the Bone)
    filter_probe = SagnacEvolutionaryFilter(t_max=3.0)
    optimizer = torch.optim.Adam(syncytium.parameters(), lr=0.001)
    
    target_coherence = 0.995
    epochs = 1000
    
    # Convert axioms to tensors
    input_noise = torch.randn(1, DIMENSION, dtype=torch.float32, device=device)
    psi_in = torch.complex(torch.cos(input_noise), torch.sin(input_noise))
    psi_in = F.normalize(psi_in, p=2, dim=-1)
    
    # Combine axioms into a target state or train sequentially
    # We will train on the sum of the axioms to fossilize a universal prior
    combined_target = sum(axioms.values())
    combined_target = combined_target.to(device)
    
    psi_target = F.normalize(combined_target, p=2, dim=-1)
    
    optimizer = torch.optim.Adam(syncytium.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        psi_out = syncytium(psi_in)
        
        # Sagnac Coherence for stopping condition
        coherence_val = torch.abs(torch.mean(psi_out * torch.conj(psi_target), dim=-1)).mean()
        
        # MSE Loss on real/imag components for strong, non-vanishing gradients
        loss = F.mse_loss(torch.view_as_real(psi_out), torch.view_as_real(psi_target))
        
        loss.backward()
        optimizer.step()
        
        # Keep the Bone on the Stiefel Manifold after Adam step
        with torch.no_grad():
            for layer in syncytium.layers:
                layer.weight.copy_(StiefelManifoldProjector.retract(layer.weight))
                
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch:04d} | Sagnac Coherence: {coherence_val.item():.4f} | Loss: {loss.item():.4f}")
            
        if coherence_val.item() >= target_coherence:
            logging.info(f"Sagnac Coherence reached {coherence_val.item():.4f} at epoch {epoch}. Bone is now locked.")
            break
            
    # 4. Save the fossilized core
    torch.save(syncytium.state_dict(), "henri_fossilized_core.pt")
    logging.info("[SUCCESS] Epistemic Fossilization Complete. Artifact saved to henri_fossilized_core.pt")

if __name__ == "__main__":
    asyncio.run(fossilize())
