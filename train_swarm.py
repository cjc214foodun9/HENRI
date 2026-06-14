import os
import json
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Reconfigure console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# 1. Sealed Dataset Loader (Only returning numeric matrix space to avoid batching collation collapse)
class HenriSwarmDataset(Dataset):
    def __init__(self, dataset_dir="c:/Users/chan/Desktop/HENRI 7B SWARM/HENRI/archive/raw_sources"):
        # We target the compiled esc_compiled_dataset folder where JSON packets are staged
        self.dataset_dir = "c:/Users/chan/Desktop/HENRI 7B SWARM/HENRI/esc_compiled_dataset"
        if not os.path.exists(self.dataset_dir):
            self.dataset_dir = "./esc_compiled_dataset" # fallback
            
        if os.path.exists(self.dataset_dir):
            self.files = [os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir) if f.endswith(".json")]
        else:
            self.files = []
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        with open(self.files[idx], "r", encoding="utf-8") as f:
            data = json.load(f)
        # Return only the numerical matrix space for seamless tensor batching
        return torch.tensor(data["tensor_data"], dtype=torch.float32)

# Try importing the remote orchestrator layers (Vast.ai training context)
# Fallback to local mocks to allow syntax validation
try:
    from cloud_orchestrator import Henri7BSwarmCore, calculate_topological_loss, UnitaryLinearLayer
    HAS_REMOTE_ORCHESTRATOR = True
except ImportError:
    HAS_REMOTE_ORCHESTRATOR = False
    print("[WARN] Could not import cloud_orchestrator. Using mock placeholders for local validation.")
    
    class Henri7BSwarmCore(nn.Module):
        def __init__(self, dim=4096, depth=32):
            super().__init__()
            self.dim = dim
            self.depth = depth
            # Simple mock layer
            self.dummy_layer = nn.Linear(dim, dim)
            
        def forward(self, x, temperature=0.01):
            return self.dummy_layer(x)
            
    def calculate_topological_loss(output_wave, boundary_vectors):
        return torch.mean((output_wave - boundary_vectors) ** 2)
        
    class UnitaryLinearLayer(nn.Module):
        def force_unitary_manifold(self):
            pass

# 2. Complete Execution Framework
def execute_master_train_run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ACTIVE SUBSTRATE] Launching full-scale core on device: {device}")

    # Build the full 32-layer configuration footprint
    model = Henri7BSwarmCore(dim=4096, depth=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    dataset = HenriSwarmDataset()
    if len(dataset) == 0:
        print("[ERROR] Dataset is empty. Please run the Epistemic Data Foundry first to compile packets.")
        # If running mock verification, generate dummy data if needed
        return

    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
    print(f"[+] Loaded dataset of size: {len(dataset)} | Batches: {len(loader)}")

    model.train()
    for epoch in range(5):
        for batch_idx, boundary_vectors in enumerate(loader):
            optimizer.zero_grad()
            
            boundary_vectors = boundary_vectors.to(device)
            mock_initial_state = torch.randn_like(boundary_vectors).to(device)
            
            # Forward Pass: Running the wave mechanics through the 32 layers
            output_wave = model(mock_initial_state, temperature=0.01)
            
            # Compute physical stress and conformance loss
            free_energy = calculate_topological_loss(output_wave, boundary_vectors)
            free_energy.backward()
            
            optimizer.step()
            
            # Force compliance with strict orthogonal properties post-step
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, UnitaryLinearLayer):
                        module.force_unitary_manifold()
                        
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:03d} | Loss Free Energy: {free_energy.item():.6f}")

    # Save native PyTorch state dict for server deployment
    output_weights_path = "./henri_core_final.pt"
    torch.save(model.state_dict(), output_weights_path)
    print(f"[SUCCESS] Training cycle complete. Native weights anchored at: {output_weights_path}")

if __name__ == "__main__":
    execute_master_train_run()
