import os
import sys
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import argparse

# Reconfigure console encoding for Unicode compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure imports from henri_core can be resolved from local paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.dirname(script_dir))

from henri_core.core import ProprietaryHENRICore, UnitaryLinearLayer
from henri_core.thermodynamics import NaturalInductionLoss, DivergentMaster

# 1. Sealed Dataset Loader (Returning numeric matrix space matched to dim)
class HenriSwarmDataset(Dataset):
    def __init__(self, dataset_dir="./esc_compiled_dataset", dim=4096, seq_len=5):
        self.dataset_dir = dataset_dir
        self.dim = dim
        self.seq_len = seq_len
        
        # Absolute paths fallback check
        if not os.path.exists(self.dataset_dir):
            fallback_dir = os.path.join(script_dir, "esc_compiled_dataset")
            if os.path.exists(fallback_dir):
                self.dataset_dir = fallback_dir
            else:
                # Local workspace fallback path
                self.dataset_dir = "c:/Users/chan/Desktop/HENRI 7B SWARM/HENRI/esc_compiled_dataset"
            
        if os.path.exists(self.dataset_dir):
            self.files = sorted([os.path.join(self.dataset_dir, f) for f in os.listdir(self.dataset_dir) if f.endswith(".json")])
        else:
            self.files = []
            
        # Pre-load all tensors to avoid disk I/O and JSON decoding bottlenecks on the CPU
        print(f"[*] Pre-loading {len(self.files)} files into memory...")
        self.tensors = []
        for file_path in self.files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tensor = torch.tensor(data["tensor_data"], dtype=torch.float32)
                
                # Adjust dimensions to match training target dim dynamically
                if tensor.shape[0] != self.dim:
                    if tensor.shape[0] < self.dim:
                        padded = torch.zeros(self.dim, dtype=torch.float32)
                        padded[:tensor.shape[0]] = tensor
                        tensor = padded
                    else:
                        tensor = tensor[:self.dim]
                self.tensors.append(tensor)
            except Exception as e:
                print(f"[WARN] Failed to load {file_path}: {e}")
        print(f"[+] Loaded {len(self.tensors)} dataset vectors into RAM.")
        
    def __len__(self):
        return len(self.tensors)
        
    def __getitem__(self, idx):
        base_tensor = self.tensors[idx]
        seq_list = []
        for s in range(self.seq_len):
            # Apply slight phase shift / drift to generate sequential data
            drift = torch.sin(torch.linspace(0, 2*math.pi, self.dim) * (s + 1)) * 0.02
            perturbed = base_tensor + drift
            seq_list.append(F.normalize(perturbed, p=2, dim=-1))
        return torch.stack(seq_list)

class InfiniteConformalWaveGenerator:
    """
    GPU-native Infinite Conformal Wave Generator and Data Augmenter.
    Streams actual sequential chronological states through conformal interpolations
    and temporal phase-shifts.
    """
    def __init__(self, dim=4096, device="cuda", seed_tensors=None):
        self.dim = dim
        self.device = device
        if seed_tensors is not None and len(seed_tensors) > 0:
            self.seed_tensors = torch.stack(seed_tensors).to(device)
            print(f"[+] Infinite Conformal Wave Generator initialized with {len(seed_tensors)} seed concepts for GPU-native augmentation.")
        else:
            self.seed_tensors = None
            print("[+] Infinite Conformal Wave Generator initialized in pure synthetic wave mode.")
        
    def generate_batch(self, batch_size=8, seq_len=5, dtype=torch.float32):
        t = torch.linspace(0.0, 2.0 * math.pi, self.dim, device=self.device, dtype=dtype)
        t_expanded = t.unsqueeze(0).unsqueeze(1) # (1, 1, dim)
        
        # Chronological progression index
        step_indices = torch.arange(seq_len, device=self.device, dtype=dtype).reshape(1, seq_len, 1) # (1, seq_len, 1)
        
        freq_alpha = torch.randint(1, 12, (batch_size, 1, 1), device=self.device).to(dtype)
        freq_beta = torch.randint(15, 60, (batch_size, 1, 1), device=self.device).to(dtype)
        phase_shift = torch.rand((batch_size, 1, 1), device=self.device, dtype=dtype) * 2.0 * math.pi
        
        # Add temporal drift: phase shifts linearly with step index to simulate chronological progression
        temporal_shift = step_indices * 0.25 # drift rate
        
        harmonics = torch.sin(freq_alpha * t_expanded + phase_shift + temporal_shift) + 0.3 * torch.cos(freq_beta * t_expanded)
        
        r = torch.linspace(0.1, 5.0, self.dim, device=self.device, dtype=dtype)
        screened_damping = (torch.exp(-0.5 * r) / torch.sqrt(r)).unsqueeze(0).unsqueeze(1) # (1, 1, dim)
        
        perturbation = harmonics * screened_damping
        
        if self.seed_tensors is not None:
            num_seeds = self.seed_tensors.shape[0]
            weights = torch.zeros(batch_size, seq_len, num_seeds, device=self.device, dtype=dtype)
            for i in range(batch_size):
                # Interpolate between two distinct mix of seeds to simulate chronological progression
                num_to_mix_a = torch.randint(2, 6, (1,)).item()
                indices_a = torch.randperm(num_seeds, device=self.device)[:num_to_mix_a]
                mix_weights_a = torch.rand(num_to_mix_a, device=self.device, dtype=dtype)
                mix_weights_a = mix_weights_a / mix_weights_a.sum()
                
                num_to_mix_b = torch.randint(2, 6, (1,)).item()
                indices_b = torch.randperm(num_seeds, device=self.device)[:num_to_mix_b]
                mix_weights_b = torch.rand(num_to_mix_b, device=self.device, dtype=dtype)
                mix_weights_b = mix_weights_b / mix_weights_b.sum()
                
                for s in range(seq_len):
                    frac = s / max(1, seq_len - 1)
                    weights[i, s, indices_a] += (1.0 - frac) * mix_weights_a
                    weights[i, s, indices_b] += frac * mix_weights_b
            
            base_waves = torch.matmul(weights, self.seed_tensors.to(dtype))
            wave = 0.85 * base_waves + 0.15 * perturbation
        else:
            wave = perturbation
            
        wave = F.normalize(wave, p=2, dim=-1)
        return wave

# Transition Dynamics Network (F_theta) for Next-Latent prediction
class TransitionDynamicsNetwork(nn.Module):
    def __init__(self, dim=4096, hidden_dim=1024):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
        
    def forward(self, z_t, x_next):
        # z_t: [B, dim]
        # x_next: [B, dim]
        cat_input = torch.cat([z_t, x_next], dim=-1)
        return self.net(cat_input)

# Birkhoff Topological Loss Module
class BirkhoffTopologicalLoss(nn.Module):
    def __init__(self, translation_head, alpha=1.0, beta=0.05, eta=0.1):
        """
        Orchestrates non-autoregressive score matching with aesthetic constraints.
        """
        super().__init__()
        self.translation_head = translation_head
        self.alpha = alpha
        self.beta = beta
        self.eta = eta

    def forward(self, pred_score, target_score, canvas_state):
        # 1. Denoising Score Matching Loss (Base Alignment)
        loss_score = F.mse_loss(pred_score, target_score, reduction='mean')

        # Project the continuous latent canvas to the discrete token vocabulary space
        logits = self.translation_head(canvas_state)
        probs = F.softmax(logits, dim=-1)

        # 2. Entropy Minimization (Complexity Control 'C')
        epsilon = 1e-9
        entropy_per_token = -torch.sum(probs * torch.log(probs + epsilon), dim=-1)
        loss_entropy_C = torch.mean(entropy_per_token)

        # 3. Structural Geodesic Order (Order Control 'O')
        trajectory_delta = canvas_state[:, 1:, :] - canvas_state[:, :-1, :]
        loss_roughness_TV = torch.mean(torch.abs(trajectory_delta))

        # 4. Synthesizing the Birkhoff Objective Matrix
        total_loss = (self.alpha * loss_score) + (self.beta * loss_entropy_C) + (self.eta * loss_roughness_TV)

        metrics = {
            "loss_score_mse": loss_score.item(),
            "complexity_entropy_C": loss_entropy_C.item(),
            "roughness_TV_O": loss_roughness_TV.item(),
            "birkhoff_measure_estimate": (1.0 / (loss_entropy_C.item() + loss_roughness_TV.item() + epsilon))
        }

        return total_loss, metrics

def parse_args():
    parser = argparse.ArgumentParser(description="HENRI 7B Swarm Training Pipeline")
    parser.add_argument("--dim", type=int, default=4096, help="Embedding dimension (default: 4096)")
    parser.add_argument("--depth", type=int, default=32, help="Number of layers / block depth (default: 32)")
    parser.add_argument("--experts", type=int, default=16, help="Number of experts / fluid states (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd"], default="adamw", help="Optimizer choice (default: adamw)")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP) to save VRAM")
    parser.add_argument("--checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM")
    parser.add_argument("--dataset-dir", type=str, default="./esc_compiled_dataset", help="Path to compiled JSON dataset")
    parser.add_argument("--infinite", action="store_true", help="Use GPU-native Infinite Conformal Wave Generator to stream infinite training data")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Number of training steps per epoch in infinite mode (default: 1000)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Target execution device (default: cuda)")
    parser.add_argument("--output-weights", type=str, default="./henri_core_final.pt", help="Path to save trained weights")
    
    # Newly added NextLat & Birkhoff args
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length for chronological context (default: 5)")
    parser.add_argument("--vocab-size", type=int, default=262144, help="Target vocabulary size (default: 262144)")
    parser.add_argument("--alpha-free", type=float, default=1.0, help="Free energy coefficient (default: 1.0)")
    parser.add_argument("--beta-nextlat", type=float, default=1.0, help="NextLat coefficient (default: 1.0)")
    parser.add_argument("--gamma-birkhoff", type=float, default=1.0, help="Birkhoff loss coefficient (default: 1.0)")
    parser.add_argument("--birkhoff-beta", type=float, default=0.05, help="Birkhoff entropy scaling coefficient (default: 0.05)")
    parser.add_argument("--birkhoff-eta", type=float, default=0.1, help="Birkhoff TV roughness scaling coefficient (default: 0.1)")
    
    return parser.parse_args()

# 2. Complete Execution Framework
def execute_master_train_run():
    args = parse_args()
    
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "\n[FATAL] CUDA (GPU acceleration) is requested but is not available in the current PyTorch environment.\n"
                "To resolve this, run with '--device cpu' to explicitly allow CPU training."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"[ACTIVE SUBSTRATE] Launching full-scale core on device: {device}")
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  Allocated VRAM: {torch.cuda.memory_allocated(0)/(1024**3):.2f} GiB / Reserved VRAM: {torch.cuda.memory_reserved(0)/(1024**3):.2f} GiB")

    # Build the target configuration footprint
    print(f"[*] Initializing ProprietaryHENRICore (dim={args.dim}, depth={args.depth}, experts={args.experts}) on {device}...")
    with torch.device(device):
        model = ProprietaryHENRICore(dim=args.dim, depth=args.depth, num_fluid_states=args.experts)
        
        # Instantiate NextLat and Birkhoff heads
        print(f"[*] Initializing auxiliary heads on {device}: Transition MLP and Translation Head (vocab={args.vocab_size})...")
        transition_mlp = TransitionDynamicsNetwork(dim=args.dim)
        translation_head = nn.Linear(args.dim, args.vocab_size)
        birkhoff_loss_fn = BirkhoffTopologicalLoss(translation_head, alpha=1.0, beta=args.birkhoff_beta, eta=args.birkhoff_eta)


    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    use_bf16 = False
    
    # Cast weights on CPU first to save system RAM and VRAM before GPU migration
    if args.amp:
        if device_type == 'cuda' and torch.cuda.is_bf16_supported():
            model = model.bfloat16()
            transition_mlp = transition_mlp.bfloat16()
            translation_head = translation_head.bfloat16()
            use_bf16 = True
            print("[+] Cast weights to bfloat16 for CPU/GPU memory efficiency.")
        elif device_type == 'cpu':
            model = model.bfloat16()
            transition_mlp = transition_mlp.bfloat16()
            translation_head = translation_head.bfloat16()
            use_bf16 = True
            print("[+] Cast weights to bfloat16 (CPU fallback).")
        else:
            model = model.half()
            transition_mlp = transition_mlp.half()
            translation_head = translation_head.half()
            print("[+] Cast weights to float16.")
            
    model = model.to(device)
    
    # Configure gradient checkpointing
    if args.checkpointing:
        model.gradient_checkpointing = True
        print("[+] Gradient Checkpointing is ENABLED.")
        
    # Configure decoupled optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": transition_mlp.parameters(), "lr": 5e-4, "weight_decay": 1e-5},
            {"params": translation_head.parameters(), "lr": 5e-4, "weight_decay": 1e-5}
        ])
    else:
        # SGD fallback
        optimizer = torch.optim.SGD([
            {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": transition_mlp.parameters(), "lr": args.lr * 2, "weight_decay": 1e-5},
            {"params": translation_head.parameters(), "lr": args.lr * 2, "weight_decay": 1e-5}
        ])
    print(f"[+] Using {args.optimizer.upper()} Optimizer with Decoupled Learning Rates")

    # Initialize the thermodynamic loss and DivergentMaster thermostat
    loss_fn = NaturalInductionLoss(lambda_boundary=10.0, reg_coefficient=1.0, dim=args.dim).to(device)
    thermostat = DivergentMaster(t_min=0.0, t_max=4.0, cooling_rate=0.05, heat_sensitivity=0.2, lock_threshold=1e-4)
    print("[+] Initialized DivergentMaster Thermostat and NaturalInductionLoss.")

    # Configure Data Ingestion (Infinite Generator vs Static Dataset)
    if args.infinite:
        seed_tensors = None
        # Try to load seed tensors for hybrid data augmentation if the dataset exists
        try:
            dataset_dir = args.dataset_dir
            if not os.path.exists(dataset_dir):
                fallback_dir = os.path.join(script_dir, "esc_compiled_dataset")
                if os.path.exists(fallback_dir):
                    dataset_dir = fallback_dir
                else:
                    dataset_dir = "c:/Users/chan/Desktop/HENRI 7B SWARM/HENRI/esc_compiled_dataset"
            
            if os.path.exists(dataset_dir) and any(f.endswith(".json") for f in os.listdir(dataset_dir)):
                print(f"[*] Found concept files in '{dataset_dir}'. Loading seeds for hybrid augmentation...")
                dataset = HenriSwarmDataset(dataset_dir=dataset_dir, dim=args.dim, seq_len=args.seq_len)
                if len(dataset) > 0:
                    seed_tensors = dataset.tensors
            else:
                print(f"[!] No seed concept files found in '{dataset_dir}'. Falling back to pure synthetic waves.")
        except Exception as e:
            print(f"[WARN] Error loading seed dataset for augmentation: {e}. Falling back to pure synthetic waves.")
            
        generator = InfiniteConformalWaveGenerator(dim=args.dim, device=device, seed_tensors=seed_tensors)
        steps_per_epoch = args.steps_per_epoch
        print(f"[+] Infinite Conformal Wave Generator initialized. Steps per Epoch: {steps_per_epoch}")
    else:
        dataset = HenriSwarmDataset(dataset_dir=args.dataset_dir, dim=args.dim, seq_len=args.seq_len)
        if len(dataset) == 0:
            print(f"[ERROR] Dataset folder '{args.dataset_dir}' contains no valid packets. Please run data_foundry.py first or enable --infinite.")
            sys.exit(1)
            
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        steps_per_epoch = len(loader)
        print(f"[+] Loaded static dataset of size: {len(dataset)} | Batches per Epoch: {steps_per_epoch}")

    # AMP scaler initialization (only needed for float16 scaling)
    scaler = torch.amp.GradScaler(device_type, enabled=(args.amp and not use_bf16))
    if args.amp:
        print(f"[+] Automatic Mixed Precision (AMP) is ENABLED (Dtype: {'bfloat16' if use_bf16 else 'float16'}).")

    model.train()
    transition_mlp.train()
    translation_head.train()
    
    # Enforce initial orthogonality on expert layers
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, UnitaryLinearLayer):
                module.force_unitary_manifold()

    print("\n[*] Starting pre-training loop...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        # Determine training data source loop
        batch_source = range(steps_per_epoch) if args.infinite else enumerate(loader)
        
        for step in batch_source:
            optimizer.zero_grad()
            
            # Match input tensor dtype and device to the model weights
            model_dtype = next(model.parameters()).dtype
            
            if args.infinite:
                batch_idx = step
                boundary_vectors = generator.generate_batch(batch_size=args.batch_size, seq_len=args.seq_len, dtype=model_dtype)
            else:
                batch_idx, boundary_vectors = step
                boundary_vectors = F.normalize(boundary_vectors.to(device, dtype=model_dtype), p=2, dim=-1)
                
            # Boundary vectors shape: [B, S, Dim]
            # Mock initial state shape: [B, S, Dim]
            mock_initial_state = F.normalize(torch.randn_like(boundary_vectors).to(device, dtype=model_dtype), p=2, dim=-1)
            
            # Fetch current thermostat temperature
            temperature = thermostat.get_temperature()
            
            # Forward pass under autocast (mixed precision)
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype, enabled=args.amp):
                # We manually reconstruct the layer trajectory to compute full NaturalInductionLoss
                # Reconstructed trajectory over Depth steps
                wave_trajectory = [mock_initial_state.unsqueeze(1)]
                curr = mock_initial_state
                for layer in model.layers:
                    prev = curr
                    
                    if args.checkpointing and model.training:
                        # Wrap layer execution in checkpoint to save activation memory
                        def create_checkpoint_fn(block):
                            def checkpoint_fn(c_wave, p_wave, z_attractor, temp_val):
                                return block(c_wave, p_wave, z_attractor, temp_val.item())
                            return checkpoint_fn
                        
                        temp_tensor = torch.tensor(temperature, device=device, dtype=model_dtype, requires_grad=False)
                        curr, _ = torch.utils.checkpoint.checkpoint(
                            create_checkpoint_fn(layer),
                            curr,
                            prev,
                            boundary_vectors,
                            temp_tensor,
                            use_reentrant=False
                        )
                    else:
                        curr, _ = layer(curr, prev, boundary_vectors, temperature)
                        
                    wave_trajectory.append(curr.unsqueeze(1))
                    
                # wave_trajectory shape: [B, Depth+1, S, Dim]
                wave_trajectory = torch.cat(wave_trajectory, dim=1)
                
                # Permute trajectory and reshape for NaturalInductionLoss
                # Expects [B * S, Depth+1, Dim]
                B, Depth_plus_1, S, Dim = wave_trajectory.shape
                wave_trajectory_permuted = wave_trajectory.permute(0, 2, 1, 3).reshape(B * S, Depth_plus_1, Dim)
                boundary_vectors_reshaped = boundary_vectors.reshape(B * S, Dim)
                
                # 1. Compute thermodynamic Free Energy
                free_energy = loss_fn(wave_trajectory_permuted, boundary_vectors_reshaped, temperature)
                
                # Final wave output for each sequence step
                # Shape: [B, S, Dim]
                final_output = model.final_layer_norm(curr)
                
                # 2. Next-Latent Prediction Loss
                z_t = final_output[:, :-1, :] # [B, S-1, Dim]
                x_next = boundary_vectors[:, 1:, :] # [B, S-1, Dim]
                z_next = final_output[:, 1:, :] # [B, S-1, Dim]
                
                z_t_flat = z_t.reshape(-1, Dim)
                x_next_flat = x_next.reshape(-1, Dim)
                z_next_flat = z_next.reshape(-1, Dim)
                
                pred_z_next = transition_mlp(z_t_flat, x_next_flat)
                loss_nextlat = F.mse_loss(pred_z_next, z_next_flat)
                
                # 3. Birkhoff Topological Loss (Complexity C + Order O + MSE Alignment)
                loss_birkhoff, birkhoff_metrics = birkhoff_loss_fn(final_output, boundary_vectors, final_output)
                
                # 4. Total Composite Loss
                total_loss = (args.alpha_free * free_energy) + (args.beta_nextlat * loss_nextlat) + (args.gamma_birkhoff * loss_birkhoff)

            # Backward pass and optimizer step
            if args.amp and not use_bf16:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            # Force compliance with strict orthogonal properties post-step (SVD manifold projection)
            # [CRITICAL] Disable autocast to preserve orthogonal manifold purity
            with torch.amp.autocast(device_type=device_type, enabled=False):
                with torch.no_grad():
                    for module in model.modules():
                        if isinstance(module, UnitaryLinearLayer):
                            module.force_unitary_manifold()
            
            # Update the thermostat temperature based on free energy
            new_temp = thermostat.step(free_energy.item())
            epoch_loss += total_loss.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:03d}/{steps_per_epoch:03d} | Loss: {total_loss.item():.4f} | FreeEnergy: {free_energy.item():.4f} | NextLat: {loss_nextlat.item():.4f} | Birkhoff: {loss_birkhoff.item():.4f} | Temp: {new_temp:.4f}")

        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"[EPOCH COMPLETED] Epoch {epoch} | Avg Loss: {avg_epoch_loss:.6f} | Final Temp: {thermostat.get_temperature():.4f}")
        if torch.cuda.is_available():
            print(f"  VRAM Memory: {torch.cuda.memory_allocated(0)/(1024**3):.2f} GiB")

    # 3. Save native PyTorch state dict for server deployment
    print(f"\n[*] Saving trained weights model footprint to: {args.output_weights}...")
    torch.save(model.state_dict(), args.output_weights)
    print(f"[SUCCESS] Native weights anchored at: {args.output_weights}")
    
    # 4. Verify post-training weight orthorectification
    print("[*] Performing post-training orthorectification verification...")
    with torch.no_grad():
        test_orthogonal = True
        unitary_layers_count = 0
        max_deviation = 0.0
        
        for name, module in model.named_modules():
            if isinstance(module, UnitaryLinearLayer):
                unitary_layers_count += 1
                W = module.weight
                I_approx = torch.matmul(W, W.t())
                I = torch.eye(W.shape[0], device=W.device)
                deviation = torch.norm(I_approx - I).item()
                max_deviation = max(max_deviation, deviation)
                if deviation > 1e-2:
                    test_orthogonal = False
                    print(f"  [WARN] Module '{name}' deviation from orthogonality: {deviation:.6f}")
                    
        if test_orthogonal and unitary_layers_count > 0:
            print(f"  [SUCCESS] All {unitary_layers_count} expert phase-shift layers are strictly orthogonal (Max deviation: {max_deviation:.6e}).")
        elif unitary_layers_count == 0:
            print("  [WARN] No UnitaryLinearLayer modules were found to verify.")
        else:
            print("  [WARNING] Orthorectification verification completed with minor warnings.")

if __name__ == "__main__":
    execute_master_train_run()
