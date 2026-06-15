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
    def __init__(self, dataset_dir="./esc_compiled_dataset", dim=4096):
        self.dataset_dir = dataset_dir
        self.dim = dim
        
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
        return self.tensors[idx]

class InfiniteConformalWaveGenerator:
    """
    GPU-native Infinite Conformal Wave Generator and Data Augmenter.
    Generates high-density, multi-scale physical boundary waveforms directly in GPU VRAM.
    If seed_tensors are provided, it generates infinite batches by performing random
    conformal interpolations (superpositions), phase-shifting, and adding multi-scale
    harmonic waves and screened Poisson decays to the concepts, preventing overfitting
    on small datasets.
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
        
    def generate_batch(self, batch_size=8, dtype=torch.float32):
        # 1. Base spatial grid coordinate (t)
        t = torch.linspace(0.0, 2.0 * math.pi, self.dim, device=self.device, dtype=dtype)
        t_expanded = t.unsqueeze(0) # (1, dim)
        
        # 2. Randomize wave frequency components (MIT Fourier spatial frequencies)
        freq_alpha = torch.randint(1, 12, (batch_size, 1), device=self.device).to(dtype)
        freq_beta = torch.randint(15, 60, (batch_size, 1), device=self.device).to(dtype)
        phase_shift = torch.rand((batch_size, 1), device=self.device, dtype=dtype) * 2.0 * math.pi
        
        # Harmonic sum representing Dirichlet potential boundaries
        harmonics = torch.sin(freq_alpha * t_expanded + phase_shift) + 0.3 * torch.cos(freq_beta * t_expanded)
        
        # 3. Screened Poisson Equation boundary simulation (Bessel-like screening decay G(r) ~ e^{-lambda*r} / sqrt(r))
        r = torch.linspace(0.1, 5.0, self.dim, device=self.device, dtype=dtype)
        screened_damping = torch.exp(-0.5 * r) / torch.sqrt(r)
        screened_damping = screened_damping.unsqueeze(0) # (1, dim)
        
        # Combine the boundary harmonics with localized screened damping
        perturbation = harmonics * screened_damping
        
        if self.seed_tensors is not None:
            # 4. Generate random superposition weights (batch_size, num_seeds)
            num_seeds = self.seed_tensors.shape[0]
            weights = torch.zeros(batch_size, num_seeds, device=self.device, dtype=dtype)
            
            for i in range(batch_size):
                # Pick 2-5 random concepts to mix
                num_to_mix = torch.randint(2, 6, (1,)).item()
                indices = torch.randperm(num_seeds, device=self.device)[:num_to_mix]
                mix_weights = torch.rand(num_to_mix, device=self.device, dtype=dtype)
                weights[i, indices] = mix_weights / mix_weights.sum()
                
            # Perform batch matrix multiplication to get base waves
            base_waves = torch.matmul(weights, self.seed_tensors.to(dtype))
            
            # Mix base concept waves with perturbation (85% concept, 15% physical wave perturbation)
            # This maintains semantic concept structure while injecting geometric noise
            wave = 0.85 * base_waves + 0.15 * perturbation
        else:
            wave = perturbation
            
        # 5. Enforce strict VSA unit-hypersphere normalization (energy conservation)
        wave = F.normalize(wave, p=2, dim=-1)
        return wave

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
    return parser.parse_args()

# 2. Complete Execution Framework
def execute_master_train_run():
    args = parse_args()
    
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "\n[FATAL] CUDA (GPU acceleration) is requested but is not available in the current PyTorch environment.\n"
                "To resolve this, verify that:\n"
                "  1. A compatible NVIDIA GPU driver is installed.\n"
                "  2. PyTorch with CUDA support is installed. You can install it via:\n"
                "     pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall\n"
                "  3. Your CUDA_VISIBLE_DEVICES environment variable is configured correctly.\n"
                "Alternatively, run with '--device cpu' to explicitly allow CPU training."
            )
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"[ACTIVE SUBSTRATE] Launching full-scale core on device: {device}")
    if device.type == "cuda":
        print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"  Allocated VRAM: {torch.cuda.memory_allocated(0)/(1024**3):.2f} GiB / Reserved VRAM: {torch.cuda.memory_reserved(0)/(1024**3):.2f} GiB")

    # Build the target configuration footprint
    print(f"[*] Initializing ProprietaryHENRICore (dim={args.dim}, depth={args.depth}, experts={args.experts})...")
    model = ProprietaryHENRICore(dim=args.dim, depth=args.depth, num_fluid_states=args.experts)
    
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    use_bf16 = False
    
    # Cast weights on CPU first to save system RAM and VRAM before GPU migration
    if args.amp:
        if device_type == 'cuda' and torch.cuda.is_bf16_supported():
            model = model.bfloat16()
            use_bf16 = True
            print("[+] Cast model weights to bfloat16 for CPU/GPU memory efficiency.")
        elif device_type == 'cpu':
            model = model.bfloat16()
            use_bf16 = True
            print("[+] Cast model weights to bfloat16 (CPU fallback).")
        else:
            model = model.half()
            print("[+] Cast model weights to float16.")
            
    model = model.to(device)
    
    # Configure gradient checkpointing
    if args.checkpointing:
        model.gradient_checkpointing = True
        print("[+] Gradient Checkpointing is ENABLED.")
        
    # Configure optimizer
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(f"[+] Using {args.optimizer.upper()} Optimizer (lr={args.lr:.1e}, weight_decay={args.weight_decay:.1e})")

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
                dataset = HenriSwarmDataset(dataset_dir=dataset_dir, dim=args.dim)
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
        dataset = HenriSwarmDataset(dataset_dir=args.dataset_dir, dim=args.dim)
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
                boundary_vectors = generator.generate_batch(batch_size=args.batch_size, dtype=model_dtype)
            else:
                batch_idx, boundary_vectors = step
                boundary_vectors = F.normalize(boundary_vectors.to(device, dtype=model_dtype), p=2, dim=-1)
                
            mock_initial_state = F.normalize(torch.randn_like(boundary_vectors).to(device, dtype=model_dtype), p=2, dim=-1)
            
            # Fetch current thermostat temperature
            temperature = thermostat.get_temperature()
            
            # Forward pass under autocast (mixed precision)
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype, enabled=args.amp):
                # We manually reconstruct the layer trajectory to compute full NaturalInductionLoss
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
                    
                wave_trajectory = torch.cat(wave_trajectory, dim=1)
                
                # Compute thermodynamic Free Energy
                free_energy = loss_fn(wave_trajectory, boundary_vectors, temperature)

            # Backward pass and optimizer step
            if args.amp and not use_bf16:
                scaler.scale(free_energy).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                free_energy.backward()
                optimizer.step()
            
            # Force compliance with strict orthogonal properties post-step (SVD manifold projection)
            # [CRITICAL] Disable autocast to preserve orthogonal manifold purity
            with torch.amp.autocast(device_type=device_type, enabled=False):
                with torch.no_grad():
                    for module in model.modules():
                        if isinstance(module, UnitaryLinearLayer):
                            module.force_unitary_manifold()
            
            # Update the thermostat temperature
            new_temp = thermostat.step(free_energy.item())
            epoch_loss += free_energy.item()
            
            if batch_idx % 5 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:03d}/{steps_per_epoch:03d} | Loss Free Energy: {free_energy.item():.6f} | Temp: {new_temp:.4f}")

        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"[EPOCH COMPLETED] Epoch {epoch} | Avg Loss Free Energy: {avg_epoch_loss:.6f} | Final Temp: {thermostat.get_temperature():.4f}")
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
