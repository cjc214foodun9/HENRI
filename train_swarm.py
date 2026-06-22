import os
import sys
import json
import math
import h5py
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

# Ensure imports can be resolved
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, "6"))

from henri_core.core import ProprietaryHENRICore, UnitaryLinearLayer
from henri_core.thermodynamics import NaturalInductionLoss, DivergentMaster

# 1. HDF5 Wave Dataset Loader
class HenriWaveDataset(Dataset):
    def __init__(self, hdf5_path: str, seq_len: int = 5, dim: int = 4096, domain_key: str = None):
        self.hdf5_path = hdf5_path
        self.seq_len = seq_len
        self.dim = dim
        self.keys = []
        
        # Resolve path fallback check
        if not os.path.exists(self.hdf5_path):
            fallback_dir = os.path.join(script_dir, "henri_corpus_4096.h5")
            if os.path.exists(fallback_dir):
                self.hdf5_path = fallback_dir
            else:
                self.hdf5_path = "c:/Users/chan/Desktop/HENRI 7B SWARM/HENRI/henri_corpus_4096.h5"
            
        if os.path.exists(self.hdf5_path):
            with h5py.File(self.hdf5_path, 'r') as hf:
                if domain_key:
                    domain_keys = [domain_key]
                else:
                    domain_keys = list(hf.keys())
                
                for dk in domain_keys:
                    for k in hf[dk].keys():
                        self.keys.append((dk, k))
            print(f"[+] HenriWaveDataset loaded {len(self.keys)} wave signatures from HDF5 groups: {domain_keys}")
        else:
            print(f"[WARN] HDF5 file not found at: {self.hdf5_path}. Dataset is empty.")
        
    def __len__(self):
        return len(self.keys)
        
    def __getitem__(self, idx):
        domain_key, key = self.keys[idx]
        with h5py.File(self.hdf5_path, 'r') as hf:
            raw_data = hf[domain_key][key][:]
            
        real_part = torch.from_numpy(raw_data[..., 0])
        imag_part = torch.from_numpy(raw_data[..., 1])
        complex_wave = torch.complex(real_part, imag_part)
        
        # Project complex representation to real coordinates
        real_wave = torch.real(complex_wave)
        
        # Adjust dimensions to match training target dim
        if real_wave.shape[0] != self.dim:
            if real_wave.shape[0] < self.dim:
                padded = torch.zeros(self.dim, dtype=torch.float32)
                padded[:real_wave.shape[0]] = real_wave
                real_wave = padded
            else:
                real_wave = real_wave[:self.dim]
                
        # Expand into sequence using phase shifts
        seq_list = []
        for s in range(self.seq_len):
            drift = torch.sin(torch.linspace(0, 2*math.pi, self.dim) * (s + 1)) * 0.02
            perturbed = real_wave + drift
            seq_list.append(F.normalize(perturbed, p=2, dim=-1))
            
        return torch.stack(seq_list)

class InfiniteConformalWaveGenerator:
    """
    GPU-native Infinite Conformal Wave Generator and Data Augmenter.
    """
    def __init__(self, dim=4096, device="cuda", seed_tensors=None):
        self.dim = dim
        self.device = device
        if seed_tensors is not None and len(seed_tensors) > 0:
            self.seed_tensors = torch.stack(seed_tensors).to(device)
            print(f"[+] Infinite Conformal Wave Generator initialized with {len(seed_tensors)} seed concepts.")
        else:
            self.seed_tensors = None
            print("[+] Infinite Conformal Wave Generator initialized in pure synthetic wave mode.")
        
    def generate_batch(self, batch_size=8, seq_len=5, dtype=torch.float32):
        t = torch.linspace(0.0, 2.0 * math.pi, self.dim, device=self.device, dtype=dtype)
        t_expanded = t.unsqueeze(0).unsqueeze(1)
        
        step_indices = torch.arange(seq_len, device=self.device, dtype=dtype).reshape(1, seq_len, 1)
        temporal_shift = step_indices * 0.25
        
        freq_alpha = torch.randint(1, 12, (batch_size, 1, 1), device=self.device).to(dtype)
        freq_beta = torch.randint(15, 60, (batch_size, 1, 1), device=self.device).to(dtype)
        phase_shift = torch.rand((batch_size, 1, 1), device=self.device, dtype=dtype) * 2.0 * math.pi
        
        harmonics = torch.sin(freq_alpha * t_expanded + phase_shift + temporal_shift) + 0.3 * torch.cos(freq_beta * t_expanded)
        
        r = torch.linspace(0.1, 5.0, self.dim, device=self.device, dtype=dtype)
        screened_damping = (torch.exp(-0.5 * r) / torch.sqrt(r)).unsqueeze(0).unsqueeze(1)
        
        perturbation = harmonics * screened_damping
        
        if self.seed_tensors is not None:
            num_seeds = self.seed_tensors.shape[0]
            weights = torch.zeros(batch_size, seq_len, num_seeds, device=self.device, dtype=dtype)
            for i in range(batch_size):
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
        cat_input = torch.cat([z_t, x_next], dim=-1)
        return self.net(cat_input)

# Sketch Isotropic Gaussian Regularizer (SIGReg)
class SIGRegRegularizer(nn.Module):
    def __init__(self, knots: int = 17, num_proj: int = 256, dim: int = 4096):
        super().__init__()
        self.num_proj = num_proj
        self.dim = dim
        
        t_vals = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi_window = torch.exp(-t_vals.square() / 2.0)
        
        self.register_buffer("t", t_vals)
        self.register_buffer("phi", phi_window)
        self.register_buffer("weights", weights * phi_window)

    def forward(self, latent_trajectory: torch.Tensor) -> torch.Tensor:
        B, H, D = latent_trajectory.shape
        flat_latent = latent_trajectory.view(B * H, D).float()
        
        A = torch.randn(D, self.num_proj, device=latent_trajectory.device)
        A = A / (A.norm(p=2, dim=0, keepdim=True) + 1e-8)
        
        sliced_projections = torch.matmul(flat_latent, A).unsqueeze(-1)  # [B*H, N_proj, 1]
        x_t = sliced_projections * self.t                                # [B*H, N_proj, Knots]
        
        err = (x_t.cos().mean(dim=0) - self.phi).square() + x_t.sin().mean(dim=0).square()
        statistic = torch.matmul(err, self.weights) * float(flat_latent.size(0))
        
        return statistic.mean()

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

    def forward(self, pred_score, target_score, canvas_state, gumbel_temp=1.0):
        # 1. Denoising Score Matching Loss (Base Alignment)
        loss_score = F.mse_loss(pred_score, target_score, reduction='mean')

        # Project continuous latent canvas using dynamic Gumbel-Softmax annealing
        logits = self.translation_head(canvas_state)
        probs = F.gumbel_softmax(logits, tau=gumbel_temp, hard=False, dim=-1)

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
    parser = argparse.ArgumentParser(description="HENRI 8.59B Swarm Training Pipeline")
    parser.add_argument("--dim", type=int, default=4096, help="Embedding dimension (default: 4096)")
    parser.add_argument("--depth", type=int, default=32, help="Number of layers / block depth (default: 32)")
    parser.add_argument("--experts", type=int, default=16, help="Number of experts / fluid states (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--optimizer", type=str, choices=["adamw", "sgd", "adafactor"], default="adafactor", help="Optimizer choice (default: adafactor)")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (AMP) to save VRAM")
    parser.add_argument("--checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM")
    parser.add_argument("--dataset-path", type=str, default="./henri_corpus_4096.h5", help="Path to compiled HDF5 dataset")
    parser.add_argument("--infinite", action="store_true", help="Use GPU-native Infinite Conformal Wave Generator to stream infinite training data")
    parser.add_argument("--steps-per-epoch", type=int, default=1000, help="Number of training steps per epoch in infinite mode (default: 1000)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Target execution device (default: cuda)")
    parser.add_argument("--output-weights", type=str, default="./henri_core_final.pt", help="Path to save trained weights")
    
    # NextLat & Birkhoff args
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length for chronological context (default: 5)")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Target vocabulary footprint (default: 32000)")
    parser.add_argument("--alpha-free", type=float, default=1.0, help="Free energy coefficient (default: 1.0)")
    parser.add_argument("--beta-nextlat", type=float, default=1.0, help="NextLat coefficient (default: 1.0)")
    parser.add_argument("--gamma-birkhoff", type=float, default=1.0, help="Birkhoff loss coefficient (default: 1.0)")
    parser.add_argument("--beta-sigreg", type=float, default=1.0, help="SIGReg loss coefficient (default: 1.0)")
    parser.add_argument("--birkhoff-beta", type=float, default=0.05, help="Birkhoff entropy scaling coefficient (default: 0.05)")
    parser.add_argument("--birkhoff-eta", type=float, default=0.1, help="Birkhoff TV roughness scaling coefficient (default: 0.1)")
    
    return parser.parse_args()

def execute_master_train_run():
    args = parse_args()
    
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("[FATAL] CUDA is requested but is not available. Run with '--device cpu' for fallback.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"[ACTIVE SUBSTRATE] Launching full-scale core on device: {device}")
    
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    use_bf16 = False
    init_dtype = torch.float32
    
    if args.amp:
        if device_type == 'cuda' and torch.cuda.is_bf16_supported():
            init_dtype = torch.bfloat16
            use_bf16 = True
            print("[+] Will initialize and train in bfloat16 for memory efficiency.")
        else:
            init_dtype = torch.float16
            print("[+] Will initialize and train in float16 for memory efficiency.")
            
    print(f"[*] Initializing ProprietaryHENRICore (dim={args.dim}, depth={args.depth}, experts={args.experts}) on CPU in {init_dtype}...")
    
    # Temporarily set default dtype for memory-efficient CPU initialization
    torch.set_default_dtype(init_dtype)
    
    model = ProprietaryHENRICore(dim=args.dim, depth=args.depth, num_fluid_states=args.experts)
    transition_mlp = TransitionDynamicsNetwork(dim=args.dim)
    translation_head = nn.Linear(args.dim, args.vocab_size)
    birkhoff_loss_fn = BirkhoffTopologicalLoss(translation_head, alpha=1.0, beta=args.birkhoff_beta, eta=args.birkhoff_eta)
    sigreg_reg = SIGRegRegularizer(dim=args.dim)
    
    # Restore default dtype to float32
    torch.set_default_dtype(torch.float32)

    # Cast to final low-precision types if they aren't already
    if args.amp:
        if use_bf16:
            model = model.bfloat16()
            transition_mlp = transition_mlp.bfloat16()
            translation_head = translation_head.bfloat16()
            sigreg_reg = sigreg_reg.bfloat16()
        else:
            model = model.half()
            transition_mlp = transition_mlp.half()
            translation_head = translation_head.half()
            sigreg_reg = sigreg_reg.half()
            
    print(f"[*] Moving model weights to target device: {device}...")
    model = model.to(device)
    transition_mlp = transition_mlp.to(device)
    translation_head = translation_head.to(device)
    sigreg_reg = sigreg_reg.to(device)
    
    if args.checkpointing:
        model.gradient_checkpointing = True
        print("[+] Gradient Checkpointing is ENABLED.")
        
    print(f"[*] Instantiating optimizer: {args.optimizer}...")
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": transition_mlp.parameters(), "lr": 5e-4, "weight_decay": 1e-5},
            {"params": translation_head.parameters(), "lr": 5e-4, "weight_decay": 1e-5}
        ])
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD([
            {"params": model.parameters(), "lr": args.lr},
            {"params": transition_mlp.parameters(), "lr": 5e-4},
            {"params": translation_head.parameters(), "lr": 5e-4}
        ], momentum=0.9)
    elif args.optimizer == "adafactor":
        from transformers.optimization import Adafactor
        optimizer = Adafactor([
            {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": transition_mlp.parameters(), "lr": 5e-4, "weight_decay": 1e-5},
            {"params": translation_head.parameters(), "lr": 5e-4, "weight_decay": 1e-5}
        ], scale_parameter=False, relative_step=False, warmup_init=False, lr=args.lr)

    loss_fn = NaturalInductionLoss(lambda_boundary=10.0, reg_coefficient=1.0, dim=args.dim).to(device)
    thermostat = DivergentMaster(t_min=0.0, t_max=4.0, cooling_rate=0.05, heat_sensitivity=0.2, lock_threshold=1e-4)

    # Configure Data Ingestion from compiled HDF5 dataset
    if args.infinite:
        generator = InfiniteConformalWaveGenerator(dim=args.dim, device=device)
        steps_per_epoch = args.steps_per_epoch
        print(f"[+] Infinite Conformal Wave Generator initialized. Steps per Epoch: {steps_per_epoch}")
    else:
        dataset = HenriWaveDataset(hdf5_path=args.dataset_path, seq_len=args.seq_len, dim=args.dim)
        if len(dataset) == 0:
            print(f"[ERROR] Dataset path '{args.dataset_path}' has no valid signatures. Run data_foundry_compiler.py first or set --infinite.")
            sys.exit(1)
            
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        steps_per_epoch = len(loader)
        print(f"[+] Loaded HDF5 dataset size: {len(dataset)} | Batches per Epoch: {steps_per_epoch}")

    scaler = torch.amp.GradScaler(device_type, enabled=(args.amp and not use_bf16))

    model.train()
    transition_mlp.train()
    translation_head.train()
    sigreg_reg.train()
    
    # Enforce initial orthogonality
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, UnitaryLinearLayer):
                module.force_unitary_manifold()

    print("\n[*] Starting pre-training loop...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        batch_source = range(steps_per_epoch) if args.infinite else enumerate(loader)
        
        for step in batch_source:
            optimizer.zero_grad()
            model_dtype = next(model.parameters()).dtype
            
            if args.infinite:
                batch_idx = step
                boundary_vectors = generator.generate_batch(batch_size=args.batch_size, seq_len=args.seq_len, dtype=model_dtype)
            else:
                batch_idx, boundary_vectors = step
                boundary_vectors = F.normalize(boundary_vectors.to(device, dtype=model_dtype), p=2, dim=-1)
                
            # Dynamic Gumbel-Softmax temperature exponential annealing from 1.0 down to 0.01
            fraction = batch_idx / max(1, steps_per_epoch - 1)
            gumbel_temp = 1.0 * ((0.01 / 1.0) ** fraction)
                
            mock_initial_state = F.normalize(torch.randn_like(boundary_vectors).to(device, dtype=model_dtype), p=2, dim=-1)
            temperature = thermostat.get_temperature()
            
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.amp.autocast(device_type=device_type, dtype=autocast_dtype, enabled=args.amp):
                wave_trajectory = [mock_initial_state.unsqueeze(1)]
                curr = mock_initial_state
                for layer in model.layers:
                    prev = curr
                    if args.checkpointing and model.training:
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
                B, Depth_plus_1, S, Dim = wave_trajectory.shape
                wave_trajectory_permuted = wave_trajectory.permute(0, 2, 1, 3).reshape(B * S, Depth_plus_1, Dim)
                boundary_vectors_reshaped = boundary_vectors.reshape(B * S, Dim)
                
                # 1. Compute thermodynamic Free Energy
                free_energy = loss_fn(wave_trajectory_permuted, boundary_vectors_reshaped, temperature)
                
                # Final wave output
                final_output = model.final_layer_norm(curr)
                
                # 2. Next-Latent Prediction Loss
                z_t = final_output[:, :-1, :]
                x_next = boundary_vectors[:, 1:, :]
                z_next = final_output[:, 1:, :]
                
                z_t_flat = z_t.reshape(-1, Dim)
                x_next_flat = x_next.reshape(-1, Dim)
                z_next_flat = z_next.reshape(-1, Dim)
                
                pred_z_next = transition_mlp(z_t_flat, x_next_flat)
                loss_nextlat = F.mse_loss(pred_z_next, z_next_flat)
                
                # 3. Birkhoff Topological Loss (Complexity C + Order O + MSE Alignment) using dynamic Gumbel temp
                loss_birkhoff, birkhoff_metrics = birkhoff_loss_fn(final_output, boundary_vectors, final_output, gumbel_temp=gumbel_temp)
                
                # 4. Yann LeCun's SIGReg entropy loss to prevent collapse
                # Expects shape [B * S, Depth+1, Dim]
                loss_sigreg = sigreg_reg(wave_trajectory_permuted)
                
                # 5. Total Composite Loss
                total_loss = (args.alpha_free * free_energy) + (args.beta_nextlat * loss_nextlat) + (args.gamma_birkhoff * loss_birkhoff) + (args.beta_sigreg * loss_sigreg)

            if args.amp and not use_bf16:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()
            
            # Post-step Stiefel manifold projection in FP32
            with torch.amp.autocast(device_type=device_type, enabled=False):
                with torch.no_grad():
                    for module in model.modules():
                        if isinstance(module, UnitaryLinearLayer):
                            module.force_unitary_manifold()
            
            new_temp = thermostat.step(free_energy.item())
            epoch_loss += total_loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:03d}/{steps_per_epoch:03d} | Loss: {total_loss.item():.4f} | FreeEnergy: {free_energy.item():.4f} | NextLat: {loss_nextlat.item():.4f} | Birkhoff: {loss_birkhoff.item():.4f} | SIGReg: {loss_sigreg.item():.4f} | Gumbel-T: {gumbel_temp:.4f} | Temp: {new_temp:.4f}")

        avg_epoch_loss = epoch_loss / steps_per_epoch
        print(f"[EPOCH COMPLETED] Epoch {epoch} | Avg Loss: {avg_epoch_loss:.6f} | Final Temp: {thermostat.get_temperature():.4f}")

    # Save unified checkpoint format
    print(f"\n[SERIALIZATION] Compiling unified programmatic blueprint to: {args.output_weights}...")
    unified_checkpoint = {
        "config": {
            "dim": args.dim,
            "depth": args.depth,
            "num_fluid_states": args.experts,
            "vocab_size": args.vocab_size
        },
        "model_state_dict": model.state_dict(),
        "translation_head_state_dict": translation_head.state_dict()
    }
    torch.save(unified_checkpoint, args.output_weights)
    print("[SUCCESS] Checkpoint unified. Transduction bridge coordinates successfully locked into binary asset.")

if __name__ == "__main__":
    execute_master_train_run()
