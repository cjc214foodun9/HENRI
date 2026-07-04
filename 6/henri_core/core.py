import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
import math
from henri_core.kernels import fused_circular_conv

def fast_orthogonal_init(tensor, gain=1.0):
    orig_dtype = tensor.dtype
    orig_device = tensor.device
    
    # Run on GPU if available to be super fast!
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    temp_tensor = torch.empty(tensor.shape, device=dev, dtype=torch.float32)
    
    rows = temp_tensor.size(0)
    cols = temp_tensor.numel() // rows
    
    if rows == 4096 and cols == 4096:
        with torch.no_grad():
            A_rand = torch.randn(16, 16, device=dev, dtype=torch.float32)
            B_rand = torch.randn(256, 256, device=dev, dtype=torch.float32)
            A, _ = torch.linalg.qr(A_rand)
            B, _ = torch.linalg.qr(B_rand)
            C = torch.kron(A, B)
            temp_tensor.copy_(C.view_as(temp_tensor) * gain)
    else:
        torch.nn.init.orthogonal_(temp_tensor, gain)
        
    with torch.no_grad():
        tensor.copy_(temp_tensor.to(device=orig_device, dtype=orig_dtype))
    return tensor

class ContinuousPhaseRouter(nn.Module):
    """
    Replaces the discrete softmax gating network of standard MoE.
    Calculates the thermodynamic resonance between the incoming HRR wave
    and the continuous phase attractors of the fluid bulk using cosine similarity.
    Integrates the three-tier Thermodynamic Step-Size Coupling Protocol:
    1. Adaptive Operator-Norm Coupling (Lipschitz step-size)
    2. Krasnoselskii-Mann Damping
    3. Tokyo-Style Langevin Balancing
    """
    def __init__(self, dim=4096, num_fluid_states=16, depth=32):
        super().__init__()
        self.dim = dim
        self.num_fluid_states = num_fluid_states
        self.depth = depth
        
        # Phase attractors: "centers of gravity" for routing
        self.phase_attractors = nn.Parameter(torch.randn(num_fluid_states, dim))
        fast_orthogonal_init(self.phase_attractors)
        
        # Thermodynamic temperature scalar controls strictness of routing
        self.beta = nn.Parameter(torch.tensor(10.0))
        
        # Learnable scaling parameters (Theorem 1) constrained to (0, 1) via sigmoid
        # Init: alpha_1_raw = 1.0986 -> sigmoid(1.0986) = 0.75
        # Init: alpha_2_raw = -1.0986 -> sigmoid(-1.0986) = 0.25
        self.alpha_1_raw = nn.Parameter(torch.tensor(1.0986))
        self.alpha_2_raw = nn.Parameter(torch.tensor(-1.0986))
        
        # Dynamic Step-Size Coupling parameters
        self.eta_0 = nn.Parameter(torch.tensor(0.1))
        self.gamma_B = nn.Parameter(torch.tensor(0.1))
        self.gamma_C = nn.Parameter(torch.tensor(0.1))
        self.L_B_base = nn.Parameter(torch.tensor(1.0))
        self.L_C_base = nn.Parameter(torch.tensor(1.0))
        
        # Krasnoselskii-Mann parameters
        self.alpha_0 = nn.Parameter(torch.tensor(0.9))
        self.kappa = nn.Parameter(torch.tensor(1.0))
        
        # Top-level weight allocations w_B and w_C
        self.w_B = nn.Parameter(torch.tensor(1.0))
        self.w_C = nn.Parameter(torch.tensor(1.0))

    @property
    def alpha_1(self):
        return torch.sigmoid(self.alpha_1_raw)
        
    @property
    def alpha_2(self):
        return torch.sigmoid(self.alpha_2_raw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, dim) or (batch_size, dim)
        has_seq_len = (x.ndim == 3)
        if not has_seq_len:
            x = x.unsqueeze(1) # (batch_size, 1, dim)
            
        x_norm = F.normalize(x, p=2, dim=-1)
        attractors_norm = F.normalize(self.phase_attractors, p=2, dim=-1)
        
        # Calculate geometric resonance (Cosine Similarity in 4096-D space)
        # resonance shape: (batch_size, seq_len, num_fluid_states)
        resonance = torch.einsum('bsd,ed->bse', x_norm, attractors_norm)
        
        # Fluid routing weights scaled by beta
        routing_weights = F.softmax(resonance * self.beta, dim=-1)
        
        if not has_seq_len:
            routing_weights = routing_weights.squeeze(1) # (batch_size, num_fluid_states)
            
        return routing_weights

    def _compute_analytical_gradients(self, z_t, zone_c_attractor):
        """
        Computes analytical gradients of E_B and E_C with respect to z_t.
        E_B(z) = 1 - sum_i w_i (z_norm * a_i)
        E_C(z) = 1 - z_norm * c_norm
        """
        # Ensure z_t has shape (batch, seq_len, dim)
        is_2d = (z_t.ndim == 2)
        if is_2d:
            z_t = z_t.unsqueeze(1)
            
        z_t_norm = F.normalize(z_t, p=2, dim=-1)
        attractors_norm = F.normalize(self.phase_attractors, p=2, dim=-1)
        
        routing_weights = self.forward(z_t)
        if is_2d:
            routing_weights = routing_weights.unsqueeze(1)
            
        resonance = torch.einsum('bsd,ed->bse', z_t_norm, attractors_norm)
        weighted_attractors = torch.einsum('bse,ed->bsd', routing_weights, attractors_norm)
        weighted_similarity = torch.sum(routing_weights * resonance, dim=-1, keepdim=True)
        
        norm_z = torch.norm(z_t, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
        grad_E_B = - (1.0 / norm_z) * (weighted_attractors - weighted_similarity * z_t_norm)
        
        if zone_c_attractor.ndim == 2:
            c_norm = F.normalize(zone_c_attractor.unsqueeze(1), p=2, dim=-1)
        else:
            c_norm = F.normalize(zone_c_attractor, p=2, dim=-1)
            
        s_C = torch.sum(z_t_norm * c_norm, dim=-1, keepdim=True)
        grad_E_C = - (1.0 / norm_z) * (c_norm - s_C * z_t_norm)
        
        if is_2d:
            grad_E_B = grad_E_B.squeeze(1)
            grad_E_C = grad_E_C.squeeze(1)
            
        return grad_E_B, grad_E_C

    def couple_thermodynamics(self, z_t, z_prev, zone_c_attractor, temperature):
        """
        Executes the three-tier Thermodynamic Step-Size Coupling Protocol.
        Returns the updated state and the routing weights.
        """
        # Ensure correct shapes
        is_2d = (z_t.ndim == 2)
        if is_2d:
            z_t = z_t.unsqueeze(1)
            z_prev = z_prev.unsqueeze(1)
            
        # Get routing weights
        routing_weights = self.forward(z_t)
        
        # Compute gradients
        grad_E_B, grad_E_C = self._compute_analytical_gradients(z_t, zone_c_attractor)
        grad_E_B_prev, grad_E_C_prev = self._compute_analytical_gradients(z_prev, zone_c_attractor)
        
        w_B_val = torch.abs(self.w_B)
        w_C_val = torch.abs(self.w_C)
        grad_E = w_B_val * grad_E_B + w_C_val * grad_E_C
        grad_E_prev = w_B_val * grad_E_B_prev + w_C_val * grad_E_C_prev
        
        # Tier 1: Adaptive Operator-Norm Lipschitz step-size
        delta_z = z_t - z_prev
        norm_delta_z = torch.sqrt(torch.sum(delta_z**2, dim=-1, keepdim=True) + 1e-12)
        delta_grad = grad_E - grad_E_prev
        norm_delta_grad = torch.sqrt(torch.sum(delta_grad**2, dim=-1, keepdim=True) + 1e-12)
        L_hat = norm_delta_grad / norm_delta_z
        
        L_bound = self.gamma_B * w_B_val * self.L_B_base + self.gamma_C * w_C_val * self.L_C_base
        eta_t = torch.abs(self.eta_0) / (torch.maximum(L_hat, L_bound) + 1e-8)
        
        # Tier 3: Tokyo-Style Precision-Weighted Langevin Balancing
        # noise variance is 2 * eta_t * temperature / (w_B + w_C + 1e-4)
        noise_std = torch.sqrt((2.0 * eta_t * temperature) / (w_B_val + w_C_val + 1e-4) + 1e-12)
        zeta = torch.randn_like(z_t)
        noise_term = noise_std * zeta
        
        G_z = z_t - eta_t * grad_E + noise_term
        
        # Tier 2: Krasnoselskii-Mann Damping
        w_norm = torch.sqrt(w_B_val**2 + w_C_val**2 + 1e-8)
        alpha_t = torch.abs(self.alpha_0) * torch.sigmoid(self.kappa / w_norm)
        
        z_next = (1.0 - alpha_t) * z_t + alpha_t * G_z
        z_next = F.normalize(z_next, p=2, dim=-1)
        
        if is_2d:
            z_next = z_next.squeeze(1)
            routing_weights = routing_weights.squeeze(1)
            
        return z_next, routing_weights

class UnitaryLinearLayer(nn.Linear):
    """
    Custom linear layer whose weights are projected back to the unitary/orthogonal manifold
    using Björck-Newton iterations: W_{k+1} = 1.5 * W_k - 0.5 * W_k * W_k^T * W_k.
    Calculations are done in FP32 to ensure precision and convergence.
    This runs entirely as fast matrix multiplications (torch.matmul) natively on GPU.
    """
    def force_unitary_manifold(self):
        with torch.no_grad():
            W = self.weight.float()
            # Björck-Newton converges quadratically for near-orthogonal matrices.
            # Run 5 iterations directly in FP32 to prevent half-precision underflow.
            for _ in range(5):
                W = 1.5 * W - 0.5 * torch.matmul(torch.matmul(W, W.t()), W)
            self.weight.copy_(W.to(dtype=self.weight.dtype))

class OrthogonalFluidExpert(nn.Module):
    """
    A single frequency modulator in the continuous MoE.
    Weights are strictly orthogonal to mimic physical phase-shifts in a waveguide,
    preserving magnitude and conserving energy.
    """
    def __init__(self, dim=4096):
        super().__init__()
        self.phase_shift = UnitaryLinearLayer(dim, dim, bias=False)
        fast_orthogonal_init(self.phase_shift.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, dim) or (batch_size, dim)
        shifted_wave = self.phase_shift(x)
        return F.normalize(shifted_wave, p=2, dim=-1)

class ThermoActiveFluidBlock(nn.Module):
    """
    FunctorFlow-compliant MoE Block with Native Natural Induction.
    Computes morphisms, Colimits, and thermodynamic stress in a single pass.
    """
    def __init__(self, dim=4096, num_fluid_states=16, lambda_boundary=10.0, depth=32):
        super().__init__()
        self.dim = dim
        self.num_fluid_states = num_fluid_states
        self.lambda_boundary = lambda_boundary
        
        # Categorical Router (Colimit weights)
        self.router = ContinuousPhaseRouter(dim, num_fluid_states, depth=depth)
        
        # Functors (Orthogonal Phase Shifts)
        self.experts = nn.ModuleList([OrthogonalFluidExpert(dim) for _ in range(num_fluid_states)])
        
        # HRR Value Projection (Circular Convolution Involution)
        self.output_binding_geometry = nn.Parameter(torch.randn(1, dim))
        fast_orthogonal_init(self.output_binding_geometry)
        
        # Learnable beta_1 parameter initialized to preserve variance: sqrt(1 - 0.75^2) ≈ 0.661437
        self.beta_1 = nn.Parameter(torch.tensor(0.661437))

    @property
    def alpha_1(self):
        return self.router.alpha_1

    @property
    def beta_2(self):
        alpha_1 = self.router.alpha_1
        alpha_2 = self.router.alpha_2
        L = self.router.depth
        beta_2 = 1.0 - alpha_2 * (alpha_1 ** (2 * L))
        return beta_2

    def _hrr_bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Fused circular convolution using Triton kernel or high-performance PyTorch fallback
        return fused_circular_conv(x, y)

    def forward(self, 
                current_wave: torch.Tensor, 
                previous_wave: torch.Tensor, 
                zone_c_attractor: torch.Tensor, 
                temperature: float):
        """
        current_wave: active state entering this layer, shape (batch, seq_len, dim) or (batch, dim)
        previous_wave: active state from previous layer, same shape
        zone_c_attractor: target attractor from database, shape (batch, dim)
        temperature: scalar Langevin heat injected
        """
        # Ensure we work with 3D tensors for seq_len processing consistency if needed
        is_2d = (current_wave.ndim == 2)
        if is_2d:
            current_wave = current_wave.unsqueeze(1)
            previous_wave = previous_wave.unsqueeze(1)
            
        batch_size, seq_len, dim = current_wave.shape
        
        # Establishing pre-norm (layer input pre-normalization)
        z_pre = F.normalize(current_wave, p=2, dim=-1)
        
        # 1. Step-size coupled contractive update and routing weights (using z_pre)
        shaken_wave, routing_weights = self.router.couple_thermodynamics(
            z_pre, previous_wave, zone_c_attractor, temperature
        )
        if shaken_wave.ndim == 2:
            shaken_wave = shaken_wave.unsqueeze(1)
        
        # 2. Execute Functors (Phase Shifts) on z_pre
        # expert_outputs shape: (batch, seq_len, dim, num_fluid_states)
        expert_outputs = torch.stack([expert(z_pre) for expert in self.experts], dim=-1)
        
        # 3. Collapse into the Superposition Wave (Colimit)
        superposition_wave = torch.einsum('bsde,bse->bsd', expert_outputs, routing_weights)
        superposition_wave = F.normalize(superposition_wave, p=2, dim=-1)

        # 4. Local Thermodynamic Stress (Natural Induction Loss)
        # Gradient of the wave across depth (between sequential layers)
        internal_stress = 0.5 * torch.sum((shaken_wave - previous_wave)**2, dim=-1)
        
        # Dirichlet boundary resonance penalty (cosine similarity with target attractor)
        # Target attractor shape is (batch, dim), need to expand to match shaken_wave
        attractor_expanded = zone_c_attractor.unsqueeze(1) if zone_c_attractor.ndim == 2 else zone_c_attractor
        # Ensure target is normalized
        attractor_norm = F.normalize(attractor_expanded, p=2, dim=-1)
        resonance = torch.sum(shaken_wave * attractor_norm, dim=-1)
        boundary_penalty = self.lambda_boundary * (1.0 - resonance)
        
        local_free_energy = internal_stress + boundary_penalty
        
        # 5. Final HRR Binding
        output_wave = self._hrr_bind(shaken_wave, self.output_binding_geometry)
        
        # Clamp beta_1 structurally to maintain contractive Banach envelope: beta_1 <= sqrt(1 - alpha_1^2)
        max_beta = torch.sqrt((1.0 - self.alpha_1 ** 2).clamp(min=0.0) + 1e-12)
        beta_1_clamped = torch.minimum(self.beta_1, max_beta)
        
        # Theorem 1 Equation (2) - strictly pre-norm and NO post-normalization
        final_wave = self.alpha_1 * current_wave + beta_1_clamped * output_wave

        if is_2d:
            final_wave = final_wave.squeeze(1)
            local_free_energy = local_free_energy.squeeze(1)
            
        return final_wave, local_free_energy

def quantize_nvfp4(x: torch.Tensor) -> torch.Tensor:
    # 1-2-1 bit float format has 1 sign bit, 2 exponent bits, 1 mantissa bit
    # S = [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0] and their negatives
    orig_device = x.device
    orig_dtype = x.dtype
    
    x_f32 = x.to(torch.float32)
    sign = torch.sign(x_f32)
    abs_x = torch.abs(x_f32)
    
    vals = torch.tensor([0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=orig_device)
    
    # Broadcast subtraction to find closest matching bins
    shape = abs_x.shape
    abs_x_flat = abs_x.view(-1, 1)
    dists = torch.abs(abs_x_flat - vals.unsqueeze(0))
    indices = torch.argmin(dists, dim=-1)
    
    quantized_flat = vals[indices] * sign.view(-1)
    return quantized_flat.view(shape).to(dtype=orig_dtype)

def quantize_complex_nvfp4(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        real_q = quantize_nvfp4(x.real)
        imag_q = quantize_nvfp4(x.imag)
        return torch.complex(real_q, imag_q)
    else:
        return quantize_nvfp4(x)

class ProprietaryHENRICore(nn.Module):
    """
    The Optoelectronic Reasoning Engine
    Supports both 8.59B legacy configuration and the new 100M disaggregated looped recurrent swarm.
    """
    def __init__(self, dim=4096, depth=32, num_fluid_states=16, max_loops=10, fp_threshold=0.1, looped_recurrent=True):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.max_loops = max_loops
        self.fp_threshold = fp_threshold
        self.looped_recurrent = looped_recurrent
        
        # Initialize Three-Tier Nested Kuramoto Synchronization Grid
        import sys
        import os
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        from hierarchical_sync_core import HenriHierarchicalSyncCore
        from chimera_phase_gate import HenriChimeraPhaseGating
        self.hierarchical_sync = HenriHierarchicalSyncCore(num_oscillators=dim, num_layers=depth, num_experts=num_fluid_states)
        self.chimera_gate = HenriChimeraPhaseGating(num_oscillators=dim)
        
        # Instantiate layers. If looped_recurrent, we will only use self.layers[0] recursively to keep parameter size compact (~100M).
        num_layers_to_instantiate = 1 if self.looped_recurrent else depth
        self.layers = nn.ModuleList([ThermoActiveFluidBlock(dim, num_fluid_states, depth=depth) for _ in range(num_layers_to_instantiate)])
        self.final_layer_norm = nn.LayerNorm(dim)
        self.gradient_checkpointing = False
        
        # Depth-wise 2D convolution for localized spatial grid coordinate bias (3x3 kernel)
        # Assuming channels = 1. Handles the 2D grid dimensions based on dim.
        self.depthwise_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=False
        )
        nn.init.dirac_(self.depthwise_conv.weight)

    def apply_depthwise_conv(self, x: torch.Tensor) -> torch.Tensor:
        # Reshapes wave vector to spatial grid and applies 2D depthwise convolution
        orig_shape = x.shape
        is_3d = (x.ndim == 3)
        if is_3d:
            batch, seq_len, dim = x.shape
            x_flat = x.reshape(batch * seq_len, dim)
        else:
            x_flat = x
            dim = x.shape[-1]
            
        if dim == 4096:
            grid = x_flat.view(-1, 1, 64, 64)
        elif dim == 2048:
            grid = x_flat.view(-1, 1, 32, 64)
        elif dim == 1024:
            grid = x_flat.view(-1, 1, 32, 32)
        else:
            h = int(math.sqrt(dim))
            w = dim // h
            if h * w == dim:
                grid = x_flat.view(-1, 1, h, w)
            else:
                return x # Skip if cannot reshape cleanly
                
        orig_dtype = grid.dtype
        conv_dtype = self.depthwise_conv.weight.dtype
        grid_conv = grid.to(dtype=conv_dtype)
        conv_out = self.depthwise_conv(grid_conv).to(dtype=orig_dtype)
        res = conv_out.view(orig_shape)
        return F.normalize(res, p=2, dim=-1)

    def forward(self, x: torch.Tensor, zone_c_attractor: torch.Tensor, temperature: float, expert_activations: torch.Tensor = None, nudge_context: torch.Tensor = None):
        """
        x: Ingested HRR wave vector, shape (batch, seq_len, dim) or (batch, dim)
        zone_c_attractor: Target Attractor from TimescaleDB, shape (batch, dim)
        temperature: Langevin heat injected by thermostat
        expert_activations: Continuous activations from MoE experts (e.g. from routing_weights or dummy)
        nudge_context: Phase nudge pull vector [Batch, 4096] (for Equilibrium Propagation)
        """
        # Handle shape for energy accumulator
        is_3d = (x.ndim == 3)
        batch_size = x.shape[0]
        seq_len = x.shape[1] if is_3d else 1
        
        # Extract initial phase angles from the complex input wavefront to work in the real phase domain
        if torch.is_complex(x):
            z_current = torch.angle(x)
        elif x.ndim == 3 and x.size(-1) == 2:
            z_current = torch.atan2(x[..., 1], x[..., 0])
        else:
            z_current = x.clone()
        z_current = z_current.to(dtype=self.layers[0].beta_1.dtype)
        
        # Transduce complex target attractors to the real phase domain
        if zone_c_attractor is not None:
            if torch.is_complex(zone_c_attractor):
                zone_c_attractor = torch.angle(zone_c_attractor)
            elif zone_c_attractor.ndim == 3 and zone_c_attractor.size(-1) == 2:
                zone_c_attractor = torch.atan2(zone_c_attractor[..., 1], zone_c_attractor[..., 0])
            zone_c_attractor = zone_c_attractor.to(dtype=self.layers[0].beta_1.dtype)
        
        # Ensure we have expert_activations of shape [Batch, 16]
        if expert_activations is None:
            # We can use the first layer's router to get initial routing activations
            first_router = self.layers[0].router
            # Cast input to match router parameter dtype
            first_router_in = z_current.mean(dim=1) if is_3d else z_current
            expert_activations = first_router(first_router_in)
        total_system_energy = torch.zeros(batch_size, seq_len, device=x.device, dtype=x.dtype) if is_3d else torch.zeros(batch_size, device=x.device, dtype=x.dtype)
        
        # Let's run the 32 continuous diffractive layers unrolling block
        for step_idx in range(self.depth):
            previous_wave = z_current
            layer = self.layers[0] if self.looped_recurrent else self.layers[step_idx]
            
            # Apply 2D Depthwise convolution at the start of the unrolling loop
            if step_idx == 0:
                z_current = self.apply_depthwise_conv(z_current)
                
            # Forward pass through the active fluid layer (computes phase shifts on the experts)
            if getattr(self, "gradient_checkpointing", False) and self.training:
                def create_checkpoint_fn(block):
                    def checkpoint_fn(c_wave, p_wave, z_attractor, t_val):
                        return block(c_wave, p_wave, z_attractor, t_val.item())
                    return checkpoint_fn

                temp_tensor = torch.tensor(temperature, device=x.device, requires_grad=False)
                z_current, local_energy = torch.utils.checkpoint.checkpoint(
                    create_checkpoint_fn(layer),
                    z_current,
                    previous_wave,
                    zone_c_attractor,
                    temp_tensor,
                    use_reentrant=False
                )
            else:
                z_current, local_energy = layer(z_current, previous_wave, zone_c_attractor, temperature)
            
            # Accumulate variational free energy
            total_system_energy += local_energy
            
        # Meso-to-Macro / Settle phase landscape via Kuramoto Hierarchical Sync
        z_2d = z_current.mean(dim=1) if is_3d else z_current
        sync_wave, layer_order_params, zone_phases = self.hierarchical_sync(
            z_2d, expert_activations, timesteps=50, dt=0.02, temperature=temperature, nudge_context=nudge_context
        )
        
        # Spatial Chimera Phase Gating (Rigor/Flux split)
        sync_wave_unrolled = torch.stack([sync_wave.real, sync_wave.imag], dim=-1)
        chimera_wave_unrolled, r_rigor, r_flux = self.chimera_gate(
            sync_wave_unrolled, timesteps=80, dt=0.03, flux_temperature=temperature
        )
        chimera_wave = torch.complex(chimera_wave_unrolled[..., 0], chimera_wave_unrolled[..., 1])
        
        # Keep the telemetry parameters accessible on self
        self.latest_R_macro = sum(layer_order_params) / len(layer_order_params) if layer_order_params else 1.0
        self.latest_r_rigor = r_rigor.mean().item()
        self.latest_r_flux = r_flux.mean().item()
        self.latest_zone_phases = zone_phases.detach()
        
        # Reshape back to original shape
        if is_3d:
            final_output = chimera_wave.unsqueeze(1).repeat(1, x.shape[1], 1)
        else:
            final_output = chimera_wave
            
        final_output = F.normalize(final_output, p=2, dim=-1)
        
        # final LayerNorm (apply to real/imag separately to support ComplexFloat)
        norm_real = self.final_layer_norm(final_output.real.to(dtype=self.layers[0].beta_1.dtype))
        norm_imag = self.final_layer_norm(final_output.imag.to(dtype=self.layers[0].beta_1.dtype))
        final_output = torch.complex(norm_real.float(), norm_imag.float())
        return final_output, total_system_energy.mean()

    @torch.no_grad()
    def orthonormalize_experts(self):
        """
        Applies Björck-Newton iterations to enforce Stiefel manifold invariants
        on all expert linear layers across all fluid blocks.
        """
        for layer_block in self.layers:
            if hasattr(layer_block, 'experts'):
                for expert in layer_block.experts:
                    if hasattr(expert, 'phase_shift') and hasattr(expert.phase_shift, 'force_unitary_manifold'):
                        expert.phase_shift.force_unitary_manifold()
