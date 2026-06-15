import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

class ContinuousPhaseRouter(nn.Module):
    """
    Replaces the discrete softmax gating network of standard MoE.
    Calculates the thermodynamic resonance between the incoming HRR wave
    and the continuous phase attractors of the fluid bulk using cosine similarity.
    """
    def __init__(self, dim=4096, num_fluid_states=16):
        super().__init__()
        self.dim = dim
        self.num_fluid_states = num_fluid_states
        
        # Phase attractors: "centers of gravity" for routing
        self.phase_attractors = nn.Parameter(torch.randn(num_fluid_states, dim))
        nn.init.orthogonal_(self.phase_attractors)
        
        # Thermodynamic temperature scalar controls strictness of routing
        self.beta = nn.Parameter(torch.tensor(10.0))

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

class UnitaryLinearLayer(nn.Linear):
    """
    Custom linear layer whose weights are projected back to the unitary/orthogonal manifold
    using Björck-Newton iterations: W_{k+1} = 1.5 * W_k - 0.5 * W_k * W_k^T * W_k.
    This runs entirely as fast matrix multiplications (torch.matmul) natively on GPU.
    """
    def force_unitary_manifold(self):
        with torch.no_grad():
            W = self.weight
            # Björck-Newton converges quadratically for near-orthogonal matrices.
            # Run 5 iterations directly in the current precision and device.
            for _ in range(5):
                W.copy_(1.5 * W - 0.5 * torch.matmul(torch.matmul(W, W.t()), W))

class OrthogonalFluidExpert(nn.Module):
    """
    A single frequency modulator in the continuous MoE.
    Weights are strictly orthogonal to mimic physical phase-shifts in a waveguide,
    preserving magnitude and conserving energy.
    """
    def __init__(self, dim=4096):
        super().__init__()
        self.phase_shift = UnitaryLinearLayer(dim, dim, bias=False)
        nn.init.orthogonal_(self.phase_shift.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, dim) or (batch_size, dim)
        shifted_wave = self.phase_shift(x)
        return F.normalize(shifted_wave, p=2, dim=-1)

class ThermoActiveFluidBlock(nn.Module):
    """
    FunctorFlow-compliant MoE Block with Native Natural Induction.
    Computes morphisms, Colimits, and thermodynamic stress in a single pass.
    """
    def __init__(self, dim=4096, num_fluid_states=16, lambda_boundary=10.0):
        super().__init__()
        self.dim = dim
        self.num_fluid_states = num_fluid_states
        self.lambda_boundary = lambda_boundary
        
        # Categorical Router (Colimit weights)
        self.router = ContinuousPhaseRouter(dim, num_fluid_states)
        
        # Functors (Orthogonal Phase Shifts)
        self.experts = nn.ModuleList([OrthogonalFluidExpert(dim) for _ in range(num_fluid_states)])
        
        # HRR Value Projection (Circular Convolution Involution)
        self.output_binding_geometry = nn.Parameter(torch.randn(1, dim))
        nn.init.orthogonal_(self.output_binding_geometry)

    def _hrr_bind(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Internal FFT circular convolution binding
        X_freq = torch.fft.rfft(x, dim=-1)
        Y_freq = torch.fft.rfft(y, dim=-1)
        z = torch.fft.irfft(X_freq * Y_freq, n=self.dim, dim=-1)
        return F.normalize(z, p=2, dim=-1)

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
        
        # 1. Calculate Colimit routing weights
        routing_weights = self.router(current_wave) # (batch, seq_len, num_fluid_states)
        
        # 2. Execute Functors (Phase Shifts)
        # expert_outputs shape: (batch, seq_len, dim, num_fluid_states)
        expert_outputs = torch.stack([expert(current_wave) for expert in self.experts], dim=-1)
        
        # 3. Collapse into the Superposition Wave (Colimit)
        superposition_wave = torch.einsum('bsde,bse->bsd', expert_outputs, routing_weights)
        superposition_wave = F.normalize(superposition_wave, p=2, dim=-1)
        
        # 4. Langevin Noise Injection (thermodynamic shaking)
        if temperature > 0.0:
            langevin_noise = torch.randn_like(superposition_wave) * temperature
            shaken_wave = superposition_wave + langevin_noise
            shaken_wave = F.normalize(shaken_wave, p=2, dim=-1)
        else:
            shaken_wave = superposition_wave

        # 5. Local Thermodynamic Stress (Natural Induction Loss)
        # Gradient of the wave across depth (between sequential layers)
        internal_stress = 0.5 * torch.norm(shaken_wave - previous_wave, p=2, dim=-1)**2
        
        # Dirichlet boundary resonance penalty (cosine similarity with target attractor)
        # Target attractor shape is (batch, dim), need to expand to match shaken_wave
        attractor_expanded = zone_c_attractor.unsqueeze(1) if is_2d or zone_c_attractor.ndim == 2 else zone_c_attractor
        # Ensure target is normalized
        attractor_norm = F.normalize(attractor_expanded, p=2, dim=-1)
        resonance = torch.sum(shaken_wave * attractor_norm, dim=-1)
        boundary_penalty = self.lambda_boundary * (1.0 - resonance)
        
        local_free_energy = internal_stress + boundary_penalty
        
        # 6. Final HRR Binding (Residual Connection)
        output_wave = self._hrr_bind(shaken_wave, self.output_binding_geometry)
        final_wave = F.normalize(current_wave + output_wave, p=2, dim=-1)

        if is_2d:
            final_wave = final_wave.squeeze(1)
            local_free_energy = local_free_energy.squeeze(1)
            
        return final_wave, local_free_energy

class ProprietaryHENRICore(nn.Module):
    """
    The 7B Parameter Optoelectronic Reasoning Engine
    """
    def __init__(self, dim=4096, depth=32, num_fluid_states=16):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList([ThermoActiveFluidBlock(dim, num_fluid_states) for _ in range(depth)])
        self.final_layer_norm = nn.LayerNorm(dim)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor, zone_c_attractor: torch.Tensor, temperature: float):
        """
        x: Ingested HRR wave vector, shape (batch, seq_len, dim) or (batch, dim)
        zone_c_attractor: Target Attractor from TimescaleDB, shape (batch, dim)
        temperature: Langevin heat injected by thermostat
        """
        # Handle shape for energy accumulator
        if x.ndim == 3:
            total_system_energy = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        else:
            total_system_energy = torch.zeros(x.shape[0], device=x.device)
            
        current_wave = x
        for layer in self.layers:
            previous_wave = current_wave
            
            if getattr(self, "gradient_checkpointing", False) and self.training:
                # Wrap forward pass of block in checkpoint
                def create_checkpoint_fn(block):
                    def checkpoint_fn(c_wave, p_wave, z_attractor, temp_val):
                        return block(c_wave, p_wave, z_attractor, temp_val.item())
                    return checkpoint_fn

                # temperature must be a tensor to work with torch.utils.checkpoint
                temp_tensor = torch.tensor(temperature, device=x.device, requires_grad=False)
                current_wave, local_energy = torch.utils.checkpoint.checkpoint(
                    create_checkpoint_fn(layer),
                    current_wave,
                    previous_wave,
                    zone_c_attractor,
                    temp_tensor,
                    use_reentrant=False
                )
            else:
                current_wave, local_energy = layer(current_wave, previous_wave, zone_c_attractor, temperature)
                
            total_system_energy += local_energy
            
        final_output = self.final_layer_norm(current_wave)
        return final_output, total_system_energy.mean()
