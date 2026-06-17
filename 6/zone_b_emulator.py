import torch
import torch.nn as nn
import math

class AngularSpectrumPropagator(nn.Module):
    """
    Simulates physical free-space light diffraction between BTO phase masks.
    Calculates the exact Helmholtz transfer function to prevent aliasing.
    Removes in-place modifications to preserve Autograd activation snapshots cleanly.
    """
    def __init__(self, N: int, length: float, wavelength: float, z: float, device='cpu'):
        super().__init__()
        # Calculate spatial frequencies on CPU to avoid DirectML complex/unsupported op errors during init
        dx = length / N
        fx = torch.fft.fftfreq(N, d=dx, device='cpu')
        fy = torch.fft.fftfreq(N, d=dx, device='cpu')
        FX, FY = torch.meshgrid(fx, fy, indexing='ij')
        
        # Free-space transfer function argument
        k = 2 * math.pi / wavelength
        argument = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2
        
        # evanescent wave filter (eliminates non-propagating waves to save compute)
        evanescent_filter = (argument > 0).float()
        argument = torch.clamp(argument, min=0.0)
        
        phase = k * z * torch.sqrt(argument)
        
        # Precompute the complex transfer function H(f_x, f_y)
        H = torch.polar(evanescent_filter, phase)
        # Store real and imaginary parts separately as real buffers to prevent DirectML complex type crashes
        self.register_buffer('H_real', H.real.to(device))
        self.register_buffer('H_imag', H.imag.to(device))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # Reconstruct complex H on the wave's device (which is CPU)
        dev = wave.device
        H = torch.complex(self.H_real.to(dev), self.H_imag.to(dev))
        # 1. Transform to spatial frequency domain via Orthonormal 2D FFT
        wave_fft = torch.fft.fft2(wave, norm='ortho')
        # 2. Out-of-place multiplication ensures Autograd retains clean forward state access
        wave_fft_modulated = wave_fft * H
        # 3. Transform back to the physical spatial domain
        return torch.fft.ifft2(wave_fft_modulated, norm='ortho')


class OpticalD2NNLayer(nn.Module):
    """
    A single BTO crystal cross-section.
    Acts as a learnable phase mask during HITL training.
    Utilizes Bounded Phase Perturbation to prevent gradient dead zones.
    """
    def __init__(self, N: int, device='cpu', max_phase_depth=0.1):
        super().__init__()
        self.N = N
        # Bounded Physical Phase Perturbation (Independent of N, prevents Xavier collapse)
        init_bounds = max_phase_depth * math.pi
        self.phase_mask = nn.Parameter(
            torch.empty((N, N), dtype=torch.float32, device=device).uniform_(-init_bounds, init_bounds)
        )
        
    def inject_langevin_noise(self, heat: float):
        """Thermodynamic Annealing: physically shakes the local BTO topology."""
        if heat > 0.0:
            with torch.no_grad():
                noise = torch.randn_like(self.phase_mask) * math.sqrt(heat)
                self.phase_mask.add_(noise)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        # Cast phase_mask from DirectML/GPU to CPU (wave's device) to avoid ComplexFloat errors on DirectML
        dev = wave.device
        mask = self.phase_mask.to(dev)
        # Calculate complex transmission coefficient t = exp(i * phi) (out of place, no modulo)
        t = torch.polar(torch.ones_like(mask), mask)
        # Out-of-place element-wise wave modulation
        return wave * t


class SagnacInterferometer(nn.Module):
    """
    Simulates the physical Sagnac loops that test logical consistency.
    Constructive interference = Truth; Destructive interference = Hallucination.
    """
    def __init__(self):
        super().__init__()
        import sys
        import os
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root_dir not in sys.path:
            sys.path.append(root_dir)
        from sagnac_veto import SagnacInterferometer
        self.sagnac = SagnacInterferometer()

    def forward(self, current_wave: torch.Tensor, target_manifold: torch.Tensor):
        return self.sagnac(current_wave, target_manifold)


class ZoneBEmulator(nn.Module):
    """
    The 200-Million Parameter Software-Defined Physics Engine.
    """
    def __init__(self, resolution_scale: float = 1.0, length=0.016, wavelength=1550e-9, z=0.001, device='cpu'):
        super().__init__()
        self.N = int(6324 * resolution_scale)
        self.device = device
        
        # 5 cascading Diffractive Deep Neural Network (D2NN) layers
        self.layers = nn.ModuleList([OpticalD2NNLayer(self.N, device) for _ in range(5)])
        # Free-space propagation between the layers
        self.propagators = nn.ModuleList([AngularSpectrumPropagator(self.N, length, wavelength, z, device) for _ in range(4)])
        # The final Logic Gate
        self.sagnac_veto = SagnacInterferometer()

    def set_microheaters(self, langevin_heat: float):
        """Pulses heat into all 5 crystal phase masks to escape logical minima."""
        for layer in self.layers:
            layer.inject_langevin_noise(langevin_heat)

    def forward(self, wave: torch.Tensor, target_manifold: torch.Tensor, langevin_heat: float = 0.0):
        # 1. Check if the Zone A Divergent Master is calling for a thermodynamic shake
        if langevin_heat > 0.0:
            self.set_microheaters(langevin_heat)

        # 2. Prevent mutating the input tensor if grad is required by Zone A
        current_wave = wave.clone() 

        # 3. Propagate light through the 5 D2NN boundaries
        for i in range(5):
            current_wave = self.layers[i](current_wave)
            if i < 4:
                current_wave = self.propagators[i](current_wave)
                
        # 4. Geometrically verify the resulting thought against the Zone C Axiom
        transmission_truth, reflection_delta, error_energy = self.sagnac_veto(current_wave, target_manifold)
        
        return transmission_truth, reflection_delta, error_energy


class ZoneBPhysicalEmulator:
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator

    def evaluate_wavefront(self, candidate_code: str, target_label: str = "SCADA_Pressure_Control") -> tuple:
        """
        Projects candidate code to wave space, propagates it through Zone B D2NN emulator,
        and checks Dirichlet/Neumann boundaries using the BoundaryAxiomValidator.
        """
        import torch
        import numpy as np
        
        with torch.no_grad():
            # 1. Project candidate solution to 4096-D complex wave using L3 Router
            emb_res = self.orchestrator.base_model.create_embedding(candidate_code)
            h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
            h_7b_lora = self.orchestrator.lora_managers[0].apply_lora(h_7b_raw)
            if len(h_7b_lora.shape) == 2:
                h_7b_lora = torch.mean(h_7b_lora, dim=0)
                
            is_symbolic = any(kw in target_label.lower() or kw in candidate_code.lower() for kw in ["symbolic", "mathematical", "derivation", "weyl", "anomaly", "expansion", "coefficient", "thermodynamic", "conservation", "coeff"])
            
            if is_symbolic:
                psi_candidate_focused = self.orchestrator.l3_router.activation_to_wave(h_7b_lora)
                if len(psi_candidate_focused.shape) == 2:
                    psi_candidate_focused = torch.mean(psi_candidate_focused, dim=0)
                psi_candidate_focused = psi_candidate_focused.reshape(64, 64)
            else:
                # Replicate candidate activation across all 16 streams to create [16, 1, gemma_dim]
                activations_stack = h_7b_lora.unsqueeze(0).unsqueeze(0).repeat(16, 1, 1)
                psi_candidate, _, _ = self.orchestrator.l3_router(activations=activations_stack)
                
                # Apply lens downsampling
                N = psi_candidate.size(-1)
                x = torch.linspace(-1.0, 1.0, N, device=psi_candidate.device)
                y = torch.linspace(-1.0, 1.0, N, device=psi_candidate.device)
                X, Y = torch.meshgrid(x, y, indexing='ij')
                lens_phase = -50.0 * (X**2 + Y**2)
                lens = torch.polar(torch.ones_like(lens_phase), lens_phase)
                
                wave_lensed = psi_candidate.squeeze(0) * lens
                focal_plane = torch.fft.fft2(wave_lensed, norm='ortho')
                focal_plane_shifted = torch.fft.fftshift(focal_plane)
                
                start = (N - 64) // 2
                end = start + 64
                focused_64 = focal_plane_shifted[start:end, start:end]
                
                mags = torch.abs(focused_64).clamp(min=1e-8)
                psi_candidate_focused = focused_64 / mags

            # 3. Fire the wave in Zone B
            target_vector = self.orchestrator.hopfield.vocabulary.get(target_label)
            if target_vector is None:
                target_vector = self.orchestrator.get_stream_address(0)
                
            target_np = target_vector.detach().cpu().numpy().astype(np.complex64)
            psi_candidate_flat = psi_candidate_focused.flatten()
            
            # Query memory cache and blend history
            retrieved_wave = self.orchestrator.memory_engines[0].retrieve_from_cache(query_key=psi_candidate_flat)
            retrieved_wave = retrieved_wave.to(psi_candidate_flat.device)
            blended_focused = psi_candidate_flat + retrieved_wave
            blended_mags = torch.abs(blended_focused).clamp(min=1e-8)
            psi_candidate_resolved = blended_focused / blended_mags
            
            psi_cand_np = psi_candidate_resolved.detach().cpu().numpy().astype(np.complex64)

            truth_np, delta_np, alignment = self.orchestrator.optical_core.forward(
                hr_wavefront=psi_cand_np,
                target_manifold=target_np,
                langevin_heat=0.0
            )

            # 4. Boundary Validation
            truth_tensor = torch.tensor(truth_np, dtype=torch.complex64, device=self.orchestrator.optical_core.device)
            is_valid, veto_reason, error_energy, h_cft = self.orchestrator.boundary_validator.validate_boundary(truth_tensor)
            
            if not is_valid:
                feedback = f"Sagnac Veto: Dirichlet boundary axioms violated. Reason: {veto_reason} | Error Energy: {error_energy:.4f}"
                return False, feedback, error_energy, truth_tensor, delta_np
            
            return True, "Dirichlet boundaries verified. Sagnac alignment achieved.", error_energy, truth_tensor, delta_np
