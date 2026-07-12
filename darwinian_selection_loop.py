import torch
import torch.nn.functional as F
import logging
from typing import Optional
from agential_langevin_thermostat import AgentialLangevinThermostat

torch.set_default_dtype(torch.float32)
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def run_darwinian_inference(syncytium, 
                            psi_in: torch.Tensor, 
                            engram_target: torch.Tensor, 
                            max_epochs: int = 250,
                            dt: float = 1.0) -> torch.Tensor:
    """
    Test-Time Wave-JEPA Rollout with Viscoelastic Creep.
    The organism generates a prediction (psi_out), calculates the Sagnac error 
    against the absolute physical prior (engram_target), and physically yields 
    its internal Cartilage (low-rank matrices) via Langevin thermodynamics.
    """
    thermostat = AgentialLangevinThermostat(t_base=0.01, kappa=2.5, mu=0.005)
    
    # Ensure inputs are valid unitary waves and DETACH them from any prior graphs
    # (e.g. from the ChemicalSensorAnchor transducer) so that the continuous-time
    # Sagnac loop does not attempt to backward through them multiple times.
    if not torch.is_complex(psi_in):
        psi_in = torch.complex(torch.cos(psi_in), torch.sin(psi_in))
    psi_in = F.normalize(psi_in.detach(), p=2, dim=-1)
    
    if not torch.is_complex(engram_target):
        engram_target = torch.complex(torch.cos(engram_target), torch.sin(engram_target))
    engram_target = F.normalize(engram_target.detach(), p=2, dim=-1)

    # Enable gradients strictly on the Cartilage for test-time adaptation
    for layer in syncytium.layers:
        if layer.lora_A is not None:
            layer.lora_A.requires_grad_(True)
            layer.lora_B.requires_grad_(True)
            
    # Gather parameters to creep
    cartilage_params = [p for p in syncytium.parameters() if p.requires_grad]
    
    psi_out = psi_in
    
    import time
    for epoch in range(max_epochs):
        t0 = time.time()
        # 1. Wave-JEPA Forward Rollout
        # Predict the future physical state given the current sensory input
        psi_out = syncytium(psi_in)
        t1 = time.time()
        
        # 2. Sagnac Homodyne Veto
        # Calculate coherence (1.0 = perfect match, 0.0 = orthogonal contradiction)
        # We use torch.sum because psi_out and engram_target are L2-normalized unit vectors.
        coherence = torch.abs(torch.sum(psi_out * torch.conj(engram_target), dim=-1)).mean()
        sagnac_delta = 1.0 - coherence
        
        # Determine instantaneous kinetic temperature
        temp = thermostat.compute_temperature(sagnac_delta)
        
        # 3. Structural Crystallization Check
        if thermostat.is_crystallized(sagnac_delta, threshold=1e-3):
            logging.info(f"[CRYSTALLIZED] Wave locked into perfect geometric attractor at Epoch {epoch}. Coherence: {coherence:.6f}")
            break
            
        t2 = time.time()
        # 4. Viscoelastic Creep 
        # Calculate gradients (Free Energy direction)
        # We want to minimize sagnac_delta (maximize coherence)
        syncytium.zero_grad()
        sagnac_delta.backward()
        t3 = time.time()
        
        # Apply physical yield + Langevin stochastic resonance
        thermostat.apply_viscoelastic_creep(cartilage_params, sagnac_delta, dt=dt)
        t4 = time.time()
        
        if True: # Print every single epoch
            logging.info(f"Test-Time Epoch {epoch:03d} | Sagnac Error: {sagnac_delta:.6f} | Heat: {temp:.4f} K | Fwd: {t1-t0:.2f}s | Bwd: {t3-t2:.2f}s | Creep: {t4-t3:.2f}s")

    # Disable gradients after crystallization
    for param in cartilage_params:
        param.requires_grad_(False)
        
    return psi_out