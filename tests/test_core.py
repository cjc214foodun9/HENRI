import pytest
import torch
import torch.nn as nn
from henri_core.hrr import HRRInputLayer
from henri_core.core import ContinuousPhaseRouter, OrthogonalFluidExpert, ProprietaryHENRICore
from henri_core.thermodynamics import NaturalInductionLoss, DivergentMaster

def test_hrr_binding_properties():
    dim = 1024  # use a smaller dimension for fast testing
    layer = HRRInputLayer(dim=dim)
    
    # Generate random orthogonal vectors
    x = torch.randn(1, dim)
    y = torch.randn(1, dim)
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    y = torch.nn.functional.normalize(y, p=2, dim=-1)
    
    # 1. Bind
    z = layer.bind(x, y)
    assert z.shape == (1, dim)
    assert torch.allclose(torch.norm(z, p=2, dim=-1), torch.tensor([1.0]), atol=1e-5)
    
    # 2. Unbind
    y_recovered = layer.unbind(z, x)
    assert y_recovered.shape == (1, dim)
    assert torch.allclose(torch.norm(y_recovered, p=2, dim=-1), torch.tensor([1.0]), atol=1e-5)
    
    # Test that cosine similarity between recovered y and true y is positive/high (approximate recovery)
    similarity = torch.sum(y_recovered * y, dim=-1).item()
    # In VSA/HRR, binding with another vector behaves like noise, but unbinding recovers key.
    # We expect positive similarity.
    assert similarity > 0.1

def test_continuous_phase_router():
    dim = 256
    num_fluid_states = 8
    router = ContinuousPhaseRouter(dim=dim, num_fluid_states=num_fluid_states)
    
    x = torch.randn(2, 5, dim) # batch=2, seq_len=5
    weights = router(x)
    
    assert weights.shape == (2, 5, num_fluid_states)
    assert torch.allclose(torch.sum(weights, dim=-1), torch.ones(2, 5), atol=1e-5)

def test_orthogonal_expert():
    dim = 512
    expert = OrthogonalFluidExpert(dim=dim)
    
    # Verify weights are orthogonal
    W = expert.phase_shift.weight
    identity = torch.matmul(W, W.t())
    expected_identity = torch.eye(dim, device=W.device)
    assert torch.allclose(identity, expected_identity, atol=1e-3)
    
    x = torch.randn(3, dim)
    x = torch.nn.functional.normalize(x, p=2, dim=-1)
    x_out = expert(x)
    assert torch.allclose(torch.norm(x_out, p=2, dim=-1), torch.ones(3), atol=1e-5)

def test_thermodynamics_loss_and_master():
    dim = 1024
    loss_fn = NaturalInductionLoss(lambda_boundary=10.0, reg_coefficient=0.5, dim=dim)
    
    # batch=2, depth=4, dim=1024
    wave_trajectory = torch.randn(2, 4, dim)
    zone_c_attractors = torch.randn(2, dim)
    
    loss = loss_fn(wave_trajectory, zone_c_attractors, temperature=1.5)
    assert loss.ndim == 0
    assert not torch.isnan(loss)
    
    # Test DivergentMaster dynamics
    thermostat = DivergentMaster(t_min=0.0, t_max=5.0, cooling_rate=0.1, heat_sensitivity=1.0)
    
    # Expect temperature to cool down if energy is low and constant
    temp1 = thermostat.step(0.01)
    temp2 = thermostat.step(0.005)
    assert temp2 <= temp1
    
    # Simulate a logic lock (high energy, stagnated change)
    for _ in range(10):
        temp = thermostat.step(2.0)
    # The thermostat should trigger a shock and return high temperature
    assert temp > 1.0
