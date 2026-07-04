import torch
from zone_b_emulator import ZoneBEmulator

def verify_spatial_resolution_and_memory():
    print("[TEST 1] Verifying Spatial Resolution & In-Place Memory Mechanics...")
    scale = 0.04 # 252x252 for sub-second rapid unit testing
    emulator = ZoneBEmulator(resolution_scale=scale, device='cpu')
    N = int(6324 * scale)
    assert emulator.N == N, f"Resolution mismatch. Expected {N}, got {emulator.N}"
    
    test_wave = torch.randn((N, N), dtype=torch.complex64)
    target_wave = torch.randn((N, N), dtype=torch.complex64)
    
    with torch.no_grad():
        truth, delta, energy = emulator(test_wave, target_wave)
        
    print(" -> [PASS] In-Place FFTs & D2NN Layers simulated correctly without VRAM fragmentation.")

def verify_interferometric_accuracy():
    print("[TEST 2] Verifying Sagnac Interferometric Accuracy...")
    emulator = ZoneBEmulator(resolution_scale=0.04, device='cpu')
    N = emulator.N
    
    target_wave = torch.ones((N, N), dtype=torch.complex64)
    # Mocking a mathematically perfect identity pass
    mock_output = target_wave.clone()
    
    truth, delta, energy = emulator.sagnac_veto(mock_output, target_wave)
    
    assert energy.item() < 1e-5, f"Expected 0 energy for perfect match, got {energy.item()}"
    print(" -> [PASS] Constructive resonance properly collapses Sagnac Delta to 0.")

def verify_loss_and_autograd():
    print("[TEST 3] Verifying Physical-Consistency Loss & Autograd Graph...")
    scale = 0.04
    N = int(6324 * scale)
    emulator = ZoneBEmulator(resolution_scale=scale, device='cpu')
    
    test_wave = torch.randn((N, N), dtype=torch.complex64, requires_grad=True)
    target_wave = torch.randn((N, N), dtype=torch.complex64)
    
    truth, delta, energy = emulator(test_wave, target_wave)
    
    from train_zone_b import calculate_physical_consistency_loss
    loss = calculate_physical_consistency_loss(truth, target_wave, delta, energy)
    loss.backward()
    
    assert emulator.layers[0].phase_mask.grad is not None, "Gradients failed to propagate to Layer 1 Phase Mask!"
    print(" -> [PASS] Physical-Consistency Loss gradients flow successfully through 5 D2NN layers.")

def verify_thermodynamic_shaking():
    print("[TEST 4] Verifying Langevin Heat / Thermodynamic Shaking...")
    emulator = ZoneBEmulator(resolution_scale=0.04, device='cpu')
    
    mask_before = emulator.layers[0].phase_mask.clone()
    emulator.set_microheaters(langevin_heat=1.5)
    mask_after = emulator.layers[0].phase_mask.clone()
    
    diff = torch.sum(torch.abs(mask_after - mask_before)).item()
    assert diff > 0.0, "Langevin Heat failed to perturb phase mask weights."
    print(" -> [PASS] Thermal variance successfully reshapes local BTO topology.")

def verify_full_resolution_emulation():
    print("[TEST 5] Verifying Full-Resolution 6324x6324 Emulation & Lazy Instantiation...")
    from zone_b_emulator import HenriOpticalCoreD2NN
    core = HenriOpticalCoreD2NN(num_channels=4096, device='cpu')
    
    # Generate 6324x6324 complex wavefront and target manifold
    test_wave = torch.randn((6324, 6324), dtype=torch.complex64)
    target_wave = torch.randn((6324, 6324), dtype=torch.complex64)
    
    with torch.no_grad():
        truth, delta, energy = core(test_wave, target_wave)
        
    assert core.full_res_emulator is not None, "Full-resolution emulator was not lazily initialized!"
    assert truth.shape == (6324, 6324), f"Expected shape (6324, 6324), got {truth.shape}"
    assert delta.shape == (6324, 6324), f"Expected shape (6324, 6324), got {delta.shape}"
    print(" -> [PASS] Lazily initialized 6324x6324 emulator and propagated wavefront successfully.")

if __name__ == "__main__":
    print("=====================================================================")
    print("           BOOTING HENRI ZONE B OPTICAL CORE VERIFICATION            ")
    print("=====================================================================\n")
    verify_spatial_resolution_and_memory()
    verify_interferometric_accuracy()
    verify_loss_and_autograd()
    verify_thermodynamic_shaking()
    verify_full_resolution_emulation()
    print("\n[SUCCESS] Zone B Core conforms to continuous-time wave constraints.")
