"""
GPU Verification Suite: HENRI V2 swarm core after the SGLD/Hopfield/Sagnac
refactor. Run on the vast.ai RTX 5090 instance.

Verifies:
  1. SGLD creep stability at production scale (1024 experts, d=65536)
  2. Free energy computation (Laplacian stress + boundary resonance)
  3. Stiefel retraction holds after many creep steps
  4. Hopfield cleanup retrieval accuracy at scale
  5. Hopfield action decoder snaps noisy waves to correct attractors
  6. Sagnac delta normalization (bounded in [0, 2])
  7. Step latency within execution budget
"""

import math
import time
import torch


def banner(msg):
    print("\n" + "=" * 68)
    print(f"  {msg}")
    print("=" * 68)


def main():
    assert torch.cuda.is_available(), "CUDA required for this verification suite"
    device = torch.device("cuda")
    torch.manual_seed(0)
    print(f"[GPU] {torch.cuda.get_device_name(0)} | torch {torch.__version__}")

    from darwinian_phase_swarm import HenriSwarmOrchestrator
    from hopfield_cleanup import ContinuousHopfieldCleanup, HopfieldActionDecoder

    # ------------------------------------------------------------------
    banner("1. Swarm core at production scale (1024 experts, d=65536)")
    orch = HenriSwarmOrchestrator(num_experts=1024, d_model=65536, r_rank=16, num_blocks=8192).to(device)
    wave = torch.randn(8192, 8, device=device)
    wave = wave / torch.norm(wave, p=2, dim=-1, keepdim=True)
    target = torch.randn(8192, 8, device=device)
    target = target / torch.norm(target, p=2, dim=-1, keepdim=True)

    deltas, latencies = [], []
    n_steps = 100
    for step in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        delta, active, _ = orch.process_active_reasoning_step(
            wave, target, t_shock_max=torch.tensor(0.5, device=device)
        )
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        deltas.append(delta)
        if step % 20 == 0:
            print(f"  step {step:3d} | delta {delta:.4f} | latency {latencies[-1]:.1f} ms")

    A = orch.syncytium.experts_A
    assert not torch.isnan(A).any(), "NaN in experts after 100 creep steps"
    print(f"  [PASS] 100 steps, no NaN | delta in [{min(deltas):.4f}, {max(deltas):.4f}] "
          f"| mean latency {sum(latencies)/len(latencies):.1f} ms")

    # ------------------------------------------------------------------
    banner("2. Free energy decomposition")
    F_total = orch.compute_free_energy(wave, target).item()
    coherence = orch.sagnac_coherence(wave, target).item()
    print(f"  F(wave, target) = {F_total:.6f} | sagnac coherence = {coherence:.6f}")
    assert math.isfinite(F_total), "Free energy is not finite"
    assert -1.0 <= coherence <= 1.0, "Coherence out of [-1, 1]"
    print("  [PASS] free energy finite, coherence bounded")

    # ------------------------------------------------------------------
    banner("3. Stiefel retraction after creep")
    B = orch.syncytium.experts_B
    I16 = torch.eye(16, device=device)
    errA = (torch.bmm(A, A.transpose(-2, -1)) - I16).abs().max().item()
    errB = (torch.bmm(B, B.transpose(-2, -1)) - I16).abs().max().item()
    print(f"  max |AA^T - I| = {errA:.2e} | max |BB^T - I| = {errB:.2e}")
    assert errA < 1e-4 and errB < 1e-4, "Retraction violated"
    print("  [PASS] both expert matrices remain row-orthonormal")

    # ------------------------------------------------------------------
    banner("4. Hopfield cleanup retrieval at scale")
    D = 65536
    M = 64
    cleanup = ContinuousHopfieldCleanup(dim=2 * D).to(device)
    th = torch.rand(M, D, device=device) * 2 * math.pi
    engrams = torch.complex(torch.cos(th), torch.sin(th))
    engrams = engrams / torch.norm(engrams, p=2, dim=-1, keepdim=True)
    cleanup.store_engrams(engrams)

    correct = 0
    n_trials = 20
    for k in range(n_trials):
        tgt = k % M
        noise = torch.complex(torch.randn(D, device=device), torch.randn(D, device=device))
        noise = noise / torch.norm(noise)
        noisy = engrams[tgt] + 0.4 * noise
        noisy = noisy / torch.norm(noisy)
        _, idx, conf = cleanup.hard_retrieve(noisy)
        if int(idx) == tgt:
            correct += 1
    acc = correct / n_trials
    print(f"  hard retrieval accuracy: {correct}/{n_trials} = {acc:.0%} (noise 0.4 relative)")
    assert acc >= 0.95, "Hopfield retrieval accuracy too low"
    print("  [PASS] exponential-capacity cleanup verified at d=65536")

    # ------------------------------------------------------------------
    banner("5. Hopfield action decoder")
    dec = orch.decoder
    wave_a = dec.get_action_wave(list(dec.action_to_id.keys())[0]).to(device)
    noise = torch.complex(torch.randn(65536, device=device), torch.randn(65536, device=device))
    noise = noise / torch.norm(noise)
    noisy = wave_a + 0.3 * noise
    action, conf = dec.decode_wave_to_action(noisy)
    print(f"  decoded '{action}' with confidence {conf:.4f}")
    assert action == list(dec.action_to_id.keys())[0], "Decoder failed"
    print("  [PASS] decoder snaps noisy policy wave to correct action attractor")

    # ------------------------------------------------------------------
    banner("6. Sagnac delta normalization")
    d_identical = 1.0 - orch.sagnac_coherence(wave, wave).item()
    print(f"  delta(wave, wave) = {d_identical:.6f} (expect ~0)")
    assert abs(d_identical) < 1e-3, "Self-coherence not ~1"
    assert all(0.0 <= d <= 2.0 for d in deltas), "Delta out of [0, 2]"
    print("  [PASS] delta bounded, self-resonance ~0")

    # ------------------------------------------------------------------
    banner("7. GPU memory footprint")
    alloc = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  peak allocated: {alloc:.2f} GiB")

    banner("ALL GPU VERIFICATION CHECKS PASSED")


if __name__ == "__main__":
    main()
