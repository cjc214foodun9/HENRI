"""
Phase 8.36 C++/CUDA VRAM Harness Verification Suite
====================================================
Verifies that CUDAZoneCEnvironmentHarness achieves step latency <= 2.0 ms
on CUDA and maintains exact frame hash determinism in GPU memory.

Imports use the repository flat-import convention (PYTHONPATH="HENRI V2"),
not the inbox source's `HENRI_V2.zone_c_cuda_harness` dotted path.

Correctness probes (CPU and CUDA):
    - Per-action deterministic transforms on a known 4x4 grid.
    - NO-OP (0) preserves state and hash.
    - Hash determinism: identical state -> identical hash.
    - Hash sensitivity: rotated state -> different hash.

Latency gate (CUDA only): average of 45 measured steps <= 2.0 ms after
5 warmup steps, batch 16, 30x30 grids, actions drawn per step.
"""

import time

import torch
import pytest

from zone_c_cuda_harness import CUDAZoneCEnvironmentHarness


def _known_grid():
    # 4x4 grid with distinct values, includes a zero cell.
    return torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 1, 2],
            [3, 4, 5, 6],
        ],
        dtype=torch.int32,
    )


def test_phase836_action_correctness_cpu():
    """Deterministic per-action transform probes on CPU.

    Canvas is sized 4x4 so the loaded grid fills the full canvas; rotation
    semantics then match the isolated-grid expectation (the harness rotates
    the full canvas, as the production ARC grid semantics require).
    """
    device = "cpu"
    harness = CUDAZoneCEnvironmentHarness(max_batch_size=4, max_h=4, max_w=4).to(device)
    g = _known_grid()
    harness.load_environments(g.unsqueeze(0))

    # ROTATE_90: clockwise rotation.
    harness.step_cuda(torch.tensor([1], device=device))
    got = harness.grid_states[0, :4, :4].tolist()
    expected = torch.rot90(g, k=3, dims=[0, 1]).tolist()
    assert got == expected, f"ROTATE_90 mismatch: {got}"

    harness.load_environments(g.unsqueeze(0))
    # FLIP_H: horizontal axis reflection.
    harness.step_cuda(torch.tensor([2], device=device))
    got = harness.grid_states[0, :4, :4].tolist()
    assert got == torch.flip(g, dims=[1]).tolist(), f"FLIP_H mismatch: {got}"

    harness.load_environments(g.unsqueeze(0))
    # SHIFT_RIGHT: circular horizontal shift.
    harness.step_cuda(torch.tensor([3], device=device))
    got = harness.grid_states[0, :4, :4].tolist()
    assert got == torch.roll(g, shifts=1, dims=1).tolist(), f"SHIFT_RIGHT mismatch: {got}"

    harness.load_environments(g.unsqueeze(0))
    # SHIFT_DOWN: circular vertical shift.
    harness.step_cuda(torch.tensor([4], device=device))
    got = harness.grid_states[0, :4, :4].tolist()
    assert got == torch.roll(g, shifts=1, dims=0).tolist(), f"SHIFT_DOWN mismatch: {got}"

    harness.load_environments(g.unsqueeze(0))
    # INVERT_COLOR: modular color rotation on nonzero cells only.
    harness.step_cuda(torch.tensor([5], device=device))
    got = harness.grid_states[0, :4, :4].tolist()
    assert got == torch.where(g > 0, (g % 9) + 1, g).tolist(), f"INVERT_COLOR mismatch: {got}"


def test_phase836_noop_and_hash_determinism_cpu():
    """NO-OP preserves state/hash; identical state reproduces identical hash."""
    device = "cpu"
    harness = CUDAZoneCEnvironmentHarness(max_batch_size=4, max_h=30, max_w=30).to(device)
    g = _known_grid()
    harness.load_environments(g.unsqueeze(0))
    h0 = harness.frame_hashes[0].item()

    # NO-OP
    harness.step_cuda(torch.tensor([0], device=device))
    assert harness.frame_hashes[0].item() == h0, "NO-OP changed the frame hash"
    assert harness.grid_states[0, :4, :4].tolist() == g.tolist(), "NO-OP changed the grid"

    # Determinism: re-hash the same state.
    harness._compute_frame_hashes_vram(1)
    assert harness.frame_hashes[0].item() == h0, "Frame hash not deterministic"

    # Sensitivity: a different grid must hash differently.
    harness.load_environments(torch.rot90(g, k=3, dims=[0, 1]).unsqueeze(0))
    assert harness.frame_hashes[0].item() != h0, "Rotated grid produced identical hash"


def test_phase836_cuda_harness_latency_and_correctness():
    """Inbox-specified gate: batch 16, 30x30, 50 steps; <= 2.0 ms avg on CUDA."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Instantiate Harness
    harness = CUDAZoneCEnvironmentHarness(max_batch_size=32, max_h=30, max_w=30).to(device)

    # 2. Create Random Grid Batch (B=16, H=30, W=30)
    grid_batch = torch.randint(0, 10, (16, 30, 30), device=device, dtype=torch.int32)
    harness.load_environments(grid_batch)

    initial_hashes = harness.frame_hashes[:16].clone()

    # 3. Execute 50 Parallel CUDA Steps (fresh action draw per step)
    warmup_steps = 5
    measured_steps = 45

    total_ms = 0.0
    wall_total_ms = 0.0
    for step in range(warmup_steps + measured_steps):
        action_batch = torch.randint(1, 6, (16,), device=device)
        wall_t0 = time.perf_counter()
        out = harness.step_cuda(action_batch)
        wall_total_ms += (time.perf_counter() - wall_t0) * 1000.0
        if step >= warmup_steps:
            total_ms += out["step_latency_ms"]

    avg_latency_ms = total_ms / measured_steps
    avg_wall_ms = wall_total_ms / measured_steps
    print(f"\n[*] Measured Average Step Latency: {avg_latency_ms:.4f} ms")
    print(f"[*] Measured Average Wall-Clock Step: {avg_wall_ms:.4f} ms")

    # Assertions
    # 1. Frame hashes changed after action steps
    assert not torch.equal(initial_hashes, harness.frame_hashes[:16]), "FALSIFIED: Frame hashes failed to update"

    # 2. Step latency gate (<= 2.0 ms on CUDA)
    if device == "cuda":
        assert avg_latency_ms <= 2.0, f"FALSIFIED: Step latency {avg_latency_ms:.2f} ms exceeds 2.0 ms limit"
        print(f"[✓] Phase 8.36 Latency Gate MET: {avg_latency_ms:.4f} ms <= 2.0 ms")
