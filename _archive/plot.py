import json
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

telemetry_file = "HENRI V2/telemetry_logs/vast_ai_sync_20260719_204835.jsonl"
output_image = "HENRI_ARC_AGI_3.png"

# Read JSONL
phase_deltas = []
status = []
real_phases = []
imag_phases = []

with open(telemetry_file, 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            phase_deltas.append(data.get("phase_delta", 0))
            status.append(data.get("sagnac_clearance", False))
            # Just take the first one or a sample for the histogram
            if len(real_phases) < 5:
                real_phases.append(np.array(data.get("real_phases", [])))
                imag_phases.append(np.array(data.get("imag_phases", [])))
        except json.JSONDecodeError:
            continue

# Reversing because order might be DESC from SQL
phase_deltas = phase_deltas[::-1]
status = status[::-1]
real_phases = real_phases[::-1]
imag_phases = imag_phases[::-1]

plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 8))
fig.suptitle("HENRI O-VSA Telemetry: ARC-AGI 3 Ingestion", fontsize=18, color='cyan')

# 1. Sagnac Phase Delta over steps
ax1 = plt.subplot(2, 2, (1, 2))
steps = np.arange(len(phase_deltas))
ax1.plot(steps, phase_deltas, marker='o', color='magenta', linewidth=2, label="Phase Delta (L2 Norm)")
ax1.axhline(y=0.1, color='green', linestyle='--', label="Resonance Threshold (0.1)")
ax1.set_title("Sagnac Interference over Causal Steps")
ax1.set_xlabel("Time Step (Test-Time Yielding)")
ax1.set_ylabel("Phase Delta")
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Final Wave State Histogram (Real vs Imaginary)
ax2 = plt.subplot(2, 2, 3)
if len(real_phases) > 0 and len(real_phases[-1]) > 0:
    ax2.hist(real_phases[-1], bins=50, color='cyan', alpha=0.6, label='Real')
    ax2.hist(imag_phases[-1], bins=50, color='magenta', alpha=0.6, label='Imaginary')
    ax2.set_title("Wave-JEPA $S^{4095}$ State Distribution (Latest)")
    ax2.legend()
else:
    ax2.text(0.5, 0.5, "No Phase Data", ha='center')

# 3. Complex Phase Scatter (First 500 dims for visibility)
ax3 = plt.subplot(2, 2, 4)
if len(real_phases) > 0 and len(real_phases[-1]) > 0:
    limit = min(500, len(real_phases[-1]))
    ax3.scatter(real_phases[-1][:limit], imag_phases[-1][:limit], alpha=0.5, color='white', s=5)
    ax3.set_title("Complex Superposition (Subset)")
    ax3.set_xlabel("Real")
    ax3.set_ylabel("Imaginary")
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, color='green', fill=False, linestyle='--', alpha=0.5)
    ax3.add_patch(circle)
    ax3.set_aspect('equal')
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)

plt.tight_layout()
plt.savefig(output_image, dpi=150, bbox_inches='tight')
print(f"Generated {output_image}")
