# Phase 8.33b — Latent Isolation Probe & Egress Refactor (Roadmap §3.2)

Source: `Project_HENRI__Architectural_Roadmap_to_Universal_Holographic_Vision-Language-Action_Cognition.md`
(HENRI-ROADMAP-2026-VLA-UNIVERSAL, Aletheia; read 2026-08-19, 104 lines).

## Option (1) — Latent-space isolation probe (EXECUTED, verdict TRIGGER_OPTION2)

Probe: `experiments/verification/arc_phase833_latent_probe.py` @ `99716c3`.
Bank: `trajectories_production_run_1787164827.npz` (90 records, sealed,
sha `a5d8f1b3…`; 72/18 split, frac 0.2, seed 20260819 — mirrors the
calibrator and the 8.33 kill experiment).

Metrics (CUDA, D=65,536, 400 epochs, 6.1 s; log sha `5afc332a…`):

| metric | value | gate |
|---|---|---|
| **rho_latent_heldout** | **0.8846** | >= 0.80 ✓ |
| cos_latent_train | 0.9843 | — |
| cos_ambient_heldout | 0.0915 | (reference) |
| train_jepa_loss | 0.0158 | < 0.5 ✓ |

Verdict: **TRIGGER_OPTION2** — the non-linear transition separates
true-next-state structure in latent space (ρ 0.8846), while the same
prediction lifted to ambient collapses (cos 0.09). This cleanly isolates
**E_transition << E_egress**: the transition is NOT the bottleneck; the
continuous→discrete egress interface is. (Contrast 8.32 head MSE 24.22 and
8.33 ambient Sagnac 0.9915: same egress boundary, now proven as the cause.)

## Option (2) — Egress refactor (IN PROGRESS, per roadmap trigger)

Targets: `henri_egress.py`, `hopfield_cleanup.py`.

Mechanism (roadmap §2.1): 2-layer compressed projection head with in-situ
error feedback; environment state deltas (Δs_t+1) and exteroceptive
scorecard deltas (Δν) propagate back through the unbinder head during live
inference via test-time SGLD. VLA Gate 1: I(Ψ_goal; Y) >= 0.85 (falsify:
unbinder logit entropy uniform, I = 0).

Design constraints (carried from this session):
- default-OFF behind a named flag; no change to the default egress path;
- no BPTT through the encoder; SGLD creep on the unbinder head only;
- E_cal-style held-out semantic gate before any activation;
- authorized-bank-only evaluation (never evaluation caches).
