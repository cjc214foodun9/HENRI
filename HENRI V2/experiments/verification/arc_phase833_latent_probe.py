"""Phase 8.33b — Latent-space isolation probe (Roadmap HENRI-ROADMAP-2026-VLA-UNIVERSAL, §3.2 Option 1).

Isolates E_transition from E_egress on the AUTHORIZED trajectory bank
(`trajectories_production_run_1787164827.npz`, 90 records, sealed):

  metric: rho_latent = mean held-out complex cosine between
          NonLinearWaveJEPA latent prediction (L=2048) and the latent
          projection of the true next wave (compress_wave(next_wave)).

  - rho_latent >= 0.80  -> verdict TRIGGER_OPTION2 (roadmap §3.2 step 3:
        immediately trigger Option (2): refactor henri_egress.py and
        hopfield_cleanup.py — 2-layer compressed projection head with
        in-situ error feedback / test-time SGLD).
  - rho_latent <  0.80  -> verdict NO_TRIGGER (transition itself fails in
        latent space; the bottleneck is NOT the lift).

Training and split mirror the Phase 8.33 kill experiment EXACTLY
(held_out_frac=0.2, seed=20260819, 400 Adam epochs, K=32 options,
compressed L=2048, sagnac_lambda=0.15) so the two verdicts are comparable
on the same 72/18 partition. Option id = onehot argmax (action index).

Output: compact JSON verdict on stdout (last line).
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from henri_trajectory_bank import TrajectoryBank
from henri_nonlinear_wavejepa import NonLinearWaveJEPA

DEFAULT_SEED = 20260819
HELDOUT_FRAC = 0.2
NUM_BLOCKS = 8192
BLOCK_DIM = 8
D = NUM_BLOCKS * BLOCK_DIM  # 65,536
RHO_TRIGGER = 0.80


def split_indices(n: int, frac: float, seed: int):
    """Mirror the calibrator/kill split: seeded randperm, held-out = first n_test."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_test = max(1, int(round(n * frac)))
    return perm[n_test:].tolist(), perm[:n_test].tolist()


def run_probe(bank_npz: str, manifest_path: str, wave_dim: int,
              epochs: int, seed: int, device: str) -> dict:
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()

    data = TrajectoryBank.load(bank_npz, manifest_path, verify_digest=True)
    psi = data["psi"]          # [M, D] f32
    onehot = data["actions_onehot"]  # [M, 6] f32
    nxt = data["next_wave"]
    if nxt is None or psi.shape[0] != nxt.shape[0]:
        raise RuntimeError("bank missing next_wave records")
    if wave_dim != psi.shape[1]:
        # Smoke-only path: slice the leading wave_dim columns. Production
        # default wave_dim == D (65,536) leaves this branch inactive.
        print(f"  wave_dim {wave_dim} != bank dim {psi.shape[1]} -> slicing leading columns "
              f"(smoke path only)", flush=True)
        psi = np.ascontiguousarray(psi[:, :wave_dim])
        nxt = np.ascontiguousarray(nxt[:, :wave_dim])
    M = psi.shape[0]
    action_idx = torch.from_numpy(onehot).argmax(dim=-1).long()
    run_id = str(data["manifest"].get("run_id", "unknown"))

    train_idx, held_idx = split_indices(M, HELDOUT_FRAC, seed)
    n_train, n_held = len(train_idx), len(held_idx)

    # ---- Train the same NonLinearWaveJEPA as the kill experiment ----
    torch.manual_seed(seed + 1)
    jepa = NonLinearWaveJEPA(
        full_dim=wave_dim, compressed_dim=2048,
        num_options=32, opt_dim=512, sagnac_lambda=0.15,
        device=dev.type,
    ).to(dev)
    optm = torch.optim.Adam([p for p in jepa.parameters() if p.requires_grad], lr=1e-3)

    psi_tr = torch.from_numpy(psi[train_idx]).to(dev)
    nxt_tr = torch.from_numpy(nxt[train_idx]).to(dev)
    opt_tr = action_idx[train_idx].to(dev)

    train_loss = float("nan")
    for ep in range(epochs):
        optm.zero_grad()
        out = jepa(psi_tr, opt_tr, nxt_tr)
        loss = out["loss"]
        if not torch.isfinite(loss):
            train_loss = float("nan")
            break
        loss.backward()
        optm.step()
        train_loss = float(out["jepa_loss"])
        if ep % 100 == 0:
            print(f"  jepa epoch {ep}: loss={out['loss']:.4f} "
                  f"jepa={out['jepa_loss']:.4f} sagnac={out['sagnac_stress']:.4f}", flush=True)

    # ---- Latent-space held-out evaluation (the isolation metric) ----
    def latent_cosine(psi_t: torch.Tensor, opt: torch.Tensor, tgt: torch.Tensor) -> float:
        with torch.no_grad():
            pred = jepa.predict_next_state(psi_t, opt)      # (B, L, 2)
            target = jepa.compress_wave(tgt)                # (B, L, 2)
            r_p, i_p = pred[..., 0], pred[..., 1]
            r_t, i_t = target[..., 0], target[..., 1]
            cos = torch.sum(r_p * r_t + i_p * i_t, dim=-1) / jepa.compressed_dim
            return float(cos.mean().item())

    with torch.no_grad():
        cos_tr = latent_cosine(psi_tr, opt_tr, nxt_tr)
        psi_ho = torch.from_numpy(psi[held_idx]).to(dev)
        nxt_ho = torch.from_numpy(nxt[held_idx]).to(dev)
        opt_ho = action_idx[held_idx].to(dev)
        cos_ho = latent_cosine(psi_ho, opt_ho, nxt_ho)

        # Ambient reference (1 - loss form, same as kill experiment).
        nb = wave_dim // BLOCK_DIM  # 8192 at production D=65,536; matches NUM_BLOCKS
        y = nxt_ho / (torch.norm(nxt_ho, p=2, dim=-1, keepdim=True) + 1e-9)
        pred_full = jepa.predict_full_wave(psi_ho, opt_ho,
                                           num_blocks=nb, block_dim=BLOCK_DIM)
        pf = pred_full.view(pred_full.size(0), -1)
        pf = pf / (torch.norm(pf, p=2, dim=-1, keepdim=True) + 1e-9)
        ambient_cos = float((y * pf).sum(dim=-1).mean().item())

    rho_latent = float(cos_ho)
    finite = math.isfinite(rho_latent) and math.isfinite(train_loss)
    if not finite:
        verdict = "BLOCKED_INFRA"
        reason = "NaN/divergence in latent evaluation"
    elif rho_latent >= RHO_TRIGGER:
        verdict = "TRIGGER_OPTION2"
        reason = f"rho_latent {rho_latent:.4f} >= {RHO_TRIGGER}: transition learns in latent; egress is the bottleneck"
    else:
        verdict = "NO_TRIGGER"
        reason = f"rho_latent {rho_latent:.4f} < {RHO_TRIGGER}: transition does NOT separate in latent space"

    result = {
        "schema": "henri.phase833.latent-probe.v1",
        "verdict": verdict,
        "reason": reason,
        "run_id": run_id,
        "bank_npz_sha256": data["manifest"].get("npz_sha256", ""),
        "records": {"total": M, "train": n_train, "heldout": n_held},
        "split": {"held_out_frac": HELDOUT_FRAC, "seed": seed},
        "metrics": {
            "rho_latent_heldout": rho_latent,
            "cos_latent_train": float(cos_tr),
            "cos_ambient_heldout": ambient_cos,
            "train_jepa_loss": train_loss,
        },
        "trigger": {"threshold": RHO_TRIGGER,
                    "targets": ["henri_egress.py", "hopfield_cleanup.py"] if verdict == "TRIGGER_OPTION2" else []},
        "device": dev.type,
        "epochs": epochs,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 8.33b latent isolation probe")
    p.add_argument("--bank", required=True, help="bank npz path")
    p.add_argument("--manifest", default="", help="bank manifest json path")
    p.add_argument("--wave-dim", type=int, default=D)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="")
    args = p.parse_args(argv)

    print(f"== Phase 8.33b latent isolation probe ==\nbank: {args.bank}\nseed: {args.seed} "
          f"epochs: {args.epochs}\ndevice: {args.device or 'auto'}", flush=True)
    result = run_probe(args.bank, args.manifest or None, args.wave_dim,
                       args.epochs, args.seed, args.device)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
