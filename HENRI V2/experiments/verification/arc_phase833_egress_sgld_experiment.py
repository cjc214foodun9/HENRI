"""Phase 8.33 Option (2) — Egress SGLD kill experiment (Roadmap VLA Gate 1).

Tests whether test-time SGLD creep on a 2-layer compressed egress head
restores mutual information between the GOAL wave and the action label,
measured on the AUTHORIZED trajectory bank
(`trajectories_production_run_1787164827.npz`, 90 records, sealed).

The roadmap gate is I(Psi_goal; Y): the egress unbinder reads the GOAL
wave (the predicted/next state), not the observation. Two prior failures
are explained by this distinction: the linear head on psi_t (MSE 24.22)
and the SGLD head on psi_t (acc 0.056) both read the WRONG input — a_t is
identifiable through its effect Psi_{t+1}, not from Psi_t alone.

Arms (same split: held_out_frac=0.2, seed=20260819; 72 train / 18 held):
  A) LINEAR_obs   : ridge psi_t      -> action   (historical reference)
  B) SGLD_obs     : SGLD head psi_t  -> action   (same input as A, neural)
  C) LINEAR_goal  : ridge next_wave  -> action   (linear goal-wave lift)
  D) SGLD_goal    : SGLD head next_wave -> action (roadmap §2.1 target)
  E) CLOSED_LOOP  : SGLD head (adapted on next_wave) evaluated on the
                    JEPA-PREDICTED next wave from psi_t (full production
                    path: UWE -> JEPA predict -> egress head -> action)

Metrics (held-out 18 records):
  acc, top1_rank, I_norm = I(Yhat; Y)/H(Y) from the confusion matrix,
  entropy (nats).

VLA Gate 1 verdict (roadmap §3.2): the egress gate is scored on Arm D.
  PASS  : I_norm >= 0.85 AND acc >= 0.80 AND entropy < 0.5 * ln(6)
  FAIL  : otherwise (uniform logits -> I ~ 0 is the falsification)
  BLOCKED_INFRA: NaN / digest mismatch / GPU failure (no verdict)

Arm E is informational: it quantifies how much JEPA prediction error
(known small: rho_latent 0.8846) survives the closed loop.

Output: compact JSON verdict on stdout (last line).
"""

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as Fn

from henri_trajectory_bank import TrajectoryBank
from henri_egress import CompressedProjectionHead, sgld_adapt_head
from henri_nonlinear_wavejepa import NonLinearWaveJEPA

DEFAULT_SEED = 20260819
HELDOUT_FRAC = 0.2
NUM_BLOCKS = 8192
BLOCK_DIM = 8
D = NUM_BLOCKS * BLOCK_DIM  # 65,536
HIDDEN = 1024
VOCAB = 6
RIDGE_GAMMA = 1e-3
JEPA_EPOCHS = 400


def split_indices(n: int, frac: float, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_test = max(1, int(round(n * frac)))
    return perm[n_test:].tolist(), perm[:n_test].tolist()


def _norm_mi(cm: np.ndarray) -> float:
    """Normalized mutual information I(Yhat;Y)/H(Y) from a confusion matrix."""
    cm = cm.astype(np.float64)
    cm = cm / (cm.sum() + 1e-12)
    py = cm.sum(axis=0)
    pyhat = cm.sum(axis=1)
    hy = -float((py * np.log(py + 1e-12)).sum())
    mi = 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            p = cm[i, j]
            if p > 0:
                mi += p * math.log(p / (pyhat[i] * py[j] + 1e-12))
    return float(mi / (hy + 1e-12))


def _eval_arm(logits: torch.Tensor, y: torch.Tensor) -> dict:
    pred = logits.argmax(dim=-1)
    acc = float((pred == y).float().mean().item())
    rank = float((logits.argsort(dim=-1, descending=True) == y.unsqueeze(1))
                 .nonzero(as_tuple=False)[:, 1].float().mean().item()) + 1.0
    cm = np.zeros((VOCAB, VOCAB), dtype=int)
    for a, b in zip(y.cpu().tolist(), pred.cpu().tolist()):
        cm[b, a] += 1
    return {"acc": acc, "top1_rank": rank, "I_norm": _norm_mi(cm)}


def run_experiment(bank_npz: str, manifest_path: str, device: str,
                   sgld_steps: int, seed: int) -> dict:
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()

    data = TrajectoryBank.load(bank_npz, manifest_path, verify_digest=True)
    psi = data["psi"]                    # [M, D]
    onehot = data["actions_onehot"]      # [M, 6]
    nxt = data["next_wave"]
    if nxt is None or psi.shape[0] != nxt.shape[0]:
        raise RuntimeError("bank missing next_wave records")
    if psi.shape[1] != D:
        raise RuntimeError(f"bank wave dim {psi.shape[1]} != {D} (production only)")
    M = psi.shape[0]
    action_idx = torch.from_numpy(onehot).argmax(dim=-1).long()
    run_id = str(data["manifest"].get("run_id", "unknown"))

    train_idx, held_idx = split_indices(M, HELDOUT_FRAC, seed)
    n_train, n_held = len(train_idx), len(held_idx)

    psi_tr = torch.from_numpy(psi[train_idx]).to(dev)
    nxt_tr = torch.from_numpy(nxt[train_idx]).to(dev)
    y_tr = action_idx[train_idx].to(dev)
    psi_ho = torch.from_numpy(psi[held_idx]).to(dev)
    nxt_ho = torch.from_numpy(nxt[held_idx]).to(dev)
    y_ho = action_idx[held_idx].to(dev)
    onehot_tr = Fn.one_hot(y_tr, VOCAB).float()

    def ridge_logits(waves_tr: torch.Tensor, waves_ho: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(waves_tr, full_matrices=False)
            coef = (S / (S * S + RIDGE_GAMMA)).unsqueeze(1) * (U.t() @ onehot_tr)
            W_lin = Vt.t() @ coef  # [D, VOCAB]
            return waves_ho @ W_lin

    # ---- Arm A: linear ridge on psi_t (historical reference) ----
    logits_a = ridge_logits(psi_tr, psi_ho)
    arm_a = _eval_arm(logits_a, y_ho)

    # ---- Arm C: linear ridge on next_wave (goal wave, linear lift) ----
    logits_c = ridge_logits(nxt_tr, nxt_ho)
    arm_c = _eval_arm(logits_c, y_ho)

    # ---- Arm B: SGLD head on psi_t (same input as A, neural) ----
    torch.manual_seed(seed + 1)
    head_b = CompressedProjectionHead(d_model=D, hidden_dim=HIDDEN,
                                      vocab_size=VOCAB, sagnac_lambda=0.25).to(dev)
    res_b = sgld_adapt_head(head_b, psi_tr, y_tr, lr=1e-4, steps=sgld_steps,
                            log_every=max(1, sgld_steps // 5), seed=seed + 2)
    with torch.no_grad():
        arm_b = _eval_arm(head_b(psi_ho), y_ho)
        arm_b["entropy_nats"] = float(head_b.logit_entropy(head_b(psi_ho)).mean().item())
    arm_b["train_final_loss"] = res_b["final_loss"]
    arm_b["yielded"] = res_b["yielded"]

    # ---- Arm D: SGLD head on next_wave (roadmap §2.1 target) ----
    torch.manual_seed(seed + 3)
    head_d = CompressedProjectionHead(d_model=D, hidden_dim=HIDDEN,
                                      vocab_size=VOCAB, sagnac_lambda=0.25).to(dev)
    res_d = sgld_adapt_head(head_d, nxt_tr, y_tr, lr=1e-4, steps=sgld_steps,
                            log_every=max(1, sgld_steps // 5), seed=seed + 4)
    with torch.no_grad():
        arm_d = _eval_arm(head_d(nxt_ho), y_ho)
        arm_d["entropy_nats"] = float(head_d.logit_entropy(head_d(nxt_ho)).mean().item())
    arm_d["train_final_loss"] = res_d["final_loss"]
    arm_d["yielded"] = res_d["yielded"]

    # ---- Arm E: closed loop — JEPA-predicted next wave -> head_d ----
    torch.manual_seed(seed + 5)
    jepa = NonLinearWaveJEPA(
        full_dim=D, compressed_dim=2048, num_options=32, opt_dim=512,
        sagnac_lambda=0.15, device=dev.type,
    ).to(dev)
    optm = torch.optim.Adam([p for p in jepa.parameters() if p.requires_grad], lr=1e-3)
    opt_tr = action_idx[train_idx].to(dev)
    for ep in range(JEPA_EPOCHS):
        optm.zero_grad()
        out = jepa(psi_tr, opt_tr, nxt_tr)
        loss = out["loss"]
        if not torch.isfinite(loss):
            break
        loss.backward()
        optm.step()
    with torch.no_grad():
        pred_full = jepa.predict_full_wave(
            psi_ho, action_idx[held_idx].to(dev),
            num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM).view(n_held, D)
        arm_e = _eval_arm(head_d(pred_full), y_ho)
        arm_e["pred_cos_goal"] = float(
            Fn.normalize(pred_full, p=2, dim=-1) *
            Fn.normalize(nxt_ho, p=2, dim=-1)).sum(dim=-1).mean().item()

    # ---- VLA Gate 1 verdict on Arm D (the egress gate) ----
    uniform_half = 0.5 * math.log(VOCAB)
    i_d, acc_d, ent_d = arm_d["I_norm"], arm_d["acc"], arm_d["entropy_nats"]
    if not (math.isfinite(i_d) and math.isfinite(acc_d) and math.isfinite(ent_d)):
        verdict = "BLOCKED_INFRA"
        reason = "NaN in metrics"
    elif i_d >= 0.85 and acc_d >= 0.80 and ent_d < uniform_half:
        verdict = "PASS"
        reason = (f"I_norm {i_d:.3f} >= 0.85, acc {acc_d:.3f} >= 0.80, "
                  f"entropy {ent_d:.3f} < {uniform_half:.3f}")
    else:
        verdict = "FAIL"
        reason = (f"I_norm {i_d:.3f} < 0.85 (or acc {acc_d:.3f} < 0.80 / "
                  f"entropy {ent_d:.3f} >= {uniform_half:.3f})")

    result = {
        "schema": "henri.phase833.egress-sgld-experiment.v1",
        "verdict": verdict,
        "reason": reason,
        "run_id": run_id,
        "bank_npz_sha256": data["manifest"].get("npz_sha256", ""),
        "records": {"total": M, "train": n_train, "heldout": n_held},
        "split": {"held_out_frac": HELDOUT_FRAC, "seed": seed},
        "gate": "VLA Gate 1: I(Psi_goal; Y) >= 0.85 (scored on Arm D: SGLD head on goal wave)",
        "metrics": {
            "A_linear_obs": arm_a,
            "B_sgld_obs": arm_b,
            "C_linear_goal": arm_c,
            "D_sgld_goal": arm_d,
            "E_closed_loop": arm_e,
        },
        "device": dev.type,
        "sgld_steps": sgld_steps,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 8.33 Option (2) egress SGLD kill experiment")
    p.add_argument("--bank", required=True, help="bank npz path")
    p.add_argument("--manifest", default="", help="bank manifest json path")
    p.add_argument("--steps", type=int, default=500, help="SGLD adaptation steps")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="")
    args = p.parse_args(argv)

    print(f"== Phase 8.33 Option (2) egress SGLD experiment ==\nbank: {args.bank}\n"
          f"seed: {args.seed} steps: {args.steps}\ndevice: {args.device or 'auto'}", flush=True)
    result = run_experiment(args.bank, args.manifest or None, args.device,
                            args.steps, args.seed)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
