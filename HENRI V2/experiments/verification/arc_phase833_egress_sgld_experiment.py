"""Phase 8.33 Option (2) — Egress SGLD kill experiment (Roadmap VLA Gate 1).

Tests whether test-time SGLD creep on a 2-layer compressed egress head
restores mutual information between the goal wave and the action label,
measured on the AUTHORIZED trajectory bank
(`trajectories_production_run_1787164827.npz`, 90 records, sealed).

Setup (mirrors the calibrator and the 8.33 kill experiment EXACTLY):
  split  : held_out_frac=0.2, seed=20260819, deterministic randperm
            72 train / 18 held-out records
  waves  : psi[t]  (the observation wave at record t)
  targets: action_idx[t] = argmax(actions_onehot[t])   (0..5)

Arms (same split, same bank):
  A) LINEAR reference: closed-form ridge (SVD-form, gamma=1e-3) wave->onehot
     fit on the 72 train records (the Phase 8.32 head family; known-fail
     reference at MSE 24.22 on the ambient metric).
  B) SGLD egress head: CompressedProjectionHead(D=65536, hidden=1024,
     vocab=6) adapted by `sgld_adapt_head` on the 72 train records
     (CE + 0.25*Sagnac stress, T(t)=T0(1+0.05t)^-0.55, Bingham yield,
     unit-normalized Langevin noise).

Metrics (held-out 18 records):
  acc        : argmax(logits) == action
  top1_rank  : mean rank of the true action in the logit order
  I_norm     : normalized mutual information I(Yhat; Y)/H(Y) from the
               held-out confusion matrix  (VLA Gate 1 metric)
  entropy    : mean softmax entropy over held-out (nats)

VLA Gate 1 verdicts (roadmap §3.2 table):
  PASS  : I_norm >= 0.85 AND acc >= 0.80 AND entropy < 0.5 * ln(6)
          -> I(Psi_goal; Y) restored; the adapted head is task-readable.
  FAIL  : I_norm < 0.85 (uniform logits -> I ~ 0 is the falsification).
  BLOCKED_INFRA: NaN / digest mismatch / GPU failure (no verdict).

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

DEFAULT_SEED = 20260819
HELDOUT_FRAC = 0.2
D = 65536
HIDDEN = 1024
VOCAB = 6
RIDGE_GAMMA = 1e-3


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


def run_experiment(bank_npz: str, manifest_path: str, device: str,
                   sgld_steps: int, seed: int) -> dict:
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()

    data = TrajectoryBank.load(bank_npz, manifest_path, verify_digest=True)
    psi = data["psi"]
    onehot = data["actions_onehot"]
    if psi.shape[0] != onehot.shape[0]:
        raise RuntimeError("bank record mismatch")
    M = psi.shape[0]
    action_idx = torch.from_numpy(onehot).argmax(dim=-1).long()
    run_id = str(data["manifest"].get("run_id", "unknown"))
    if psi.shape[1] != D:
        raise RuntimeError(f"bank wave dim {psi.shape[1]} != {D} (production only)")

    train_idx, held_idx = split_indices(M, HELDOUT_FRAC, seed)
    n_train, n_held = len(train_idx), len(held_idx)

    psi_tr = torch.from_numpy(psi[train_idx]).to(dev)
    y_tr = action_idx[train_idx].to(dev)
    psi_ho = torch.from_numpy(psi[held_idx]).to(dev)
    y_ho = action_idx[held_idx].to(dev)

    # ---- Arm A: linear ridge reference (wave -> onehot) ----
    with torch.no_grad():
        Xt = psi_tr.clone()  # [n, D]
        onehot_tr = Fn.one_hot(y_tr, VOCAB).float()
        U, S, Vt = torch.linalg.svd(Xt, full_matrices=False)
        # ridge weights: V diag(s/(s^2+gamma)) U^T onehot  (no [D,D] allocation)
        coef = (S / (S * S + RIDGE_GAMMA)).unsqueeze(1) * (U.t() @ onehot_tr)
        W_lin = Vt.t() @ coef  # [D, VOCAB]
        logits_ho = psi_ho @ W_lin
        acc_lin = float((logits_ho.argmax(dim=-1) == y_ho).float().mean().item())
        rank_lin = float((logits_ho.argsort(dim=-1, descending=True) == y_ho.unsqueeze(1))
                         .nonzero(as_tuple=False)[:, 1].float().mean().item()) + 1.0
        cm_lin = np.zeros((VOCAB, VOCAB), dtype=int)
        for a, b in zip(y_ho.cpu().tolist(), logits_ho.argmax(dim=-1).cpu().tolist()):
            cm_lin[b, a] += 1
        i_lin = _norm_mi(cm_lin)

    # ---- Arm B: SGLD-adapted compressed egress head ----
    torch.manual_seed(seed + 1)
    head = CompressedProjectionHead(d_model=D, hidden_dim=HIDDEN,
                                    vocab_size=VOCAB, sagnac_lambda=0.25).to(dev)
    res_sgld = sgld_adapt_head(head, psi_tr, y_tr, lr=1e-4, steps=sgld_steps,
                               t0=0.1, dt=1.0, yield_stress=0.05,
                               log_every=max(1, sgld_steps // 5), seed=seed + 2)
    with torch.no_grad():
        logits_ho = head(psi_ho)
        acc_sgld = float((logits_ho.argmax(dim=-1) == y_ho).float().mean().item())
        rank_sgld = float((logits_ho.argsort(dim=-1, descending=True) == y_ho.unsqueeze(1))
                          .nonzero(as_tuple=False)[:, 1].float().mean().item()) + 1.0
        ent_sgld = float(head.logit_entropy(logits_ho).mean().item())
        cm_sgld = np.zeros((VOCAB, VOCAB), dtype=int)
        for a, b in zip(y_ho.cpu().tolist(), logits_ho.argmax(dim=-1).cpu().tolist()):
            cm_sgld[b, a] += 1
        i_sgld = _norm_mi(cm_sgld)

    # ---- VLA Gate 1 verdict ----
    uniform_half = 0.5 * math.log(VOCAB)
    if not (math.isfinite(i_sgld) and math.isfinite(acc_sgld) and math.isfinite(ent_sgld)):
        verdict = "BLOCKED_INFRA"
        reason = "NaN in metrics"
    elif i_sgld >= 0.85 and acc_sgld >= 0.80 and ent_sgld < uniform_half:
        verdict = "PASS"
        reason = (f"I_norm {i_sgld:.3f} >= 0.85, acc {acc_sgld:.3f} >= 0.80, "
                  f"entropy {ent_sgld:.3f} < {uniform_half:.3f}")
    else:
        verdict = "FAIL"
        reason = (f"I_norm {i_sgld:.3f} < 0.85 (or acc {acc_sgld:.3f} < 0.80 / "
                  f"entropy {ent_sgld:.3f} >= {uniform_half:.3f})")

    result = {
        "schema": "henri.phase833.egress-sgld-experiment.v1",
        "verdict": verdict,
        "reason": reason,
        "run_id": run_id,
        "bank_npz_sha256": data["manifest"].get("npz_sha256", ""),
        "records": {"total": M, "train": n_train, "heldout": n_held},
        "split": {"held_out_frac": HELDOUT_FRAC, "seed": seed},
        "gate": "VLA Gate 1: I(Psi_goal; Y) >= 0.85",
        "metrics": {
            "linear": {"acc": acc_lin, "top1_rank": rank_lin, "I_norm": i_lin},
            "sgld_head": {"acc": acc_sgld, "top1_rank": rank_sgld,
                          "I_norm": i_sgld, "entropy_nats": ent_sgld,
                          "train_final_loss": res_sgld["final_loss"],
                          "train_final_entropy": res_sgld["final_entropy_nats"],
                          "yielded": res_sgld["yielded"]},
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
