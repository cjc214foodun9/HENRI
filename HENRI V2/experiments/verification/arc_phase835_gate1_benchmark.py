"""Phase 8.35 Sprint a — 5-arm VLA Gate 1 benchmark (production D).

Arms on the AUTHORIZED bank (90 records, 72/18, seed 20260819):
  A: RecursiveDualEDMD r=16            (sealed 8.33 linear baseline)
  B: CoupledRecursiveDualEDMD r=128    (8.34 ACCEPT arm)
  C: CoupledRecursiveDualEDMD r=128    (field OFF = control)
  D: DirectionalTravelingWaveCoupler AP r=128 (+k)
  E: DirectionalTravelingWaveCoupler PA r=128 (-k)

Metrics: held-out 1-cos transition loss; egress action accuracy via
DualScaleAnalogLexicalSnap (top_k=512) over a 6-action prototype
codebook built from TRAIN-split next-waves (no leakage); I_norm =
I(Y;Y_hat)/H(Y) from snap confidences; directional Sagnac (T3).

Pre-registered gate (arc_phase835_gate1_prereg.md):
  ACCEPT: D_holdout <= 0.15 AND I_norm >= 0.85 AND acc >= 0.80
  PARTIAL: transition fixed but egress unmet
  FAIL: transition not fixed
  BLOCKED_INFRA: NaN / digest mismatch / GPU failure

Output: compact JSON verdict on stdout (last line).
"""

import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from henri_trajectory_bank import TrajectoryBank
from recursive_dual_edmd import (
    CoupledRecursiveDualEDMD,
    DirectionalTravelingWaveCoupler,
    RecursiveDualEDMD,
)
from hopfield_cleanup import DualScaleAnalogLexicalSnap
from efe_planner import EFEPlanner

DEFAULT_SEED = 20260819
HELDOUT_FRAC = 0.2
NUM_BLOCKS = 8192
BLOCK_DIM = 8
D = NUM_BLOCKS * BLOCK_DIM
RANK_COUPLED = 128
RANK_BASELINE = 16
SNAP_TOPK = 512


def _action_wave_for(idx: int, num_blocks: int = NUM_BLOCKS, block_dim: int = BLOCK_DIM,
                     device: str = "cpu") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(1000 + idx)
    w = torch.randn(num_blocks, block_dim, generator=g)
    w = w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)
    return w.to(device)


def split_indices(n: int, frac: float, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_test = max(1, int(round(n * frac)))
    return perm[n_test:].tolist(), perm[:n_test].tolist()


def _holdout_loss(predict, psi_ho, nxt_ho, y_ho, dev):
    total = 0.0
    for s, y, a in zip(psi_ho, nxt_ho, y_ho):
        s_w = torch.from_numpy(s).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        a_w = _action_wave_for(int(a), device=dev.type)
        y_w = torch.from_numpy(y).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        with torch.no_grad():
            pred = predict(s_w, a_w).view(-1)
            cos = torch.dot(F.normalize(pred, p=2, dim=0),
                            F.normalize(y_w.view(-1), p=2, dim=0))
            total += float(1.0 - cos.item())
    return total / len(psi_ho)


def _egress_eval(predict, psi_ho, nxt_ho, y_ho, dev, codebook, macro, snap):
    """Action accuracy + I_norm from predicted next-wave -> lexical snap.

    codebook: [6, D] prototype waves (train-split means). macro: [2048]
    macro-option wave. No labels from held-out are used to build either.
    """
    n = len(psi_ho)
    n_actions = codebook.shape[0]
    confs = np.zeros((n, n_actions))
    preds = np.zeros(n, dtype=np.int64)
    sags = []
    for i, (s, y, a) in enumerate(zip(psi_ho, nxt_ho, y_ho)):
        s_w = torch.from_numpy(s).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        a_w = _action_wave_for(int(a), device=dev.type)
        y_w = torch.from_numpy(y).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        with torch.no_grad():
            pred = predict(s_w, a_w).view(-1)
            y_norm = F.normalize(y_w.view(-1), p=2, dim=0)
            p_norm = F.normalize(pred, p=2, dim=0)
            sags.append(float(1.0 - torch.dot(p_norm, y_norm).item()))
            idx, conf = snap.snap(pred, macro, top_k=1)
            preds[i] = int(idx.item())
            sims = torch.nn.functional.normalize(pred, p=2, dim=0) @ codebook.T
            confs[i] = torch.softmax(10.0 * sims, dim=-1).cpu().numpy()
    acc = float((preds == np.asarray(y_ho)).mean())
    # I(Y; Y_hat) via predicted-vs-empirical distributions.
    p_yhat = confs.mean(axis=0) + 1e-12
    p_y = np.bincount(np.asarray(y_ho, dtype=np.int64),
                      minlength=n_actions).astype(np.float64) / n + 1e-12
    p_y /= p_y.sum()
    p_yhat /= p_yhat.sum()
    joint = np.zeros((n_actions, n_actions))
    for i, yi in enumerate(y_ho):
        joint[int(yi), preds[i]] += confs[i, preds[i]]
    joint /= max(joint.sum(), 1e-12)
    mi = 0.0
    for i in range(n_actions):
        for j in range(n_actions):
            if joint[i, j] > 0:
                mi += joint[i, j] * math.log(joint[i, j] / (p_y[i] * p_yhat[j] + 1e-12))
    h_y = -sum(p * math.log(p) for p in p_y if p > 0)
    i_norm = mi / h_y if h_y > 0 else 0.0
    return acc, i_norm, float(np.mean(sags))


def run_experiment(bank_npz: str, manifest_path: str, device: str, seed: int,
                   limit: int = 0) -> dict:
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()

    data = TrajectoryBank.load(bank_npz, manifest_path, verify_digest=True)
    psi = data["psi"]
    onehot = data["actions_onehot"]
    nxt = data["next_wave"]
    if nxt is None or psi.shape[0] != nxt.shape[0]:
        raise RuntimeError("bank missing next_wave records")
    if psi.shape[1] != D:
        raise RuntimeError(f"bank wave dim {psi.shape[1]} != {D} (production only)")
    M = psi.shape[0]
    action_idx = torch.from_numpy(onehot).argmax(dim=-1).long()
    run_id = str(data["manifest"].get("run_id", "unknown"))

    train_idx, held_idx = split_indices(M, HELDOUT_FRAC, seed)
    if limit > 0:
        train_idx = train_idx[:limit]
    psi_tr = psi[train_idx]; nxt_tr = nxt[train_idx]
    psi_ho = psi[held_idx];  nxt_ho = nxt[held_idx]
    y_tr = action_idx[train_idx].numpy(); y_ho = action_idx[held_idx].numpy()

    # ---- Egress codebook: train-split prototype next-waves per action ----
    codebook = np.zeros((int(onehot.shape[1]), D), dtype=np.float32)
    for a in range(int(onehot.shape[1])):
        sel = np.where(y_tr == a)[0]
        if len(sel):
            codebook[a] = psi_tr[sel].mean(axis=0)
    cb = torch.from_numpy(F.normalize(torch.from_numpy(codebook), p=2, dim=-1).numpy())
    cb = F.normalize(torch.from_numpy(codebook), p=2, dim=-1).to(dev)
    macro = F.normalize(torch.from_numpy(psi_tr.mean(axis=0)), p=2, dim=0).to(dev)
    snap = DualScaleAnalogLexicalSnap(dim_micro=D, dim_macro=2048, tau=1.0,
                                      top_k=SNAP_TOPK).to(dev)
    snap.store_engrams(cb)

    # ---- Arm A: sealed r=16 linear baseline ----
    torch.manual_seed(seed)
    edmd = RecursiveDualEDMD(d_model=D, r_rank=RANK_BASELINE, lambda_forget=0.98).to(dev)
    for i in range(len(train_idx)):
        s = torch.from_numpy(psi_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        a = _action_wave_for(int(y_tr[i]), device=dev.type)
        y = torch.from_numpy(nxt_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        edmd.update_online_step(s, a, y)
    loss_a = _holdout_loss(edmd, psi_ho, nxt_ho, y_ho, dev)
    acc_a, inorm_a, sag_a = _egress_eval(edmd, psi_ho, nxt_ho, y_ho, dev, cb, macro, snap)

    # ---- Arm B: coupled r=128 field ON ----
    torch.manual_seed(seed + 1)
    coupled = CoupledRecursiveDualEDMD(
        d_model=D, r_rank=RANK_COUPLED, lambda_forget=0.98,
        num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM, field_channel=True).to(dev)
    for i in range(len(train_idx)):
        s = torch.from_numpy(psi_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        a = _action_wave_for(int(y_tr[i]), device=dev.type)
        y = torch.from_numpy(nxt_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        coupled.update_online_step(s, a, y)
    loss_b = _holdout_loss(coupled, psi_ho, nxt_ho, y_ho, dev)
    acc_b, inorm_b, sag_b = _egress_eval(coupled, psi_ho, nxt_ho, y_ho, dev, cb, macro, snap)

    # ---- Arm C: coupled r=128 field OFF (control) ----
    torch.manual_seed(seed + 2)
    control = CoupledRecursiveDualEDMD(
        d_model=D, r_rank=RANK_COUPLED, lambda_forget=0.98,
        num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM, field_channel=False).to(dev)
    for i in range(len(train_idx)):
        s = torch.from_numpy(psi_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        a = _action_wave_for(int(y_tr[i]), device=dev.type)
        y = torch.from_numpy(nxt_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        control.update_online_step(s, a, y)
    loss_c = _holdout_loss(control, psi_ho, nxt_ho, y_ho, dev)
    acc_c, inorm_c, sag_c = _egress_eval(control, psi_ho, nxt_ho, y_ho, dev, cb, macro, snap)

    # ---- Arm D: directional AP r=128 ----
    torch.manual_seed(seed + 3)
    ap = DirectionalTravelingWaveCoupler(
        d_model=D, r_rank=RANK_COUPLED, lambda_forget=0.98,
        num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM, field_channel=True,
        direction="AP", k_max=2.0).to(dev)
    for i in range(len(train_idx)):
        s = torch.from_numpy(psi_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        a = _action_wave_for(int(y_tr[i]), device=dev.type)
        y = torch.from_numpy(nxt_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        ap.update_online_step(s, a, y)
    loss_d = _holdout_loss(ap, psi_ho, nxt_ho, y_ho, dev)
    acc_d, inorm_d, sag_d = _egress_eval(ap, psi_ho, nxt_ho, y_ho, dev, cb, macro, snap)

    # ---- Arm E: directional PA r=128 ----
    torch.manual_seed(seed + 4)
    pa = DirectionalTravelingWaveCoupler(
        d_model=D, r_rank=RANK_COUPLED, lambda_forget=0.98,
        num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM, field_channel=True,
        direction="PA", k_max=2.0).to(dev)
    for i in range(len(train_idx)):
        s = torch.from_numpy(psi_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        a = _action_wave_for(int(y_tr[i]), device=dev.type)
        y = torch.from_numpy(nxt_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        pa.update_online_step(s, a, y)
    loss_e = _holdout_loss(pa, psi_ho, nxt_ho, y_ho, dev)
    acc_e, inorm_e, sag_e = _egress_eval(pa, psi_ho, nxt_ho, y_ho, dev, cb, macro, snap)

    vals = [loss_a, loss_b, loss_c, loss_d, loss_e, acc_d, inorm_d]
    verdict, reason = "FAIL", ""
    if not all(math.isfinite(v) for v in vals):
        verdict, reason = "BLOCKED_INFRA", "NaN in metrics"
    elif loss_d <= 0.15 and inorm_d >= 0.85 and acc_d >= 0.80:
        verdict = "ACCEPT"
        reason = (f"D {loss_d:.4f} <= 0.15, I_norm {inorm_d:.4f} >= 0.85, "
                  f"acc {acc_d:.4f} >= 0.80")
    elif loss_d <= 0.15:
        verdict = "PARTIAL"
        reason = (f"transition fixed (D {loss_d:.4f} <= 0.15) but egress unmet: "
                  f"I_norm {inorm_d:.4f}, acc {acc_d:.4f}")
    else:
        reason = (f"D {loss_d:.4f} > 0.15 (sealed 8.34 = 0.3153); "
                  f"I_norm {inorm_d:.4f}, acc {acc_d:.4f}")

    result = {
        "schema": "henri.phase835.gate1-benchmark.v1",
        "verdict": verdict,
        "reason": reason,
        "run_id": run_id,
        "bank_npz_sha256": data["manifest"].get("npz_sha256", ""),
        "records": {"total": M, "train": len(train_idx), "heldout": len(held_idx)},
        "split": {"held_out_frac": HELDOUT_FRAC, "seed": seed},
        "gate": ("ACCEPT iff D <= 0.15 AND I_norm >= 0.85 AND acc >= 0.80; "
                 "PARTIAL if transition fixed only; FAIL otherwise; "
                 "BLOCKED_INFRA on NaN"),
        "metrics": {
            "A_linear_r16_holdout": loss_a, "A_acc": acc_a, "A_inorm": inorm_a,
            "B_coupled_r128_holdout": loss_b, "B_acc": acc_b, "B_inorm": inorm_b,
            "C_control_r128_holdout": loss_c, "C_acc": acc_c, "C_inorm": inorm_c,
            "D_AP_r128_holdout": loss_d, "D_acc": acc_d, "D_inorm": inorm_d,
            "E_PA_r128_holdout": loss_e, "E_acc": acc_e, "E_inorm": inorm_e,
            "sagnac_directional": {"A": sag_a, "B": sag_b, "C": sag_c,
                                   "D": sag_d, "E": sag_e},
        },
        "device": dev.type,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 8.35 Gate 1 benchmark")
    p.add_argument("--bank", required=True)
    p.add_argument("--manifest", default="")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="")
    p.add_argument("--limit", type=int, default=0, help="bounded smoke (train records)")
    args = p.parse_args(argv)
    print(f"== Phase 8.35 Gate 1 benchmark ==\nbank: {args.bank}\n"
          f"seed: {args.seed} device: {args.device or 'auto'} limit: {args.limit or 'full'}",
          flush=True)
    result = run_experiment(args.bank, args.manifest or None, args.device, args.seed,
                            args.limit)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
