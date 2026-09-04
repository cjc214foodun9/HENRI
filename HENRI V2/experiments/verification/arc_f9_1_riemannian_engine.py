"""Carrier F9.1 — Riemannian Optimization & Manifold-Preserving Gradient Alignment.

Post-mortem of Carrier F9 (F9_OPTIMIZATION_FAILED, seal #be40617c): Euclidean
AdamW momentum is destroyed by per-step hard QR retraction on W_in, and
eta=1e-4 through the 65,536->128 bottleneck attenuated gradients (L_CE stuck
at ln 7). F9.1 replaces the constrained Stiefel path with an UNCONSTRAINED
pre-activation adapter: LayerNorm(W_down x + W_skip x) + residual, rank-512
bottleneck, exact L2 sphere projection, peak lr 1e-2 under cosine annealing
with warmup, gradient clipping ||g||<=1.0, joint L = L_CE + 0.1*L_transition.
TimesFM-3 synthesis dispositions (HENRI-SYNTH-2026-08-TIMESFM3-TRANSLATION):
ingress pattern (LayerNorm + up-proj + L2 normalize) implemented; patch p=32 /
lookahead goals NOT_APPLICABLE (bank inputs already embedded; no covariate
source); K=8 single-pass unroll NOT_APPLICABLE (classification CV); Zone C
dual-stream BLOCKED (prod DSN). Default-OFF (HENRI_F9_1_ACTIVE=1). Diagnostic
only. See docs/spec/f9_1_riemannian_preregistration.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

D = 65_536
N_CLASSES = 7


def require_f9_1_enabled() -> None:
    """Fail closed unless the F9.1 flag is set (default-OFF)."""
    if os.environ.get("HENRI_F9_1_ACTIVE") != "1":
        raise RuntimeError(
            "HENRI_F9_1_ACTIVE != 1: Carrier F9.1 engine is default-OFF"
        )


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank(npz_path: str, jsonl_path: str) -> dict:
    """Load and validate the REAL bank schema (F8 amendment precedent).

    psi/next_wave real float16 [N, D], actions_onehot uint8 [N, 7],
    jsonl env/step/action_name. All rows are valid for classification
    (env-boundary rows are only invalid for DeltaPsi pairing).
    """
    data = np.load(npz_path, allow_pickle=False)
    psi = data["psi"]
    nxt = data["next_wave"]
    onehot = data["actions_onehot"]
    if psi.ndim != 2 or nxt.ndim != 2:
        raise ValueError(f"psi/next_wave must be 2-D [N, D], got {psi.shape}/{nxt.shape}")
    if psi.shape != nxt.shape:
        raise ValueError(f"psi/next_wave shape mismatch: {psi.shape} != {nxt.shape}")
    if np.iscomplexobj(psi) or np.iscomplexobj(nxt):
        raise ValueError("psi/next_wave must be REAL (bank is float16 real-domain)")
    if onehot.shape != (len(psi), N_CLASSES):
        raise ValueError(f"actions_onehot must be [N, {N_CLASSES}], got {onehot.shape}")
    y = np.argmax(onehot.astype(np.float32), axis=1).astype(np.int64)
    with open(jsonl_path, "r", encoding="utf-8") as fp:
        meta = [json.loads(line) for line in fp]
    if len(meta) != len(psi):
        raise ValueError(f"jsonl/meta row mismatch: {len(meta)} != {len(psi)}")
    envs = [str(m["env"]) for m in meta]
    steps = [m.get("step") for m in meta]
    return {
        "psi": psi.astype(np.float32),
        "next": nxt.astype(np.float32),
        "y": y,
        "envs": envs,
        "steps": steps,
        "npz_sha256": sha256_file(npz_path),
        "jsonl_sha256": sha256_file(jsonl_path),
    }


def grouped_env_folds(
    envs: list[str], n_folds: int = 4, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped (env-level) folds: an environment's rows are NEVER split."""
    rng = np.random.default_rng(seed)
    unique = sorted(set(envs))
    rng.shuffle(unique)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, e in enumerate(unique):
        folds[i % n_folds].append(e)
    out = []
    for k in range(n_folds):
        te_envs = set(folds[k])
        te = [i for i, e in enumerate(envs) if e in te_envs]
        tr = [i for i, e in enumerate(envs) if e not in te_envs]
        out.append(
            (np.array(sorted(tr), dtype=np.int64), np.array(sorted(te), dtype=np.int64))
        )
    return out


def exp4(Da: torch.Tensor) -> torch.Tensor:
    """Order-4 Taylor matrix exponential per [..., 8, 8] block (F9 precedent)."""
    I = torch.eye(8, device=Da.device, dtype=Da.dtype).expand_as(Da)
    D2 = Da @ Da
    D3 = D2 @ Da
    D4 = D3 @ Da
    return I + Da + 0.5 * D2 + (1.0 / 6.0) * D3 + (1.0 / 24.0) * D4


def cosine_anneal_lr(
    epoch: int, epochs: int, lr_max: float = 1e-2, lr_min: float = 1e-5, warmup: int = 2
) -> float:
    """Linear warmup then cosine anneal lr_max -> lr_min over epochs."""
    if epoch < warmup:
        return lr_max * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, (epochs - warmup - 1))
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * t))


class RiemannianPolicyEngine(nn.Module):
    """Tier 1–3 with UNCONSTRAINED pre-activation adapter (no QR retraction).

    h = LayerNorm(W_down x + W_skip x) in R^r; Psi = normalize(W_up h) on
    S^(D-1); logits = beta * <Psi, M_a> / sqrt(D). D_a skew so(8) per block;
    M row-normalized prototypes. Only D_a skew-symmetry and M row norms are
    maintained in post_step; W_down/W_up are FREE (full AdamW momentum).
    """

    def __init__(self, d: int, r: int, n_actions: int = 7, seed: int = 0, beta: float = 8.0):
        super().__init__()
        torch.manual_seed(seed)
        self.d = d
        self.r = r
        self.n_actions = n_actions
        self.beta = beta
        self.n_blocks = d // 8
        assert d % 8 == 0, "d must be divisible by 8"

        # Tier 1: unconstrained bottleneck adapter + residual skip
        self.W_down = nn.Parameter(torch.randn(r, d) * (1.0 / (d**0.5)))
        self.W_skip = nn.Parameter(torch.randn(r, d) * (1.0 / (d**0.5)))
        self.ln = nn.LayerNorm(r)
        self.W_up = nn.Parameter(torch.randn(d, r) * (1.0 / (r**0.5)))

        # Tier 2: action-conditioned so(8) Lie coordinates, skew
        raw = torch.randn(n_actions, self.n_blocks, 8, 8) * 0.02
        raw = 0.5 * (raw - raw.transpose(-1, -2))
        self.D_a = nn.Parameter(raw)

        # Tier 3: egress prototypes, rows unit-normalized
        M = torch.randn(n_actions, d) * (1.0 / (d**0.5))
        M = M / M.norm(dim=1, keepdim=True)
        self.M = nn.Parameter(M)

    def skew_symmetrize(self) -> None:
        with torch.no_grad():
            D = self.D_a.data
            self.D_a.copy_(0.5 * (D - D.transpose(-1, -2)))

    def normalize_prototypes(self) -> None:
        with torch.no_grad():
            self.M.data.div_(self.M.data.norm(dim=1, keepdim=True).clamp_min(1e-12))

    def post_step(self) -> None:
        self.skew_symmetrize()
        self.normalize_prototypes()

    def wave(self, X: torch.Tensor) -> torch.Tensor:
        """h = LayerNorm(W_down x + W_skip x); Psi = normalize(W_up h) in S^(d-1)."""
        h = self.ln(X @ self.W_down.T + X @ self.W_skip.T)
        psi = X.new_zeros(X.shape[0], self.d)
        # W_up @ h is [d, B] -> transpose to [B, d]; chunked for memory
        bs = 64
        outs = []
        for i in range(0, X.shape[0], bs):
            hb = h[i : i + bs]
            outs.append((hb @ self.W_up.T))
        psi = torch.cat(outs, dim=0)
        return psi / psi.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def forward_logits(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        psi = self.wave(X)
        logits = self.beta * (psi @ self.M.T) / (self.d**0.5)
        z = torch.softmax(logits, dim=1)
        return logits, z, psi

    def transition_loss(self, psi: torch.Tensor, Xn: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """L_trans = mean || exp4(D_a) psi_blocks - target_blocks ||^2 (row-normalized target)."""
        B = psi.shape[0]
        pb = psi.view(B, self.n_blocks, 8)
        tb = Xn.view(B, self.n_blocks, 8)
        tb = tb / tb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Da_b = self.D_a[y]
        E = exp4(Da_b)
        pred = torch.matmul(E, pb.unsqueeze(-1)).squeeze(-1)
        return torch.mean((pred - tb) ** 2)

    def composite_loss(
        self, X: torch.Tensor, y: torch.Tensor, Xn: torch.Tensor, alpha: float = 0.1
    ) -> torch.Tensor:
        logits, _, psi = self.forward_logits(X)
        ce = nn.functional.cross_entropy(logits, y)
        lt = self.transition_loss(psi, Xn, y)
        return ce + alpha * lt


def _macro_p1(logits: np.ndarray, y: np.ndarray) -> float:
    preds = logits.argmax(axis=1)
    accs = []
    for c in range(int(y.max()) + 1):
        mask = y == c
        if mask.sum() > 0:
            accs.append(float((preds[mask] == c).mean()))
    if not accs:
        return 0.0
    return float(np.mean(accs))


def build_receipt(**kw) -> dict:
    return {
        "schema": "f9-1-riemannian.v1",
        "git_sha": kw["git_sha"],
        "bank": {"npz_sha256": kw["npz_sha256"], "jsonl_sha256": kw["jsonl_sha256"]},
        "n_valid": kw["n_valid"],
        "n_envs": kw["n_envs"],
        "folds": kw["folds"],
        "loss_ce_train": kw["loss_ce_train"],
        "p1_train": kw["p1_train"],
        "macro_p1": kw["macro_p1"],
        "min_fold_p1": kw["min_fold_p1"],
        "passive_baseline": 0.4617,
        "g3_margin": kw["macro_p1"] - 0.4617,
        "l_trans": kw["l_trans"],
        "seed": kw["seed"],
        "n_folds": kw["n_folds"],
        "epochs": kw["epochs"],
        "rank": kw["rank"],
        "lr_max": kw["lr_max"],
        "device": kw["device"],
        "gates": kw["gates"],
        "verdict": kw["verdict"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def run_gauntlet(
    bank_npz: str,
    bank_jsonl: str,
    device: str = "cuda",
    n_folds: int = 4,
    seed: int = 20260907,
    epochs: int = 40,
    git_sha: str = "unknown",
    out_dir: str | None = None,
    r: int = 512,
    lr_max: float = 1e-2,
    batch: int = 256,
) -> dict:
    """Full F9.1 gauntlet: grouped env-level 4-fold CV, gates G1–G4, verdict."""
    t0 = time.time()
    require_f9_1_enabled()
    data = load_bank(bank_npz, bank_jsonl)
    X_all = data["psi"]
    Xn_all = data["next"]
    y_all = data["y"]
    envs = data["envs"]
    n_envs = len(set(envs))
    folds = grouped_env_folds(envs, n_folds=n_folds, seed=seed)

    dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    fold_rows = []
    macro_list = []
    train_ce_list = []
    train_p1_list = []
    ltrans_list = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        eng = RiemannianPolicyEngine(d=X_all.shape[1], r=r, n_actions=N_CLASSES, seed=seed + fi).to(dev)
        Xtr = torch.from_numpy(X_all[tr_idx]).to(dev)
        ytr = torch.from_numpy(y_all[tr_idx]).to(dev)
        Xntr = torch.from_numpy(Xn_all[tr_idx]).to(dev)
        n_tr = len(tr_idx)
        opt = torch.optim.AdamW(eng.parameters(), lr=lr_max)
        for ep in range(epochs):
            lr = cosine_anneal_lr(ep, epochs, lr_max=lr_max, lr_min=1e-5, warmup=2)
            for g in opt.param_groups:
                g["lr"] = lr
            perm = torch.randperm(n_tr, generator=torch.Generator(device="cpu").manual_seed(seed + fi * 1000 + ep))
            for i in range(0, n_tr, batch):
                bidx = perm[i : i + batch]
                opt.zero_grad(set_to_none=True)
                loss = eng.composite_loss(Xtr[bidx], ytr[bidx], Xntr[bidx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(eng.parameters(), 1.0)
                opt.step()
                eng.post_step()
        eng.eval()
        with torch.no_grad():
            logits_tr, _, psi_tr = eng.forward_logits(Xtr)
            ce_tr = nn.functional.cross_entropy(logits_tr, ytr).item()
            p1_tr = float((logits_tr.argmax(1) == ytr).float().mean().item())
            lt = eng.transition_loss(psi_tr, Xntr, ytr).item()
            Xte = torch.from_numpy(X_all[te_idx]).to(dev)
            logits_te, _, _ = eng.forward_logits(Xte)
        macro = _macro_p1(logits_te.cpu().numpy(), y_all[te_idx])
        macro_list.append(macro)
        train_ce_list.append(ce_tr)
        train_p1_list.append(p1_tr)
        ltrans_list.append(lt)
        fold_rows.append({
            "fold": fi,
            "test_envs": sorted({envs[i] for i in te_idx}),
            "n_train": len(tr_idx),
            "n_test": len(te_idx),
            "macro_p1": round(macro, 4),
            "train_ce": round(ce_tr, 4),
            "train_p1": round(p1_tr, 4),
        })

    macro_p1 = float(np.mean(macro_list))
    min_fold_p1 = float(np.min(macro_list))
    loss_ce_train = float(np.mean(train_ce_list))
    p1_train = float(np.mean(train_p1_list))
    l_trans = float(np.mean(ltrans_list))

    gates = {
        "G1": {"criterion": "L_CE train <= 0.5000 AND P@1_train >= 0.8500",
               "loss_ce": loss_ce_train, "p1_train": p1_train,
               "pass": loss_ce_train <= 0.5000 and p1_train >= 0.8500, "kill": "K1"},
        "G2": {"criterion": "grouped 4-fold macro P@1 >= 0.6500",
               "value": macro_p1, "pass": macro_p1 >= 0.6500, "kill": "K2"},
        "G3": {"criterion": "macro P@1 - 0.4617 >= +0.2000",
               "value": macro_p1 - 0.4617, "pass": macro_p1 - 0.4617 >= 0.2000, "kill": "K3"},
        "G4": {"criterion": "min single-fold macro P@1 >= 0.5000",
               "value": min_fold_p1, "pass": min_fold_p1 >= 0.5000, "kill": "K4"},
    }
    g1, g2, g3, g4 = (gates[k]["pass"] for k in ("G1", "G2", "G3", "G4"))
    if g1 and g2 and g3 and g4:
        verdict = "F9_1_RIEMANNIAN_VERIFIED"
    elif not g1:
        verdict = "F9_1_OPTIMIZATION_FAILED"
    else:
        verdict = "F9_1_ACTIVE_NO_GAIN"

    receipt = build_receipt(
        git_sha=git_sha,
        npz_sha256=data["npz_sha256"],
        jsonl_sha256=data["jsonl_sha256"],
        n_valid=len(X_all),
        n_envs=n_envs,
        folds=fold_rows,
        loss_ce_train=loss_ce_train,
        p1_train=p1_train,
        macro_p1=macro_p1,
        min_fold_p1=min_fold_p1,
        l_trans=l_trans,
        seed=seed,
        n_folds=n_folds,
        epochs=epochs,
        rank=r,
        lr_max=lr_max,
        device=device,
        gates=gates,
        verdict=verdict,
    )
    receipt["elapsed_s"] = round(time.time() - t0, 2)
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "f9_1_gates_receipt.json", "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
    return receipt


def main() -> None:
    import argparse
    import subprocess

    ap = argparse.ArgumentParser(description="Carrier F9.1 Riemannian optimization gauntlet")
    ap.add_argument("--bank-npz", required=True)
    ap.add_argument("--bank-jsonl", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--rank", type=int, default=512)
    ap.add_argument("--lr-max", type=float, default=0.01)
    ap.add_argument("--out-dir", default="/tmp/henri_f9_1_riemannian/")
    ap.add_argument("--receipt-out", default="/tmp/henri_f9_1_riemannian/f9_1_gates_receipt.json")
    args = ap.parse_args()

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30
        ).stdout.strip()
    except Exception:
        sha = "unknown"

    receipt = run_gauntlet(
        bank_npz=args.bank_npz,
        bank_jsonl=args.bank_jsonl,
        device=args.device,
        n_folds=args.n_folds,
        seed=args.seed,
        epochs=args.epochs,
        git_sha=sha,
        out_dir=args.out_dir,
        r=args.rank,
        lr_max=args.lr_max,
    )
    if args.receipt_out and args.receipt_out != args.out_dir + "f9_1_gates_receipt.json":
        Path(args.receipt_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.receipt_out, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
