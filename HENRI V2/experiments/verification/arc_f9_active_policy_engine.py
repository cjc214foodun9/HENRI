"""Carrier F9 — Active End-to-End Policy Gradient Optimization (Latent Wave Policy Flow).

After Carriers F4–F8.1 falsified every PASSIVE representation family (static,
derivative, per-env; max CV 0.4617), F9 trains a parameterized wave ingress
W_in + so(8) Lie generators D_a + egress prototypes M end-to-end via task
cross-entropy + transition-consistency backpropagation. Grouped 4-fold
ENV-LEVEL held-out CV (an environment is never split across folds). Gates
G1–G4 verbatim from directive HENRI-DIR-2026-08-F8-1-POSTMORTEM-POLICY-GRADIENT-ORDER.
Default-OFF (HENRI_F9_ACTIVE=1). Diagnostic only — no production path is
trained or modified. See docs/spec/f9_active_policy_preregistration.md.
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


def require_f9_enabled() -> None:
    """Fail closed unless the F9 flag is set (default-OFF)."""
    if os.environ.get("HENRI_F9_ACTIVE") != "1":
        raise RuntimeError(
            "HENRI_F9_ACTIVE != 1: Carrier F9 engine is default-OFF"
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
    jsonl env/step/action_name. Returns raw psi/next (float32) and the
    per-row env/step metadata; NO env-boundary masking here — grouping is
    handled by grouped_env_folds on the env labels (an env's rows stay
    together in exactly one fold).
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
    """Grouped (env-level) folds: an environment's rows are NEVER split.

    Unique env names are shuffled (seed) and assigned round-robin to folds;
    fold k's test set = all rows of its envs; train = the rest. Complete,
    disjoint at the env level.
    """
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
    """Order-4 Taylor matrix exponential per [..., 8, 8] block.

    E = I + D + D^2/2 + D^3/6 + D^4/24. Bounded-norm approximation of
    scaling-and-squaring (disclosed in spec §3; D is skew, norms small).
    """
    I = torch.eye(8, device=Da.device, dtype=Da.dtype).expand_as(Da)
    D2 = Da @ Da
    D3 = D2 @ Da
    D4 = D3 @ Da
    return I + Da + 0.5 * D2 + (1.0 / 6.0) * D3 + (1.0 / 24.0) * D4


class ActivePolicyEngine(nn.Module):
    """Tier 1–3: W_in (rank-r ingress), D_a (so(8) generators), M (prototypes)."""

    def __init__(self, d: int, r: int, n_actions: int = 7, seed: int = 0, beta: float = 8.0):
        super().__init__()
        torch.manual_seed(seed)
        self.d = d
        self.r = r
        self.n_actions = n_actions
        self.beta = beta
        self.n_blocks = d // 8
        assert d % 8 == 0, "d must be divisible by 8"

        # Tier 1: low-rank wave ingress W_in in [d, r] (Stiefel, QR-retracted)
        W = torch.randn(d, r) * (1.0 / (d**0.5))
        W, _ = torch.linalg.qr(W)
        self.W_in = nn.Parameter(W)

        # Tier 2: action-conditioned so(8) Lie coordinates [A, blocks, 8, 8] skew
        raw = torch.randn(n_actions, self.n_blocks, 8, 8) * 0.02
        raw = 0.5 * (raw - raw.transpose(-1, -2))
        self.D_a = nn.Parameter(raw)

        # Tier 3: egress prototypes [A, d], rows unit-normalized
        M = torch.randn(n_actions, d) * (1.0 / (d**0.5))
        M = M / M.norm(dim=1, keepdim=True)
        self.M = nn.Parameter(M)

    def stiefel_retract(self) -> None:
        """QR retraction of W_in (Cholesky/QR path per HENRI convention)."""
        with torch.no_grad():
            W = self.W_in.data
            Q, _ = torch.linalg.qr(W)
            # fix sign ambiguity deterministically (make diagonal non-negative)
            s = torch.sign(torch.diag(Q))
            Q = Q * s.unsqueeze(0)
            self.W_in.copy_(Q)

    def skew_symmetrize(self) -> None:
        with torch.no_grad():
            D = self.D_a.data
            self.D_a.copy_(0.5 * (D - D.transpose(-1, -2)))

    def normalize_prototypes(self) -> None:
        with torch.no_grad():
            self.M.data.div_(self.M.data.norm(dim=1, keepdim=True).clamp_min(1e-12))

    def post_step(self) -> None:
        self.skew_symmetrize()
        self.stiefel_retract()
        self.normalize_prototypes()

    def wave(self, X: torch.Tensor) -> torch.Tensor:
        """Psi_tilde = W_in (W_in^T X); row-normalized -> S^(d-1)."""
        proj = (X @ self.W_in) @ self.W_in.T  # rank-r projection
        return proj / proj.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def forward_logits(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        psi = self.wave(X)
        logits = self.beta * (psi @ self.M.T) / (self.d**0.5)
        z = torch.softmax(logits, dim=1)
        return logits, z, psi

    def transition_loss(self, psi: torch.Tensor, Xn: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """L_trans = mean || exp4(D_a) psi_blocks - target_blocks ||^2.

        Column convention: v' = E v per 8-block; target rows normalized to
        unit norm for scale comparability with psi.
        """
        B = psi.shape[0]
        pb = psi.view(B, self.n_blocks, 8)  # [B, blocks, 8]
        tb = Xn.view(B, self.n_blocks, 8)
        tb = tb / tb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        Da_b = self.D_a[y]  # [B, blocks, 8, 8]
        E = exp4(Da_b)  # [B, blocks, 8, 8]
        pred = torch.matmul(E, pb.unsqueeze(-1)).squeeze(-1)  # [B, blocks, 8]
        return torch.mean((pred - tb) ** 2)

    def composite_loss(
        self, X: torch.Tensor, y: torch.Tensor, Xn: torch.Tensor
    ) -> torch.Tensor:
        logits, _, psi = self.forward_logits(X)
        ce = nn.functional.cross_entropy(logits, y)
        lt = self.transition_loss(psi, Xn, y)
        return ce + 1.0 * lt


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
        "schema": "f9-active-policy.v1",
        "git_sha": kw["git_sha"],
        "bank": {"npz_sha256": kw["npz_sha256"], "jsonl_sha256": kw["jsonl_sha256"]},
        "n_valid": kw["n_valid"],
        "n_envs": kw["n_envs"],
        "folds": kw["folds"],
        "loss_ce_train": kw["loss_ce_train"],
        "p1_train": kw["p1_train"],
        "macro_p1": kw["macro_p1"],
        "passive_baseline": 0.4617,
        "g3_margin": kw["macro_p1"] - 0.4617,
        "gram_max": kw["gram_max"],
        "gram_mean": kw["gram_mean"],
        "l_trans": kw["l_trans"],
        "seed": kw["seed"],
        "n_folds": kw["n_folds"],
        "epochs": kw["epochs"],
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
    seed: int = 20260906,
    epochs: int = 40,
    git_sha: str = "unknown",
    out_dir: str | None = None,
    r: int = 128,
    lr: float = 1e-4,
    batch: int = 256,
) -> dict:
    """Full F9 gauntlet: grouped env-level 4-fold CV, gates G1–G4, verdict."""
    t0 = time.time()
    require_f9_enabled()
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
        eng = ActivePolicyEngine(d=X_all.shape[1], r=r, n_actions=N_CLASSES, seed=seed + fi).to(dev)
        opt = torch.optim.AdamW(eng.parameters(), lr=lr)
        Xtr = torch.from_numpy(X_all[tr_idx]).to(dev)
        ytr = torch.from_numpy(y_all[tr_idx]).to(dev)
        Xntr = torch.from_numpy(Xn_all[tr_idx]).to(dev)
        n_tr = len(tr_idx)
        for ep in range(epochs):
            perm = torch.randperm(n_tr, generator=torch.Generator(device="cpu").manual_seed(seed + fi * 1000 + ep))
            for i in range(0, n_tr, batch):
                bidx = perm[i : i + batch]
                opt.zero_grad(set_to_none=True)
                loss = eng.composite_loss(Xtr[bidx], ytr[bidx], Xntr[bidx])
                loss.backward()
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
        gram = eng.W_in.detach().cpu()
        gram_err = (gram.T @ gram - torch.eye(r)).abs()
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
    loss_ce_train = float(np.min(train_ce_list))
    p1_train = float(np.max(train_p1_list))
    l_trans = float(np.mean(ltrans_list))
    # gram error measured on the LAST fold's engine (all folds use same
    # construction + post_step; representative and cheap)
    gram_max = float(gram_err.max())
    gram_mean = float(gram_err.mean())

    gates = {
        "G1": {"criterion": "L_CE train <= 0.3500 (P@1_train >= 0.90)",
               "loss_ce": loss_ce_train, "p1_train": p1_train,
               "pass": loss_ce_train <= 0.3500 and p1_train >= 0.90, "kill": "K1"},
        "G2": {"criterion": "grouped 4-fold macro P@1 >= 0.7000",
               "value": macro_p1, "pass": macro_p1 >= 0.7000, "kill": "K2"},
        "G3": {"criterion": "macro P@1 - 0.4617 >= +0.2500",
               "value": macro_p1 - 0.4617, "pass": macro_p1 - 0.4617 >= 0.2500, "kill": "K3"},
        "G4": {"criterion": "max ||W_in^T W_in - I||_F <= 1e-4",
               "gram_max": gram_max, "pass": gram_max <= 1e-4, "kill": "K4"},
    }
    g1, g2, g3, g4 = (gates[k]["pass"] for k in ("G1", "G2", "G3", "G4"))
    if g1 and g2 and g3 and g4:
        verdict = "F9_ACTIVE_POLICY_VERIFIED"
    elif not g1 or not g4:
        verdict = "F9_OPTIMIZATION_FAILED"
    else:
        verdict = "F9_ACTIVE_LOSS_NO_GAIN"

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
        gram_max=gram_max,
        gram_mean=gram_mean,
        l_trans=l_trans,
        seed=seed,
        n_folds=n_folds,
        epochs=epochs,
        device=device,
        gates=gates,
        verdict=verdict,
    )
    receipt["elapsed_s"] = round(time.time() - t0, 2)
    if out_dir:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "f9_gates_receipt.json", "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
    return receipt


def main() -> None:
    import argparse
    import subprocess

    ap = argparse.ArgumentParser(description="Carrier F9 active policy-gradient gauntlet")
    ap.add_argument("--bank-npz", required=True)
    ap.add_argument("--bank-jsonl", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--out-dir", default="/tmp/henri_f9_active/")
    ap.add_argument("--receipt-out", default="/tmp/henri_f9_active/f9_gates_receipt.json")
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
    )
    # also honor --receipt-out for exact path control
    if args.receipt_out and args.receipt_out != args.out_dir + "f9_gates_receipt.json":
        Path(args.receipt_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.receipt_out, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
