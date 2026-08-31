"""Carrier F8 — Direct Supervised Linear-Decodability Probe.

Diagnostic on the F3 v2 trajectory bank: can ANY unconstrained supervised
model predict action labels above the majority baseline from the raw bank
waves? Four probe families (P1 min-norm LS, P2 logistic, P3 MLP, P4 k-NN)
plus a temporal-difference arm (G4), 10-fold stratified CV, default-OFF
(HENRI_F8_PROBE=1). Diagnostic only — no production path is trained or
modified. See docs/spec/f8_decodability_probe_preregistration.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

D = 65_536  # production wave dimension (complex)
N_CLASSES = 7  # action labels 1..7
LABEL_MIN, LABEL_MAX = 1, 7


def require_f8_enabled() -> None:
    """Fail closed unless the F8 probe flag is set (default-OFF)."""
    if os.environ.get("HENRI_F8_PROBE") != "1":
        raise RuntimeError(
            "HENRI_F8_PROBE != 1: Carrier F8 probe is default-OFF"
        )


def load_bank(npz_path: str, jsonl_path: str) -> dict:
    """Load and validate the REAL bank schema. Raises ValueError on mismatch.

    OBSERVED schema (F3 v2 bank, hashes 9e3c01b4/1ca089b2): npz keys
    psi/next_wave/actions_onehot/action_names; psi is REAL float16
    [N, 65536] — the directive's complex-assumption is corrected to the
    real domain (F7 Appendix-B precedent). Env segmentation comes from the
    jsonl 'env' field in row order (jsonl count must equal npz rows).
    """
    data = np.load(npz_path, allow_pickle=False)
    psi = data["psi"]
    actions_onehot = data["actions_onehot"]
    if psi.ndim != 2:
        raise ValueError(f"psi must be 2-D [N, D], got {psi.shape}")
    if np.iscomplexobj(psi):
        raise ValueError(f"psi must be real float16/float32, got {psi.dtype}")
    if actions_onehot.shape != (len(psi), N_CLASSES):
        raise ValueError(
            f"actions_onehot must be [N, {N_CLASSES}], got {actions_onehot.shape}"
        )
    y = np.argmax(actions_onehot.astype(np.float32), axis=1).astype(np.int64)
    with open(jsonl_path, "r", encoding="utf-8") as fp:
        meta = [json.loads(line) for line in fp]
    if len(meta) != len(psi):
        raise ValueError(
            f"jsonl/meta row mismatch: {len(meta)} != {len(psi)}"
        )
    envs = [str(m["env"]) for m in meta]
    env_code = {e: i for i, e in enumerate(sorted(set(envs)))}
    env_ids = np.array([env_code[e] for e in envs], dtype=np.int64)
    return {
        "psi": psi.astype(np.float32),
        "y": y,
        "env_ids": env_ids,
        "env_names": sorted(set(envs)),
    }


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def features(psi: np.ndarray) -> np.ndarray:
    """Real feature map, representation-aware.

    complex psi -> concat(Re Psi, Im Psi) -> [N, 2D] float32.
    REAL psi (observed bank) -> psi itself -> [N, D] float32 (no duplicate
    concat of identical real/imag copies).
    """
    psi = np.asarray(psi)
    if np.iscomplexobj(psi):
        return np.concatenate([psi.real, psi.imag], axis=1).astype(np.float32)
    return psi.astype(np.float32)


def majority_baseline(y: np.ndarray) -> float:
    vals, counts = np.unique(y, return_counts=True)
    return float(counts.max()) / float(len(y))


def stratified_folds(
    y: np.ndarray, n_folds: int = 10, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified (by class) fold indices; disjoint, complete."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    per_class_idx = {c: np.where(y == c)[0] for c in classes}
    for c in classes:
        rng.shuffle(per_class_idx[c])
    fold_lists: list[list[int]] = [[] for _ in range(n_folds)]
    for c in classes:
        idx = per_class_idx[c]
        for i, j in enumerate(idx):
            fold_lists[i % n_folds].append(int(j))
    folds = []
    for k in range(n_folds):
        test = np.array(sorted(fold_lists[k]), dtype=np.int64)
        train = np.array(
            sorted(set(range(len(y))) - set(fold_lists[k])), dtype=np.int64
        )
        folds.append((train, test))
    return folds


# --------------------------------------------------------------------------
# Probe 1 — min-norm least squares (over-parameterized interpolant)
# --------------------------------------------------------------------------
def fit_minnorm_ls(X: np.ndarray, Y: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """W = V S (S^2 + lam I)^-1 U^T Y via thin SVD (dual; no [D,D])."""
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    if torch.cuda.is_available():
        Xt, Yt = Xt.cuda(), Yt.cuda()
    U, S, Vt = torch.linalg.svd(Xt, full_matrices=False)
    Sinv = S / (S * S + lam)
    W = (Vt.T * Sinv) @ (U.T @ Yt)
    return W.cpu().numpy()


def predict_ls(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    Xt = torch.from_numpy(X)
    Wt = torch.from_numpy(W)
    if torch.cuda.is_available():
        Xt, Wt = Xt.cuda(), Wt.cuda()
    pred = (Xt @ Wt).argmax(dim=1)
    return pred.cpu().numpy().astype(np.int64)


# --------------------------------------------------------------------------
# Probe 2 — multinomial logistic regression (torch, L2 sweep)
# --------------------------------------------------------------------------
class _Logistic(nn.Module):
    def __init__(self, d_in: int, k: int):
        super().__init__()
        self.fc = nn.Linear(d_in, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
    patience: int = 12,
    batch: int = 512,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(X)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    if dev == "cuda":
        Xt, yt = Xt.cuda(), yt.cuda()
    # hold out a small val slice for early stopping
    perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(seed))
    n_val = max(1, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    model = _Logistic(X.shape[1], int(y.max()) + 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lam)
    lossf = nn.CrossEntropyLoss()
    best_val, best_state, stall = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(tr_idx), generator=torch.Generator(device="cpu").manual_seed(seed + ep))
        for i in range(0, len(order), batch):
            bidx = tr_idx[order[i : i + batch]]
            opt.zero_grad(set_to_none=True)
            out = model(Xt[bidx])
            loss = lossf(out, yt[bidx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = lossf(model(Xt[val_idx]), yt[val_idx]).item()
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    W = model.fc.weight.detach().cpu().numpy()
    b = model.fc.bias.detach().cpu().numpy()
    return W, b


def predict_logistic(X: np.ndarray, Wb: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    W, b = Wb
    Xt = torch.from_numpy(X)
    Wt = torch.from_numpy(W)
    bt = torch.from_numpy(b)
    if torch.cuda.is_available():
        Xt, Wt, bt = Xt.cuda(), Wt.cuda(), bt.cuda()
    pred = (Xt @ Wt.T + bt).argmax(dim=1)
    return pred.cpu().numpy().astype(np.int64)


# --------------------------------------------------------------------------
# Probe 3 — 3-layer MLP (65k -> 1024 -> 256 -> 7, GELU, LayerNorm)
# --------------------------------------------------------------------------
class _MLP(nn.Module):
    def __init__(self, d_in: int, k: int, h1: int = 1024, h2: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h1),
            nn.LayerNorm(h1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(h2, k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
    patience: int = 12,
    batch: int = 512,
    device: str = "auto",
) -> _MLP:
    torch.manual_seed(seed)
    dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    n = len(X)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    if dev == "cuda":
        Xt, yt = Xt.cuda(), yt.cuda()
    perm = torch.randperm(n, generator=torch.Generator(device="cpu").manual_seed(seed))
    n_val = max(1, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    model = _MLP(X.shape[1], int(y.max()) + 1).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    best_val, best_state, stall = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        order = torch.randperm(
            len(tr_idx), generator=torch.Generator(device="cpu").manual_seed(seed + ep)
        )
        for i in range(0, len(order), batch):
            bidx = tr_idx[order[i : i + batch]]
            opt.zero_grad(set_to_none=True)
            out = model(Xt[bidx])
            loss = lossf(out, yt[bidx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = lossf(model(Xt[val_idx]), yt[val_idx]).item()
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stall = 0
        else:
            stall += 1
            if stall >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_mlp(model: _MLP, X: np.ndarray) -> np.ndarray:
    Xt = torch.from_numpy(X)
    if torch.cuda.is_available():
        Xt = Xt.cuda()
    model.eval()
    with torch.no_grad():
        pred = model(Xt).argmax(dim=1)
    return pred.cpu().numpy().astype(np.int64)


# --------------------------------------------------------------------------
# Probe 4 — k-NN under the complex Hermitian distance (directive formula)
# --------------------------------------------------------------------------
def knn_predict(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int = 1
) -> np.ndarray:
    """d(Psi_i, Psi_j) = 1 - |(1/D) <Psi_i, Psi_j>|  (complex Hermitian)."""
    Xtr_t = torch.from_numpy(Xtr)
    Xte_t = torch.from_numpy(Xte)
    if torch.cuda.is_available():
        Xtr_t, Xte_t = Xtr_t.cuda(), Xte_t.cuda()
    # batch the test side to bound memory
    preds = []
    bs = 128
    D = Xtr.shape[1]
    ytr_t = torch.from_numpy(ytr)
    if torch.cuda.is_available():
        ytr_t = ytr_t.cuda()
    with torch.no_grad():
        for i in range(0, len(Xte), bs):
            block = Xte_t[i : i + bs]
            sim = (block @ Xtr_t.conj().T).abs() / D  # [bs, Ntr]
            dist = 1.0 - sim
            vals, idx = dist.topk(k, dim=1, largest=False)
            neighbor = ytr_t[idx]  # [bs, k]
            pred = torch.mode(neighbor, dim=1).values
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds).astype(np.int64)


# --------------------------------------------------------------------------
# Temporal-difference arm (G4)
# --------------------------------------------------------------------------
def td_delta(
    X: np.ndarray, env_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-env contiguous deltas. Returns (delta, valid_pair_mask).

    delta[t] = X[t+1] * conj(X[t]) elementwise for t = 0..n-2; a pair is
    valid only when env_ids[t+1] == env_ids[t] (never across env
    boundaries). The last row has no successor and is never paired.
    """
    n = len(X)
    if n < 2:
        return np.zeros((0, X.shape[1]), dtype=X.dtype), np.zeros(0, dtype=bool)
    valid = env_ids[1:] == env_ids[:-1]
    delta = X[1:] * np.conj(X[:-1])
    return delta, valid


# --------------------------------------------------------------------------
# Gate evaluation + verdict
# --------------------------------------------------------------------------
@dataclass
class ProbeResult:
    name: str
    train_acc: float | None
    cv_acc: float
    per_fold: list[float] = field(default_factory=list)
    n_test: int = 0


def run_probe_cv(
    Xfeat: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    probe: str,
    seed: int = 0,
    device: str = "auto",
) -> ProbeResult:
    """Run one probe across folds. Returns CV mean + per-fold accuracies."""
    Y_onehot = np.eye(N_CLASSES, dtype=np.float32)[y]
    per_fold: list[float] = []
    train_accs: list[float] = []
    n_test = 0
    for fi, (tr, te) in enumerate(folds):
        if probe == "ls":
            W = fit_minnorm_ls(Xfeat[tr], Y_onehot[tr])
            pred_tr = predict_ls(Xfeat[tr], W)
            pred_te = predict_ls(Xfeat[te], W)
        elif probe == "logistic":
            Wb = fit_logistic(Xfeat[tr], y[tr], lam=1e-3, seed=seed + fi)
            pred_tr = predict_logistic(Xfeat[tr], Wb)
            pred_te = predict_logistic(Xfeat[te], Wb)
        elif probe == "mlp":
            model = fit_mlp(Xfeat[tr], y[tr], seed=seed + fi, device=device)
            pred_tr = predict_mlp(model, Xfeat[tr])
            pred_te = predict_mlp(model, Xfeat[te])
        elif probe == "knn1":
            pred_tr = knn_predict(Xfeat[tr], y[tr], Xfeat[tr], k=1)
            pred_te = knn_predict(Xfeat[tr], y[tr], Xfeat[te], k=1)
        elif probe == "knn3":
            pred_tr = knn_predict(Xfeat[tr], y[tr], Xfeat[tr], k=3)
            pred_te = knn_predict(Xfeat[tr], y[tr], Xfeat[te], k=3)
        else:
            raise ValueError(f"unknown probe {probe}")
        train_accs.append(float((pred_tr == y[tr]).mean()))
        per_fold.append(float((pred_te == y[te]).mean()))
        n_test += len(te)
    # k=1 train accuracy is vacuous (self-hit = 1.0); report None for G1.
    train_acc = None if probe == "knn1" else float(np.mean(train_accs))
    return ProbeResult(
        name=probe,
        train_acc=None if probe.startswith("knn") else train_acc,
        cv_acc=float(np.mean(per_fold)),
        per_fold=per_fold,
        n_test=n_test,
    )


def evaluate(
    npz_path: str,
    jsonl_path: str,
    seed: int = 20260902,
    device: str = "auto",
    quick: bool = False,
) -> dict:
    """Full F8 evaluation. Returns the receipt dict (JSON-serializable)."""
    require_f8_enabled()
    t0 = time.time()
    npz_sha = sha256_file(npz_path)
    jsonl_sha = sha256_file(jsonl_path)
    bank = load_bank(npz_path, jsonl_path)
    psi = bank["psi"]
    y = bank["y"]  # 0..6
    env_id = bank["env_ids"]
    env_names = bank["env_names"]
    n, d = psi.shape
    Xfeat = features(psi)

    maj = majority_baseline(y)
    folds = stratified_folds(y, n_folds=10, seed=seed)
    probes = ["ls", "logistic", "knn1", "knn3"]
    if not quick:
        probes = ["ls", "logistic", "mlp", "knn1", "knn3"]

    results: dict[str, ProbeResult] = {}
    for p in probes:
        results[p] = run_probe_cv(Xfeat, y, folds, p, seed=seed, device=device)

    # G1: in-sample train accuracy (parametric probes only)
    g1_probes = [p for p in ("ls", "logistic", "mlp") if p in results]
    g1_train = {p: results[p].train_acc for p in g1_probes}
    g1_max = max(v for v in g1_train.values() if v is not None)

    cv_accs = {p: results[p].cv_acc for p in results}
    acc_max = max(cv_accs.values())
    g3_margin = acc_max - maj
    g2_ok = acc_max >= 0.60
    g3_ok = g3_margin >= 0.25
    g1_ok = g1_max >= 0.95

    # G4: temporal-difference arm (paired rows only)
    delta, valid = td_delta(psi, env_id)
    y_delta = y[1:][valid]
    X_delta_feat = features(delta[valid])
    X_static_feat = Xfeat[1:][valid]
    td_results: dict[str, ProbeResult] = {}
    for p in ("ls", "knn1"):
        folds_delta = stratified_folds(y_delta, n_folds=10, seed=seed + 1)
        td_results[f"td_{p}"] = run_probe_cv(
            X_delta_feat, y_delta, folds_delta, p, seed=seed + 1, device=device
        )
        td_results[f"static_{p}"] = run_probe_cv(
            X_static_feat, y_delta, folds_delta, p, seed=seed + 1, device=device
        )
    acc_td = max(td_results[f"td_{p}"].cv_acc for p in ("ls", "knn1"))
    acc_static = max(td_results[f"static_{p}"].cv_acc for p in ("ls", "knn1"))
    g4_delta = acc_td - acc_static
    g4_ok = g4_delta >= 0.20

    # Trivial-env diagnostic (lp85/ft09 exclusion) — env ids are positional;
    # identify envs with label entropy ~ 0 from the label marginal.
    env_labels = {}
    for e in np.unique(env_id):
        mask = env_id == e
        vals, counts = np.unique(y[mask], return_counts=True)
        p = counts / counts.sum()
        h = float(-(p * np.log2(p)).sum())
        env_labels[int(e)] = {"n": int(mask.sum()), "H": h, "mode_frac": float(counts.max() / counts.sum())}
    trivial_envs = [e for e, v in env_labels.items() if v["H"] < 1e-6]
    nontrivial_mask = ~np.isin(env_id, trivial_envs)
    folds_nt = stratified_folds(y[nontrivial_mask], n_folds=10, seed=seed + 2)
    nt_accs = {}
    for p in ("ls", "knn1"):
        r = run_probe_cv(Xfeat[nontrivial_mask], y[nontrivial_mask], folds_nt, p, seed=seed + 2, device=device)
        nt_accs[p] = r.cv_acc
    nt_acc_max = max(nt_accs.values())

    # Verdict ternary (spec section 5)
    if acc_max <= maj + 0.05:
        verdict = "F8_PROVEN_NO_ACTION_SIGNAL"
    elif acc_max >= 0.75:
        verdict = "F8_DECODABLE_SIGNAL_EXISTS"
    else:
        verdict = "F8_INDETERMINATE"

    receipt = {
        "schema": "f8-decodability-probe.v1",
        "npz_path": npz_path,
        "jsonl_path": jsonl_path,
        "npz_sha256": npz_sha,
        "jsonl_sha256": jsonl_sha,
        "bank_shape": [n, d],
        "bank_is_complex": False,
        "n_envs": int(len(np.unique(env_id))),
        "env_names": env_names,
        "majority_baseline": maj,
        "g1_train_acc_max": g1_max,
        "g1_ok": g1_ok,
        "g2_cv_acc_max": acc_max,
        "g2_ok": g2_ok,
        "g3_margin": g3_margin,
        "g3_ok": g3_ok,
        "g4_td_minus_static": g4_delta,
        "g4_ok": g4_ok,
        "probe_cv": cv_accs,
        "probe_train": {p: results[p].train_acc for p in results},
        "per_fold_cv": {p: results[p].per_fold for p in results},
        "td_arm": {k: v.cv_acc for k, v in td_results.items()},
        "trivial_envs": trivial_envs,
        "env_label_marginals": env_labels,
        "nontrivial_env_acc_max": nt_acc_max,
        "verdict": verdict,
        "seed": seed,
        "device": device,
        "elapsed_s": round(time.time() - t0, 1),
        "commit_sha": os.environ.get("F8_COMMIT_SHA", "unset"),
    }
    return receipt


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Carrier F8 decodability probe")
    ap.add_argument("--npz", required=True, help="path to bank npz")
    ap.add_argument("--jsonl", required=True, help="path to bank jsonl (env meta)")
    ap.add_argument("--out", required=True, help="receipt json path")
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="skip MLP (plumbing smoke)")
    args = ap.parse_args()
    receipt = evaluate(
        args.npz, args.jsonl, seed=args.seed, device=args.device, quick=args.quick
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps({"verdict": receipt["verdict"], "acc_max": receipt["g2_cv_acc_max"]}))
