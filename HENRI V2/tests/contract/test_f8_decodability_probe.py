"""Carrier F8 contract tests — decodability probe primitives (toy scale, CPU).

RED phase: module absent -> collection fails (ModuleNotFoundError).
GREEN phase: C1-C8 pass on toy fixtures with PLANTED structure.

Per docs/spec/f8_decodability_probe_preregistration.md section 8.
"""
import json

import numpy as np
import pytest

from arc_f8_decodability_probe import (  # noqa: F401  (RED: module absent)
    fit_logistic,
    fit_minnorm_ls,
    knn_predict,
    load_bank,
    majority_baseline,
    predict_logistic,
    predict_ls,
    require_f8_enabled,
    stratified_folds,
    td_delta,
)

SEED = 20260902
K = 3  # toy classes


def _planted_linear(n=256, d=64, seed=0):
    """Gaussian clusters with (near-)orthogonal class centers -> separable.

    NOTE: a label y = argmax(X @ W) is NOT what min-norm LS-to-onehot
    recovers — squared-error regression to one-hot targets can disagree with
    the argmax boundary (measured: 0.906 train acc). Cluster-separated
    classes are cleanly separable by both LS-to-onehot and logistic.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((K, d)).astype(np.float32)
    centers = centers - centers.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centers, full_matrices=False)
    centers = (U * 8.0) @ Vt  # orthogonal directions, scale 8
    per = n // K
    remainder = n - per * K
    X = np.vstack(
        [rng.standard_normal((per, d)).astype(np.float32) + centers[c] for c in range(K)]
    )
    y = np.repeat(np.arange(K), per).astype(np.int64)
    if remainder:
        X = np.vstack([X, rng.standard_normal((remainder, d)).astype(np.float32) + centers[0]])
        y = np.concatenate([y, np.zeros(remainder, dtype=np.int64)])
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def _clustered(n=300, d=32, seed=1):
    """3 well-separated clusters -> k-NN must beat majority."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((K, d)) * 8.0
    X, y = [], []
    for c in range(K):
        X.append(rng.standard_normal((n // K, d)) + centers[c])
        y.append(np.full(n // K, c))
    return np.vstack(X).astype(np.float32), np.concatenate(y).astype(np.int64)


class TestC1MinNormLS:
    def test_recovers_planted_linear_labels(self):
        X, y = _planted_linear()
        Y = np.eye(K)[y].astype(np.float32)
        W = fit_minnorm_ls(X, Y)
        pred = predict_ls(X, W)
        assert (pred == y).mean() >= 0.99


class TestC2Logistic:
    def test_separates_separable_set(self):
        X, y = _planted_linear()
        folds = stratified_folds(y, n_folds=5, seed=0)
        accs = []
        for tr, te in folds:
            Wb = fit_logistic(X[tr], y[tr], lam=1e-3, epochs=200, lr=1e-2, seed=0)
            accs.append((predict_logistic(X[te], Wb) == y[te]).mean())
        assert float(np.mean(accs)) >= 0.95


class TestC3KNN:
    def test_knn_beats_majority_on_clusters(self):
        X, y = _clustered()
        folds = stratified_folds(y, n_folds=5, seed=0)
        accs = []
        for tr, te in folds:
            accs.append((knn_predict(X[tr], y[tr], X[te], k=1) == y[te]).mean())
        cv = float(np.mean(accs))
        maj = majority_baseline(y)
        assert cv >= 0.95
        assert cv - maj >= 0.25


class TestC4TemporalDifference:
    def test_td_beats_static_when_label_lives_in_delta(self):
        rng = np.random.default_rng(7)
        n, d = 400, 32
        # Class codes p_c = exp(1j*phi_c) with FULL SUPPORT and unit modulus
        # (constant phase shifts, 120 deg apart), so consecutive products
        # X[t+1]*conj(X[t]) = p_c stay unit-modulus and never collapse.
        # Real-concat code vectors v_c = [cos phi_c, sin phi_c] (x) 1 are
        # 120-deg separated: LS-to-onehot argmax is exact for every row.
        phis = np.array([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0])
        p = np.exp(1j * phis[:, None]).astype(np.complex64)  # [K, 1] -> broadcast
        phases = rng.uniform(0, 2 * np.pi, size=(n, d))
        X = np.exp(1j * phases).astype(np.complex64)
        classes = rng.integers(0, K, size=n - 1)
        # TRUE recursion: each row depends on the PREVIOUS MODIFIED row, so
        # X[t+1]*conj(X[t]) = p[c_t]*|X[t]|^2 = p[c_t] telescopes exactly.
        # (Vectorized X[1:] = p*X[:-1] does NOT telescope — measured: TD
        # features carry cross terms p[c_t]*conj(p[c_{t-1}])*X[t]*conj(X[t-1]),
        # ACC stuck at 0.57.)
        for t in range(n - 1):
            X[t + 1] = (p[classes[t]] * X[t]).astype(np.complex64)
        env_ids = np.zeros(n, dtype=np.int64)
        dX, valid = td_delta(X, env_ids)  # dX[t] = p[classes[t]] exactly
        y_delta = classes[valid]
        X_from = X[1:][valid]
        dXr = np.concatenate([dX[valid].real, dX[valid].imag], axis=1).astype(np.float32)
        Xr = np.concatenate([X_from.real, X_from.imag], axis=1).astype(np.float32)
        Wd = fit_minnorm_ls(dXr, np.eye(K)[y_delta].astype(np.float32))
        acc_td = (predict_ls(dXr, Wd) == y_delta).mean()
        Ws = fit_minnorm_ls(Xr, np.eye(K)[y_delta].astype(np.float32))
        acc_static = (predict_ls(Xr, Ws) == y_delta).mean()
        assert acc_td - acc_static >= 0.20
        assert acc_td >= 0.80


class TestC5DefaultOff:
    def test_requires_flag(self, monkeypatch):
        monkeypatch.delenv("HENRI_F8_PROBE", raising=False)
        with pytest.raises(RuntimeError):
            require_f8_enabled()

    def test_runs_with_flag(self, monkeypatch):
        monkeypatch.setenv("HENRI_F8_PROBE", "1")
        require_f8_enabled()  # must not raise


class TestC6BankLoader:
    def test_validates_schema(self, tmp_path):
        npz = tmp_path / "bank.npz"
        jsonl = tmp_path / "bank.jsonl"
        n, d = 32, 64
        rng = np.random.default_rng(3)
        psi = rng.standard_normal((n, d)).astype(np.float16)
        actions_onehot = np.eye(7, dtype=np.uint8)[rng.integers(0, 7, size=n)]
        np.savez(npz, psi=psi, next_wave=psi, actions_onehot=actions_onehot)
        envs = [f"env{i % 3}" for i in range(n)]
        with open(jsonl, "w") as f:
            for i, e in enumerate(envs):
                f.write(json.dumps({"env": e, "step": i, "action_name": "ACTION1"}) + "\n")
        out = load_bank(str(npz), str(jsonl))
        assert out["psi"].shape == (n, d)
        assert out["psi"].dtype == np.float32  # real, upcast
        assert out["y"].min() >= 0 and out["y"].max() <= 6
        assert out["env_ids"].shape == (n,)
        assert len(out["env_names"]) == 3

    def test_rejects_row_mismatch(self, tmp_path):
        npz = tmp_path / "bank_bad.npz"
        jsonl = tmp_path / "bank_bad.jsonl"
        n, d = 8, 16
        rng = np.random.default_rng(4)
        psi = rng.standard_normal((n, d)).astype(np.float16)
        actions_onehot = np.eye(7, dtype=np.uint8)[rng.integers(0, 7, size=n)]
        np.savez(npz, psi=psi, actions_onehot=actions_onehot)
        with open(jsonl, "w") as f:
            for i in range(n - 1):  # one row short -> mismatch
                f.write(json.dumps({"env": "e0", "step": i}) + "\n")
        with pytest.raises(ValueError):
            load_bank(str(npz), str(jsonl))


class TestC7FoldDisjointness:
    def test_disjoint_union(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, K, size=300).astype(np.int64)
        folds = stratified_folds(y, n_folds=10, seed=0)
        all_test = np.concatenate([te for _, te in folds])
        assert len(all_test) == 300
        assert len(np.unique(all_test)) == 300  # disjoint, complete
        for tr, te in folds:
            assert len(np.intersect1d(tr, te)) == 0


class TestC8FoldCoverage:
    def test_every_fold_has_two_classes(self):
        rng = np.random.default_rng(6)
        y = rng.integers(0, K, size=300).astype(np.int64)
        folds = stratified_folds(y, n_folds=10, seed=0)
        for _, te in folds:
            assert len(np.unique(y[te])) >= 2
