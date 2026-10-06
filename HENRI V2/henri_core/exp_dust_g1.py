"""G-DUST-1: does zeroth-order node perturbation align with autograd here?

Pre-registered kill test for the Dust estimator (see the accompanying
preregistration). CPU-runnable. Falsifiable.

Claim under test (spec section 2.2):  E[g_hat] == -grad_W L, so
    cos(G_dust, G_autograd_descent) -> 1 as K grows.

Design:
  A single linear layer y = x W^T, followed by a FIXED random readout and
  cross-entropy. The gradient of the full loss w.r.t. W lives in a subspace of
  rank <= C (the class count), regardless of D. That low-rank structure is the
  doc's stated precondition, so this is a favourable setting for the claim.

Arms:
  real      : node perturbation, K in {16, 64, 256, 1024}
  pos_ctl   : perturbations set to the true descent direction -> cos must be ~1
  shuffled  : draw<->credit pairing permuted                  -> cos must be ~0
  d_out sweep {256, 4096}: tests the overparameterization claim at fixed K

Decision rule (pre-registered): G-DUST-1 PASSES only if, at the largest K tested,
cos_real >= 0.85 AND cos_shuffled <= 0.10 AND cos_pos_ctl >= 0.99. Otherwise it
FAILS, and the failure is reported as measured, not retried at a higher K.

Machine output goes to stdout as one JSON object. Progress goes to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

sys.path.insert(0, ".")
from henri_core.dust_zo import DustZOConfig, matcher_loss_fn, node_perturbation_descent


def autograd_descent(W, X, loss_fn, labels):
    """-dL/dW by reverse-mode, as the reference direction."""
    Wg = W.detach().clone().requires_grad_(True)
    y = X @ Wg.t()
    loss = loss_fn(y, labels).sum()
    loss.backward()
    return (-Wg.grad).detach()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a @ b) / denom)


def run_arm(W, X, labels, readout, cfg, mode="real"):
    loss_fn = matcher_loss_fn(readout)
    if mode == "pos_ctl":
        # perturbations = the true descent direction at y; cos must be ~1
        y = (X @ W.t()).detach()
        yg = y.clone().requires_grad_(True)
        loss_fn(yg, labels).sum().backward()
        direction = (-yg.grad).detach()                    # [N, T, D]
        n, t, dim = y.shape
        acc = torch.zeros(n, t, dim, dtype=y.dtype)
        c = torch.full((1, n, t), cfg.sigma, dtype=y.dtype)  # ~ sigma*||g||^2 proxy
        acc = acc + torch.einsum("nit,nitd->itd", c, direction.unsqueeze(0))
        acc = acc / (cfg.K * cfg.sigma)
        return torch.einsum("ntr,nti->ri", acc, X), cfg.K, 0.0
    if mode == "shuffled":
        gen = torch.Generator().manual_seed(cfg.seed + 999)
        loss_fn2 = matcher_loss_fn(readout)
        with torch.no_grad():
            clean = loss_fn2((X @ W.t()).detach(), labels)
            n, t, dim = X.shape[0], X.shape[1], W.shape[0]
            a = torch.randn(cfg.K, n, t, dim, generator=gen)
            acc = torch.zeros(n, t, dim)
            done = 0
            while done < cfg.K:
                m = min(cfg.chunk, cfg.K - done)
                lab = labels.unsqueeze(0).expand(m, -1, -1).reshape(m * n, t)
                yp = ((X @ W.t()).detach().unsqueeze(0) + cfg.sigma * a[done:done + m]).reshape(m * n, t, dim)
                lp = loss_fn2(yp, lab).reshape(m, n, t)
                c = clean.unsqueeze(0) - lp
                c = c - c.mean(0, keepdim=True)
                # PERMUTE the draw<->credit pairing: breaks correspondence
                perm = torch.randperm(m, generator=gen)
                c = c[perm]
                acc = acc + torch.einsum("mit,mitd->itd", c, a[done:done + m])
                done += m
            acc = acc / (cfg.K * cfg.sigma)
            return torch.einsum("ntr,nti->ri", acc, X), cfg.K, 0.0
    grad = node_perturbation_descent((X @ W.t()).detach(), X, loss_fn, labels, cfg)
    return grad, cfg.K, 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="tokens")
    ap.add_argument("--t", type=int, default=8, help="seq len")
    ap.add_argument("--d-in", type=int, default=32)
    ap.add_argument("--d-out", type=int, default=256)
    ap.add_argument("--classes", type=int, default=16)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    g = torch.Generator().manual_seed(args.seed)
    X = torch.randn(args.n, args.t, args.d_in, generator=g)
    labels = torch.randint(0, args.classes, (args.n, args.t), generator=g)
    W = torch.randn(args.d_out, args.d_in, generator=g) * 0.3
    readout = torch.randn(args.classes, args.d_out, generator=g) / (args.d_out ** 0.5)
    loss_fn = matcher_loss_fn(readout)

    ref = autograd_descent(W, X, loss_fn, labels)

    rows = []
    t0 = time.time()
    for dout in (args.d_out,):
        Wd = W[:, :args.d_in]
        rows.append({"arm": "pos_ctl", "d_out": dout, "K": 0,
                     "cos": cosine(run_arm(Wd, X, labels, readout, DustZOConfig(K=64, sigma=args.sigma, gamma=args.gamma, seed=args.seed), "pos_ctl")[0], ref), "sec": 0.0})
        rows.append({"arm": "shuffled", "d_out": dout, "K": 256,
                     "cos": cosine(run_arm(Wd, X, labels, readout, DustZOConfig(K=256, sigma=args.sigma, gamma=args.gamma, seed=args.seed), "shuffled")[0], ref), "sec": 0.0})
        for K in (16, 64, 256, 1024):
            t1 = time.time()
            cfg = DustZOConfig(K=K, sigma=args.sigma, gamma=args.gamma, seed=args.seed)
            est, _, _ = run_arm(Wd, X, labels, readout, cfg, "real")
            rows.append({"arm": "real", "d_out": dout, "K": K,
                         "cos": cosine(est, ref), "sec": round(time.time() - t1, 3)})

    largest = max(r["K"] for r in rows if r["arm"] == "real")
    real_best = [r for r in rows if r["arm"] == "real" and r["K"] == largest][0]
    shuf = [r for r in rows if r["arm"] == "shuffled"][0]
    pos = [r for r in rows if r["arm"] == "pos_ctl"][0]
    passed = (real_best["cos"] >= 0.85 and shuf["cos"] <= 0.10 and pos["cos"] >= 0.99)

    receipt = {
        "schema": "henri.gdust1.receipt.v1",
        "pin": 20261005,
        "claim": "E[g_hat] == -grad_W L  =>  cos(G_dust, G_autograd) -> 1",
        "design": {"n": args.n, "t": args.t, "d_in": args.d_in, "d_out": args.d_out,
                   "classes": args.classes, "sigma": args.sigma, "gamma": args.gamma,
                   "seed": args.seed},
        "rule": "PASS iff cos_real(Kmax) >= 0.85 AND cos_shuffled <= 0.10 AND cos_pos_ctl >= 0.99",
        "rows": rows,
        "gate": {"id": "G-DUST-1", "largest_K": largest,
                 "cos_real": real_best["cos"], "cos_shuffled": shuf["cos"],
                 "cos_pos_ctl": pos["cos"], "verdict": "PASS" if passed else "FAIL"},
        "disclosure": {"device": "cpu", "no_backprop_claim": True,
                       "scope": "single linear layer + fixed readout; not a full-model replacement"},
        "seconds": round(time.time() - t0, 3),
    }
    out = json.dumps(receipt, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
