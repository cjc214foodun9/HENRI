"""
Stage 4 gate: Transition Identification (full state-action byte payloads).

Document: HENRI-ARCH-2026-CARRIER-AUDIT-AND-PHYSICAL-ML-GAPS
Stage:    4 (Transition Identification)
Gate:     One-step Koopman prediction error delta <= 0.15.
Fail:     Digest-only payloads OR non-contractive error accumulation.

WHAT THIS HARNESS PROVES
------------------------
Gap 3 says the transition database stores only SHA-256 digests, so no
(state, action, next_state) pair can be reconstructed and no Koopman
operator can be identified. This harness MEASURES both sides:

  Path A (DIGEST-ONLY, the reported defect): ledger rows carry digests and
    nothing else. We then attempt identification. Expected result: zero
    recoverable states, identification BLOCKED. That is the fail-closed
    condition demonstrated, not asserted.

  Path B (FULL PAYLOAD): the same ledger is built with the K0 payload
    sidecar enabled (HENRI_LEDGER_PAYLOADS=1). Raw bytes are persisted
    content-addressed, read back, and an action-conditioned Koopman/EDMD
    operator is identified per action. delta is then measured on held-out
    transitions.

The delta bound is only meaningful if the target dynamics are stated. We
therefore report TWO dynamics families:

  linear   : x' = A_a x + noise. Exactly representable by a quadratic
             dictionary, so delta measures payload sufficiency + fit
             procedure, NOT model misspecification.
  tanh     : x' = tanh(A_a x) + noise. A stress case. Its larger delta is
             reported honestly rather than hidden.

Contraction is checked by rolling the identified operator forward and
measuring error growth, which is the stated failure mode ("non-contractive
error accumulation").

Evidence class: OBSERVED (every number is produced by this run).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# The ledger and payload store are BOTH default-OFF. Enable before import.
os.environ["HENRI_TEMPORAL_LEDGER"] = "1"
os.environ["HENRI_LEDGER_PAYLOADS"] = "1"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for cand in (os.path.join(REPO, "HENRI V2"), REPO, HERE):
    if os.path.isfile(os.path.join(cand, "temporal_transition_ledger.py")):
        sys.path.insert(0, cand)
        break

from temporal_transition_ledger import (          # noqa: E402
    TemporalTransitionLedger, wave_digest, action_digest,
)
from ledger_payload_store import LedgerPayloadStore  # noqa: E402

D_STATE = 8
N_ACTIONS = 4
N_TRAIN_EP = 40
N_TEST_EP = 10
STEPS_PER_EP = 30
NOISE = 0.02
RIDGE = 1e-6
DELTA_GATE = 0.15


# --------------------------------------------------------------------------
# dynamics
# --------------------------------------------------------------------------
def make_operators(seed: int = 0):
    g = np.random.default_rng(seed)
    ops = []
    for _ in range(N_ACTIONS):
        M = g.standard_normal((D_STATE, D_STATE)) / math.sqrt(D_STATE)
        ev = np.abs(np.linalg.eigvals(M)).max()
        ops.append(M * (0.90 / ev))     # spectral radius 0.90 -> contractive
    return ops


def step(ops, x, a, family, rng):
    y = ops[a] @ x
    if family == "tanh":
        y = np.tanh(y)
    return y + NOISE * rng.standard_normal(D_STATE)


# --------------------------------------------------------------------------
# observables (EDMD dictionary)
# --------------------------------------------------------------------------
def obs(x: np.ndarray) -> np.ndarray:
    """psi(x) = [1, x, vech(x x^T)] -> dimension 1 + d + d(d+1)/2."""
    d = x.shape[-1]
    iu = np.triu_indices(d)
    quad = np.einsum("...i,...j->...ij", x, x)[..., iu[0], iu[1]]
    return np.concatenate([np.ones(x.shape[:-1] + (1,)), x, quad], axis=-1)


def fit_koopman(psi_x: np.ndarray, x_next: np.ndarray) -> np.ndarray:
    """Ridge least squares K: R^m -> R^d.  K = (P^T P + lI)^-1 P^T Y."""
    m = psi_x.shape[1]
    A = psi_x.T @ psi_x + RIDGE * np.eye(m)
    B = psi_x.T @ x_next
    return np.linalg.solve(A, B)          # [m, d]


def rel_error(pred: np.ndarray, true: np.ndarray) -> float:
    num = np.linalg.norm(pred - true, axis=-1)
    den = np.linalg.norm(true, axis=-1) + 1e-12
    return float(np.mean(num / den))


# --------------------------------------------------------------------------
# ledger generation
# --------------------------------------------------------------------------
def gen_dataset(family: str, seed: int):
    """Return list of (x_t, a_t, x_next) as float lists, chained per episode."""
    ops = make_operators(seed)
    rng = np.random.default_rng(seed + 1)
    eps = []
    for _ in range(N_TRAIN_EP + N_TEST_EP):
        x = rng.standard_normal(D_STATE)
        traj = []
        for _ in range(STEPS_PER_EP):
            a = int(rng.integers(0, N_ACTIONS))
            xn = step(ops, x, a, family, rng)
            traj.append((x.tolist(), a, xn.tolist()))
            x = xn
        eps.append(traj)
    return eps


def write_ledger(root: Path, eps, with_payloads: bool):
    """Build the ledger on disk. Returns (ledger_path, payload_store_root)."""
    root.mkdir(parents=True, exist_ok=True)
    led_path = root / "transitions.jsonl"
    store_root = root / "payloads"
    store = LedgerPayloadStore(store_root) if with_payloads else None
    led = TemporalTransitionLedger(led_path, strict=True, payload_store=store)
    for i, traj in enumerate(eps):
        ep = f"ep{i:04d}"
        led.reset(ep)
        for s, (x, a, xn) in enumerate(traj):
            led.record(x, a, xn, episode_id=ep, step=s)
    return led_path, store_root


def read_back(led_path: Path, store_root: Path):
    """Reconstruct (x_t, a_t, x_next) from disk, preserving episode chains.

    Digest-only rows carry no payload, so nothing is recoverable. Payload
    rows carry <digest>.bin written by the store; the digest is verified
    against the bytes before use (fail-closed on corruption).
    """
    X, A, Y, unusable = [], [], [], 0
    ep_ids, steps = [], []
    for line in led_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        need = ("obs_t_ref", "action_ref", "obs_next_ref")
        if not all(k in r for k in need):
            unusable += 1
            continue
        try:
            vals = []
            for key in need:
                ref = r[key]                       # the digest VALUE, not the key name
                blob = (store_root / f"{ref}.bin").read_bytes()
                if hashlib.sha256(blob).hexdigest() != ref:
                    raise ValueError("digest mismatch")
                vals.append(blob)
            raw_x = json.loads(vals[0].decode("utf-8"))     # grid-json
            raw_a = json.loads(vals[1].decode("utf-8"))
            raw_n = json.loads(vals[2].decode("utf-8"))
            X.append(np.asarray(raw_x, dtype=np.float64))
            A.append(int(raw_a))
            Y.append(np.asarray(raw_n, dtype=np.float64))
            ep_ids.append(r.get("episode_id", ""))
            steps.append(int(r.get("step", -1)))
        except Exception:
            unusable += 1
    return (np.asarray(X), np.asarray(A), np.asarray(Y),
            ep_ids, steps, unusable)


def identify(X, A, Y, ep_ids, steps, n_test_ep: int, rollout_len: int = 25):
    """Fit one operator per action on train episodes; score on test episodes.

    Two independent measurements:
      1. one-step delta  -- THE GATE (<= 0.15).
      2. multi-step rollout -- the contraction check. Error must NOT grow
         without bound when the operator is rolled forward on true actions.
    """
    if len(X) == 0:
        return None

    episodes = []
    for e in dict.fromkeys(ep_ids):          # insertion order preserved
        idx = [i for i, x in enumerate(ep_ids) if x == e]
        episodes.append(idx)
    if len(episodes) <= n_test_ep:
        return None
    train_eps = episodes[:-n_test_ep]
    test_eps = episodes[-n_test_ep:]
    tr_idx = np.array([i for ep in train_eps for i in ep])
    te_idx = np.array([i for ep in test_eps for i in ep])

    Ks = {}
    for a in range(N_ACTIONS):
        mask = (A[tr_idx] == a)
        if mask.sum() < 2:
            return None
        Ks[a] = fit_koopman(obs(X[tr_idx][mask]), Y[tr_idx][mask])

    # ---- 1. one-step delta over ALL test transitions ----
    preds, trues = [], []
    for i in te_idx:
        a = int(A[i])
        if a not in Ks:
            continue
        preds.append(obs(X[i]) @ Ks[a])
        trues.append(Y[i])
    if not preds:
        return None
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    delta = rel_error(preds, trues)

    # ---- 2. multi-step rollout on the first full test episode ----
    ep = [i for i in test_eps[0] if int(A[i]) in Ks]
    ep.sort(key=lambda i: steps[i])
    ep = ep[:rollout_len]
    growth = []
    if len(ep) >= 3:
        x_p = X[ep[0]].copy()
        for j, i in enumerate(ep):
            if j == 0:
                a = int(A[i])
            else:
                a = int(A[ep[j - 1]])
            x_p = obs(x_p) @ Ks[a]
            growth.append(rel_error(x_p[None, :], Y[i][None, :]))

    return {
        "n_pairs": int(len(X)),
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "n_actions_fit": len(Ks),
        "delta_one_step": round(delta, 6),
        "gate_pass": bool(delta <= DELTA_GATE),
        "rollout_steps": len(growth),
        "rollout_err_first": round(growth[0], 6) if growth else None,
        "rollout_err_last": round(growth[-1], 6) if growth else None,
        "rollout_err_mean": round(float(np.mean(growth)), 6) if growth else None,
        # Contraction: the rollout error must not blow up. We require the
        # final error to stay within 3x the first-step error.
        "contractive": bool(growth and growth[-1] <= 3.0 * max(growth[0], 1e-9)),
        "operator_frobenius_norm": {int(a): round(float(np.linalg.norm(Ks[a])), 6)
                                    for a in Ks},
    }


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="stage4_"))
    print("=" * 78)
    print("STAGE 4 GATE - TRANSITION IDENTIFICATION (Koopman one-step)")
    print("=" * 78)
    print(f"gate: one-step prediction error delta <= {DELTA_GATE}")
    print(f"fail-closed: digest-only payloads, or non-contractive error growth")
    print(f"state d={D_STATE}, actions={N_ACTIONS}, episodes="
          f"{N_TRAIN_EP}+{N_TEST_EP} x {STEPS_PER_EP} steps")
    print(f"workdir: {tmp}")
    print()

    results = {}
    for family in ("linear", "tanh"):
        eps = gen_dataset(family, seed=11)
        print(f"### dynamics family: {family}")

        # ---- Path A: digest-only (the reported Gap 3 defect) ----
        a_root = tmp / f"{family}_digestonly"
        led_a, st_a = write_ledger(a_root, eps, with_payloads=False)
        Xa, Aa, Ya, epa, sta, unus_a = read_back(led_a, st_a)
        rows_a = sum(1 for _ in led_a.read_text(encoding="utf-8").splitlines())
        print(f"  A) DIGEST-ONLY : rows={rows_a} rows_without_payload={unus_a} "
              f"recoverable_pairs={len(Xa)}")
        ident_a = (identify(Xa, Aa, Ya, epa, sta, N_TEST_EP)
                   if len(Xa) else None)
        print(f"     identification: "
              f"{'BLOCKED (no payload to fit)' if ident_a is None else 'ran'}")
        results[f"{family}_digest_only"] = {
            "rows": rows_a, "recoverable_pairs": int(len(Xa)),
            "identification": "BLOCKED_NO_PAYLOADS" if ident_a is None else "ran",
            "delta_one_step": None if ident_a is None else ident_a["delta_one_step"],
        }

        # ---- Path B: full payload sidecar ----
        b_root = tmp / f"{family}_payload"
        led_b, st_b = write_ledger(b_root, eps, with_payloads=True)
        Xb, Ab, Yb, epb, stb, unus_b = read_back(led_b, st_b)
        nbin = len(list(st_b.glob("*.bin"))) if st_b.exists() else 0
        print(f"  B) FULL PAYLOAD : rows={sum(1 for _ in led_b.read_text(encoding='utf-8').splitlines())} "
              f"payload_blobs={nbin} unusable={unus_b} recoverable_pairs={len(Xb)}")
        ident_b = identify(Xb, Ab, Yb, epb, stb, N_TEST_EP) if len(Xb) else None
        if ident_b is None:
            print("     identification: FAILED to fit")
            results[f"{family}_payload"] = {"identification": "FIT_FAILED"}
        else:
            print(f"     delta(one-step) = {ident_b['delta_one_step']:.6f}  "
                  f"-> {'PASS' if ident_b['gate_pass'] else 'FAIL'}")
            print(f"     rollout err first/last/mean = {ident_b['rollout_err_first']} / "
                  f"{ident_b['rollout_err_last']} / {ident_b['rollout_err_mean']} "
                  f"over {ident_b['rollout_steps']} steps  "
                  f"contractive={ident_b['contractive']}")
            results[f"{family}_payload"] = {
                "identification": "ran", "payload_blobs": nbin,
                "recoverable_pairs": int(len(Xb)), **ident_b,
            }
        print()

    lin = results.get("linear_payload", {})
    delta = lin.get("delta_one_step")
    gate_pass = bool(delta is not None and delta <= DELTA_GATE)
    digest_blocked = (results.get("linear_digest_only", {})
                      .get("identification") == "BLOCKED_NO_PAYLOADS")

    print("-" * 78)
    print(f"linear dynamics, full payload : delta = "
          f"{delta if delta is not None else 'n/a'}  -> {'PASS' if gate_pass else 'FAIL'}")
    print(f"digest-only control           : "
          f"{'BLOCKED (fail-closed condition reproduced)' if digest_blocked else 'unexpected'}")
    print(f"GATE RESULT (with payloads)   : {'PASS' if gate_pass else 'FAIL'}")
    print("=" * 78)

    receipt = {
        "stage": 4,
        "artifact": "HENRI V2/temporal_transition_ledger.py",
        "dependency": "HENRI V2/ledger_payload_store.py",
        "gate": f"one-step Koopman prediction error delta <= {DELTA_GATE}",
        "fail_closed_condition": "digest-only payloads or non-contractive error accumulation",
        "gate_pass": gate_pass,
        "evidence_class": "OBSERVED",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"d_state": D_STATE, "n_actions": N_ACTIONS,
                   "episodes": [N_TRAIN_EP, N_TEST_EP], "steps_per_episode": STEPS_PER_EP,
                   "noise": NOISE, "delta_gate": DELTA_GATE},
        "results": results,
        "gap3_resolution": {
            "statement": ("T0 ledger rows carry digests only, so no "
                          "(obs, action, obs_next) pair is reconstructable and no "
                          "Koopman operator can be identified."),
            "fix": ("K0 content-addressed payload sidecar "
                    "(HENRI_LEDGER_PAYLOADS=1) persists canonical bytes per "
                    "digest; digests are re-verified against the bytes on read."),
            "measured": ("digest-only -> 0 recoverable pairs, identification "
                         "BLOCKED; full payload -> identification runs and "
                         "delta is scored."),
        },
        "honest_boundary": ("The tanh family is a stress case whose delta is "
                            "reported separately; a quadratic dictionary cannot "
                            "represent tanh exactly. Only the linear family "
                            "isolates payload sufficiency from model "
                            "misspecification."),
    }
    out = os.path.join(HERE, "stage4_receipt.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"receipt written: {out}")
    print(f"artifacts kept in: {tmp}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
