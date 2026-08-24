"""
System-1 Stage-0b — Frozen Deterministic Encoder (CartPole-v1 obs -> live boundary).
====================================================================================
Reference 3 (gpt-5.6-sol) binding: CPU-only, default OFF, ZERO learning, frozen
constants, deterministic across replay AND process restart. No new tensor family:
canonical output is REAL float32 [1,16,384] (d_slot=384), unit-norm sphere per
slot. The elementwise unit-modulus torus exists ONLY as a derived diagnostic
phasor exp(i*theta); it is never passed to any consumer.

Contracts C1-C10 (see vla_stage0b_contract.md).
"""
import ast
import hashlib
import inspect
import io
import json
import os
import subprocess
import sys
import tokenize
from typing import List, Optional

import numpy as np

OBS_DIM = 4
N_SLOTS = 16
D_SLOT = 384
FIXED_SEED = 20260824
ENABLE_VAR = "HENRI_STAGE0B_ENABLE"
ENABLE = lambda: os.environ.get(ENABLE_VAR) == "1"  # read at call time

# ---- frozen deterministic constants (no nn.Parameter; numpy only) ----
_rng = np.random.RandomState(FIXED_SEED)
W1 = (_rng.standard_normal((OBS_DIM, D_SLOT)) * (2.0 / np.sqrt(OBS_DIM))).astype(np.float32)
G = (_rng.uniform(0.5, 1.5, size=(N_SLOTS, D_SLOT))).astype(np.float32)
B = (_rng.uniform(-0.5, 0.5, size=(N_SLOTS, D_SLOT))).astype(np.float32)


def init_hash() -> str:
    """SHA-256 of the frozen constant tensors (deterministic across processes)."""
    return hashlib.sha256(
        W1.tobytes() + G.tobytes() + B.tobytes()).hexdigest()


def encode(obs: np.ndarray) -> np.ndarray:
    """Encode a (4,) float32 CartPole observation.

    Default OFF: without HENRI_STAGE0B_ENABLE=1, returns the input byte-identical
    (C1 bypass). Enabled: returns REAL float32 [1,16,384], per-slot unit-norm
    sphere (||z_s||_2 = 1). Deterministic function of obs only.
    """
    obs = np.asarray(obs, dtype=np.float32)
    if not ENABLE():
        return obs
    base = obs @ W1                      # (384,)
    raw = base[None, :] * G + B          # (16, 384)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    safe = np.where(norms < 1e-12, 1.0, norms)
    z = raw / safe
    z = np.where(norms < 1e-12, 1.0 / np.sqrt(D_SLOT), z)  # zero-norm guard
    return z.reshape(1, N_SLOTS, D_SLOT).astype(np.float32)


def phasor(z: np.ndarray) -> np.ndarray:
    """DIAGNOSTIC ONLY: elementwise unit-modulus torus E = exp(i*pi*z).

    |E_sd| = 1 for every element. Never consumed downstream; used solely to
    test the unit-modulus geometry contract (C8).
    """
    z = np.asarray(z, dtype=np.float32)
    theta = np.clip(z * np.pi, -np.pi, np.pi)
    return np.exp(1j * theta)


def _tokens(src: str) -> List[str]:
    """Code tokens with comments and docstrings removed (for static scans)."""
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING, tokenize.NL,
                        tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
            continue
        out.append(tok.string)
    return out


def _forbidden(fn, words) -> List[str]:
    hit = [w for w in words if any(w in t for t in _tokens(inspect.getsource(fn)))]
    return hit


def verify() -> int:
    """Run C1-C10. Returns 0 on all-pass, 1 on failure."""
    import vla_stage0_gym_wrapper as wrappers
    failures: List[str] = []
    _ENABLED = ENABLE()

    # ---- C1: default OFF -> byte-identical passthrough ----
    obs = np.array([0.02856812, -0.17758702, -0.02690299, 0.25222883], dtype=np.float32)
    passthrough = encode(obs)
    if _ENABLED:
        failures.append("C1 FAIL: env var set at process start; bypass not testable here")
    elif not np.array_equal(passthrough, obs):
        failures.append("C1 FAIL: bypass changed bytes")
    else:
        print("C1 PASS: default OFF byte-identical passthrough")

    # ---- C2: frozen (no Parameter/backward/optimizer/torch in code) ----
    full = _tokens(open(__file__, encoding="utf-8").read())
    hits = [w for w in ("Parameter", "backward", "optimizer", "torch")
            if any(w in t for t in full)]
    if hits:
        failures.append("C2 FAIL: forbidden tokens present: %s" % hits)
    else:
        print("C2 PASS: frozen — no Parameter/backward/optimizer/torch")

    # ---- C3: deterministic init hash across instances AND processes ----
    h1 = init_hash()
    h2 = init_hash()
    code = ("import vla_stage0b_encoder as m; print(m.init_hash())")
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", code], cwd=os.getcwd(),
            stderr=subprocess.STDOUT, timeout=120).decode().strip()
    except Exception as e:
        out = "ERR %s" % e
    if h1 != h2 or out != h1:
        failures.append("C3 FAIL: init hash not deterministic (%s/%s/%s)"
                        % (h1, h2, out))
    else:
        print("C3 PASS: init hash %s identical across instances+process" % h1[:16])

    # ---- C4: replay determinism (same obs -> byte-identical, incl. process) ----
    os.environ[ENABLE_VAR] = "1"
    z1 = encode(obs)
    z2 = encode(obs)
    code4 = ("import vla_stage0b_encoder as m; print(m.encode("
             + repr(obs.tolist()) + ").tobytes().hex())")
    try:
        z3 = bytes.fromhex(subprocess.check_output(
            [sys.executable, "-c", code4], cwd=os.getcwd(),
            env=dict(os.environ, HENRI_STAGE0B_ENABLE="1"),
            stderr=subprocess.STDOUT, timeout=120).decode().strip())
    except Exception as e:
        z3 = b"ERR %s" % e
    if z1.tobytes() != z2.tobytes() or z1.tobytes() != z3:
        failures.append("C4 FAIL: replay/process encoding mismatch")
    else:
        print("C4 PASS: byte-identical encoding across replay + process")

    # ---- C5/C6/C10: real CartPole episodes (seeds 4242, 90909) ----
    all_obs = []
    for seed in (4242, 90909):
        w = wrappers.Stage0GymWrapper(seed=seed)
        r = w.reset()
        all_obs.append(np.asarray(r["observation"], dtype=np.float32))
        for i in range(30):
            try:
                w.step(0 if i % 2 == 0 else 1)
            except RuntimeError:
                break
        for t in w.transitions:
            all_obs.append(np.asarray(t["obs_t"], dtype=np.float32))
    # Deduplicate identical raw observations BEFORE distance metrics.
    # Chain continuity (record[t].obs_next == record[t+1].obs_t) means every
    # internal state appears twice in the trace; identical input -> identical
    # encoding is C4 determinism, NOT collapse. Sensitivity/collisions must be
    # measured over DISTINCT inputs only.
    seen: set = set()
    uniq_obs: List[np.ndarray] = []
    for o in all_obs:
        b = o.tobytes()
        if b not in seen:
            seen.add(b)
            uniq_obs.append(o)
    Z = np.stack([encode(o) for o in uniq_obs])          # (n,1,16,384)
    Zf = Z.reshape(Z.shape[0], N_SLOTS, D_SLOT)          # (n,16,384)
    n = Zf.shape[0]

    # C5 sensitivity
    flat = Zf.reshape(n, -1)
    d2 = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    pair_d = d2[iu]
    frac = float(np.mean(pair_d > 1e-3))
    if frac < 0.90 or pair_d.min() <= 0.0:
        failures.append("C5 FAIL: sensitivity frac %.4f min %.2e"
                        % (frac, pair_d.min()))
    else:
        print("C5 PASS: sensitivity frac %.4f min %.2e" % (frac, pair_d.min()))

    # C6 non-collapse (SVD rank + per-slot std)
    stack = Zf.reshape(n * N_SLOTS, D_SLOT)
    s = np.linalg.svd(stack, compute_uv=False)
    rank = int(np.sum(s > 1e-6))
    slot_std = Zf.std(axis=0).min()
    if rank < 2 or slot_std <= 1e-6:
        failures.append("C6 FAIL: rank %d slot_std %.2e" % (rank, slot_std))
    else:
        print("C6 PASS: SVD rank %d, min per-slot std %.2e" % (rank, slot_std))

    # C7 finite/shape/dtype
    ok7 = (np.isfinite(Zf).all() and Z.shape == (n, 1, 16, 384)
           and Z.dtype == np.float32)
    if not ok7:
        failures.append("C7 FAIL: finite/shape/dtype %s %s %s"
                        % (np.isfinite(Zf).all(), Z.shape, Z.dtype))
    else:
        print("C7 PASS: finite, shape (n,1,16,384), dtype float32")

    # C8 geometry: sphere error + torus modulus error
    norms = np.linalg.norm(Zf, axis=2)                   # (n,16)
    sphere_err = float(np.max(np.abs(norms - 1.0)))
    E = phasor(Zf)
    mod_err = float(np.max(np.abs(np.abs(E) - 1.0)))
    if sphere_err > 1e-6 or mod_err > 1e-6:
        failures.append("C8 FAIL: sphere_err %.2e mod_err %.2e"
                        % (sphere_err, mod_err))
    else:
        print("C8 PASS: sphere_err %.2e torus_mod_err %.2e"
              % (sphere_err, mod_err))

    # C9 no env/learner in encode+phasor
    hits9 = _forbidden(encode, ("env", "gym", "backward", "optimizer", "policy"))
    hits9 += _forbidden(phasor, ("env", "gym", "backward", "optimizer", "policy"))
    if hits9:
        failures.append("C9 FAIL: %s" % hits9)
    else:
        print("C9 PASS: encode/phasor have no env/learner/policy access")

    # C10 diagnostics
    collisions = int(np.sum(pair_d < 1e-6))
    telemetry = {
        "init_hash": h1, "n_obs": int(n), "seeds": [4242, 90909],
        "sensitivity_frac_gt_1e-3": frac, "min_pairwise_l2": float(pair_d.min()),
        "mean_pairwise_l2": float(pair_d.mean()),
        "svd_top5": [float(x) for x in s[:5]], "collisions": collisions,
        "max_sphere_err": sphere_err, "max_torus_mod_err": mod_err,
        "min_slot_std": float(slot_std),
    }
    with open("vla_stage0b_telemetry.json", "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=1)
    print("C10: n=%d mean_pair_l2=%.4f svd_top5=%s collisions=%d"
          % (n, telemetry["mean_pairwise_l2"],
             [round(x, 2) for x in telemetry["svd_top5"]], collisions))

    if failures:
        print("\n".join(failures))
        return 1
    print("ALL STAGE-0B CONTRACTS C1-C10 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(verify())
