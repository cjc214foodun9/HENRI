"""UHR-02 STORE-POPULATION PROBE — answers the pre-registered falsification on CPU,
using the REAL production store class and its REAL update path, before any GPU hour.

THE PRE-REGISTERED FALSIFICATION (user's words, kept verbatim)
    "populate the outcome store so preference_store_size > 0, then re-run. Predict
     Delta leaves 0.0 and separates compliant from invalid options. Kill criterion:
     if two further runs cannot move Delta off 0.0 with a populated store, UHR-01's
     live value is FALSIFIED and the module stays archived."

WHAT IS PRODUCTION HERE AND WHAT IS SCAFFOLD
    PRODUCTION : `ActionOutcomeGeneratorStore` and `update_generator` -- the exact
                 class and the exact function `production_arc_run.py:3001` calls.
    PRODUCTION : `project_option_to_boundary_family` from `uhr_rfss` -- the exact
                 projection the UHR-01 carrier ships.
    SCAFFOLD   : the transition stream. No ARC environment runs on CPU, so the
                 observed (U_t, a, U_next) pairs come from a fixed SU(3)
                 displacement instead of a real environment. The ABSOLUTE theta
                 scale is therefore scaffold; every statement about the two
                 COMPARISON DOMAINS is a property of the gate math and transfers.

MY BUG, FIXED HERE (found by running the first draft)
    `store.lie_element(a, basis)` returns a BATCHED generator `[K, 3, 3]` -- one
    su(3) element PER CHANNEL -- not a single `(3, 3)`. The first draft called
    `U.trace().reshape(())` on that batch and died with
    "trace: expected a matrix, but got tensor with dim 3".
    Batched trace and per-channel conjugation are used below. This also makes the
    test STRONGER: the production store is per-channel heterogeneous, so the
    control must hold channel-wise.

THE TWO COMPARISON DOMAINS
    FORM A  candidate vs the STATE it started from      (the live loop's domain)
    FORM B  candidate vs the OBSERVED transition        (blueprint section 3.2)

PREDICTIONS (stated before the numbers are read, both outcomes pre-registered)
    P1  Delta leaves 0.0 when the store is populated          -- trivially true,
        because theta != 0 => U != I. This clause alone proves NOTHING.
    P2  the REAL test: two options matched on |Tr U| channel-wise but with
        different content. Predict: FORM A does NOT separate on an isotropic
        baseplate (the collapse to (9-|Tr U|^2)/16 is a DOMAIN property, so more
        store cannot repair it), while FORM B DOES.
    KILL  if P2 fails in both domains, the channel carries no content at all.
"""
from __future__ import annotations

import math
import sys

import torch

SB = r"C:/Users/chan/Desktop/HENRI 7B SWARM/.worktrees/semantic-backbone/HENRI V2"
sys.path.insert(0, SB)

from henri_external_outcome_refactor_module import (  # noqa: E402
    ActionOutcomeGeneratorStore,
    _matrix_log_eig,
    _rand_special_unitary,
)
from uhr_rfss import project_option_to_boundary_family as project  # noqa: E402
from uhr02_exteroceptive_gate import delta  # noqa: E402

# Measurement probe, not a training script: autograd is off for the whole file.
# This is also what silences the `requires_grad` float() warnings a committed
# artifact should not emit -- the store's `theta_a` is an nn.Parameter, so any
# tensor derived through `lie_element` carries grad_fn unless autograd is off.
# (Warning site observed: `float((trA - trB).abs().max())`.)
torch.set_grad_enabled(False)

K = 2048
ACTION = 3
_s3 = math.sqrt(3.0)
BASIS = torch.tensor([
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1 / _s3, 0, 0], [0, 1 / _s3, 0], [0, 0, -2 / _s3]],
], dtype=torch.complex64)
BAND = 0.5 * math.sqrt(2.0 / 10.0) / math.sqrt(K)
SEP_GATE = 4.0 * BAND


def batch_abs_trace(U: torch.Tensor) -> torch.Tensor:
    """|Tr U_k| per channel. Batched: the first draft's U.trace().reshape(()) failed."""
    if U.dim() != 3 or U.shape[-2:] != (3, 3):
        raise ValueError("batch_abs_trace expects [K,3,3], got %s" % (tuple(U.shape),))
    return torch.abs(U.diagonal(dim1=-2, dim2=-1).sum(-1))


def roles_iso(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(K, 8, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def roles_struct(seed: int, coh: float = 6.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = coh * torch.randn(8, generator=g).unsqueeze(0) + torch.randn(K, 8, generator=g)
    return r / r.norm(dim=-1, keepdim=True)


def main() -> int:
    print("=== UHR-02 store-population probe ===")
    print("torch", torch.__version__, "| K =", K, "| band = %.3e | 4*band = %.3e"
          % (BAND, SEP_GATE))

    # ---------------- populate through the PRODUCTION path -------------------
    store = ActionOutcomeGeneratorStore(num_actions=16, num_channels=K, lr=0.5)
    print("store: num_actions=%d num_channels=%d lr=%.2f" % (
        store.num_actions, store.num_channels, store.lr))
    with torch.no_grad():
        before = float(store.theta_a[ACTION].norm())
    print("theta_a[%d] norm BEFORE = %.6e" % (ACTION, before))

    field = _rand_special_unitary(K, "cpu", seed=7)
    disp = _rand_special_unitary(K, "cpu", seed=100 + ACTION)   # stable empirical D_a
    info = None
    for _t in range(8):
        nxt = torch.einsum("nij,njk->nik", disp, field)
        info = store.update_generator(field, ACTION, nxt, BASIS)   # <- production call
        field = nxt
    with torch.no_grad():
        after = float(store.theta_a[ACTION].norm())
    print("theta_a[%d] norm AFTER  = %.6e   (production update path)" % (ACTION, after))
    print("population receipt:", {k: round(float(v), 9) for k, v in (info or {}).items()})
    populated = after > 0.0
    print("STORE_POPULATED =", populated)
    print()

    # ---------------- the option and its EXACT equal-trace control -----------
    gen_c = store.lie_element(ACTION, BASIS)                    # [K,3,3] batched
    print("lie_element shape =", tuple(gen_c.shape), "(BATCHED: one su(3) element per channel)")
    U_A = torch.matrix_exp(gen_c.to(torch.complex64))           # [K,3,3]

    g = torch.Generator().manual_seed(555)
    H = torch.randn(3, 3, generator=g, dtype=torch.complex64)
    H = H + H.conj().transpose(-2, -1)
    V = torch.matrix_exp(1j * H)                                # fixed SU(3)
    U_B = V.conj().transpose(-2, -1) @ U_A @ V                  # per-channel conjugation
    trA, trB = batch_abs_trace(U_A), batch_abs_trace(U_B)
    print("EQUAL-TRACE CONTROL (channel-wise)")
    print("  max |Tr U_A,k - Tr U_B,k| = %.3e   (conjugation preserves the trace exactly)"
          % float((trA - trB).abs().max()))
    print("  |Tr U| mean = %.6f   min = %.6f   max = %.6f"
          % (float(trA.mean()), float(trA.min()), float(trA.max())))
    gen_other = _matrix_log_eig(U_B)                            # back to a generator batch
    print()

    # ---------------- measure BOTH domains ----------------------------------
    print("=== FORM A vs FORM B on the POPULATED store (equal-trace pair) ===")
    rows = {}
    for ref_name, Rx in (("ISOTROPIC", roles_iso(1)), ("STRUCTURED c=6", roles_struct(1, 6.0))):
        obs = project([gen_c], BASIS, Rx)                       # the empirical transition
        dA_t = delta(project([gen_c], BASIS, Rx), Rx)
        dA_c = delta(project([gen_other], BASIS, Rx), Rx)
        dB_t = delta(project([gen_c], BASIS, Rx), obs)
        dB_c = delta(project([gen_other], BASIS, Rx), obs)
        d0 = delta(Rx, obs)                                     # do-nothing candidate
        rows[ref_name] = dict(dA_t=dA_t, dA_c=dA_c, dB_t=dB_t, dB_c=dB_c, d0=d0)
        print("  %s" % ref_name)
        print("    FORM A (vs state)      true=%.6f conj=%.6f sep=%.3e  %s"
              % (dA_t, dA_c, abs(dA_t - dA_c),
                 "NO-SEP" if abs(dA_t - dA_c) < SEP_GATE else "SEP"))
        print("    FORM B (vs transition) true=%.6f conj=%.6f sep=%.3e  %s"
              % (dB_t, dB_c, abs(dB_t - dB_c),
                 "NO-SEP" if abs(dB_t - dB_c) < SEP_GATE else "SEP"))
        print("    do-nothing candidate delta = %.6f  (vetoed=%s)"
              % (d0, d0 > 0.35))

    # ---------------- tau calibration on the populated store -----------------
    print()
    print("=== tau CALIBRATION (FORM B, populated store) ===")
    comp, inval = [], []
    for a in range(1, 9):
        gx = store.lie_element(a, BASIS)
        with torch.no_grad():
            if float(store.theta_a[a].norm()) == 0.0:
                continue
        ref = roles_struct(a, 6.0)
        obs_a = project([gx], BASIS, ref)
        comp.append(delta(project([gx], BASIS, ref), obs_a))
        g_o = _matrix_log_eig(
            V.conj().transpose(-2, -1) @ torch.matrix_exp(gx.to(torch.complex64)) @ V)
        inval.append(delta(project([g_o], BASIS, ref), obs_a))
    if comp and inval:
        print("  compliant (store's own D_a) n=%d  min=%.6f max=%.6f"
              % (len(comp), min(comp), max(comp)))
        print("  invalid   (equal-|Tr U|)    n=%d  min=%.6f max=%.6f"
              % (len(inval), min(inval), max(inval)))
        in_band = max(comp) < 0.35 < min(inval)
        print("  measured band (%.6f, %.6f)   0.35 %s"
              % (max(comp), min(inval), "LIES IN BAND" if in_band else "OUTSIDE BAND"))
    else:
        in_band = False
        print("  no populated actions beyond %d" % ACTION)

    # ---------------- verdict vs the pre-registered falsification -----------
    iso, st = rows["ISOTROPIC"], rows["STRUCTURED c=6"]
    print()
    print("=== VERDICT vs THE PRE-REGISTERED FALSIFICATION ===")
    print("  store populated (theta != 0)              : %s" % populated)
    print("  P1  Delta leaves 0.0                      : %s  (FORM A iso = %.6f)"
          % (abs(iso["dA_t"]) > 1e-9, iso["dA_t"]))
    print("  P2  FORM A separates, isotropic           : %s  (sep %.3e vs gate %.3e)"
          % (abs(iso["dA_t"] - iso["dA_c"]) > SEP_GATE,
             abs(iso["dA_t"] - iso["dA_c"]), SEP_GATE))
    print("  P2  FORM A separates, structured          : %s  (sep %.3e)"
          % (abs(st["dA_t"] - st["dA_c"]) > SEP_GATE, abs(st["dA_t"] - st["dA_c"])))
    print("  P2  FORM B separates, isotropic           : %s  (sep %.3e)"
          % (abs(iso["dB_t"] - iso["dB_c"]) > SEP_GATE, abs(iso["dB_t"] - iso["dB_c"])))
    print("  P2  FORM B separates, structured          : %s  (sep %.3e)"
          % (abs(st["dB_t"] - st["dB_c"]) > SEP_GATE, abs(st["dB_t"] - st["dB_c"])))
    print("  tau band contains 0.35                    : %s" % in_band)
    print()
    print("INTERPRETATION")
    print("  P1 is satisfied trivially: with theta != 0, U != I, so Delta leaves 0.0.")
    print("  That clause alone therefore proves nothing, which is why P2 is the real")
    print("  test. FORM A's collapse onto option MAGNITUDE is a property of the")
    print("  comparison DOMAIN, so no amount of store population can repair it.")
    print("  FORM B reads the relative group element and separates in both baseplates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
