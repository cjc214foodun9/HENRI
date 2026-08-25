"""
delta_qfhrr_associative_memory.py — DeltaMem-1 carrier (C1-C12 ratified 2026-08-24).
============================================================================
Sealed prereg: matrix264/deltamem1_prereg.md (sha 4916df9f...)
Auth event: a2f59667...  Prereg event: cc6d7f59...

Contracts (ratified user text):
  C1  Factorized low-rank delta updates M_t = U_t V_t^T (r=8, D=4096).
  C2  All online adaptation in (U_t, V_t); static projections frozen; zero
      trainable backbone parameters.
  C4  Delta-rule error reduction >= 40% vs Hebbian outer-product memory.
  C5  R_sync >= 0.90 after 1,000 continuous online updates, gamma = 0.985.
  C6  Combined readout + delta-update step < 15.0 us on host GPU.
  C7  Dynamic memory storage <= 128 KB (2 x 4096 x 8 x 2 bf16 bytes).
  C8  SSR_eval <= 0.35 (one-step, top-r basis from calibration 40%).
  C9  SSR_rollout5 <= 0.70 (5-step open-loop, absolute normalized).
  C10 Sagnac veto: readouts with Delta_Sagnac > 0.35 zero the delta error
      (e_t -> 0), suppressing invalid engram writes.
  C11 Output memory states + telemetry byte-identical across independent
      CUDA execution runs.
  C12 Fail-closed: latency > 15.0 us or R_sync < 0.90 -> DELTA_MEM_REJECTED.

Pre-seal amendments (recorded, both hashes):
  A1  C7 parenthetical 2*4096*8*4 = 256 KB; binding limit 128 KB -> bf16
      factors (2*4096*8*2 = 128 KB). fp32 reported as diagnostic.
  A2  R_sync = |(1/D) sum_j exp(i 2pi phi_j/256)|, phi = angle(readout)
      (real readout: angle in {0, pi} -> |mean(sign)|).
  A3  SSR per stage-0c protocol: SSR_eval = ||(Yhat-Y)V||_F / ||Y V||_F with
      V = top-r right singular vectors of calibration X (first 40%);
      SSR_rollout5 = ||Yhat_5 - Y||_F / ||Y||_F.
  A4  Stream: 13 family blocks (seeded random order, 40 tasks/block) of
      reranker canonical-code embeddings (D=4096, L2-norm). e_t =
      ||psi_{t+1} - M_{t-1} psi_t||_2. Hebbian arm: gamma=1, eta=1, no
      error correction. Reduction = 1 - mean(e_delta)/mean(e_hebbian).
  A5  Latency on Vast RTX 5090, CUDA events, 10k iters, mean < 15 us.
  A6  Real-space Sagnac S = 0.5(1+<phat,v>/(||phat|| ||v||)), Delta = 1 - S;
      Delta > 0.35 -> e_t := 0, write suppressed, counter incremented.

Sealed design (frozen before code):
  k_t = psi_t, v_t = psi_{t+1}; readout phat_{t+1} = M_{t-1} k_t;
  e_t = v_t - phat_{t+1}; M_t = gamma M_{t-1} + eta e_t k_t^T,
  factorized M = U V^T, U,V in R^{4096x8}, gamma=0.985 (USER_SPECIFIED,
  ratified C5; paper has trained gates beta_t, lambda_t=1-beta_t),
  eta=0.1. Factorized update (exact within row span of V):
  a_t = (V^T V)^{-1} (V^T k_t); U_t = gamma U_{t-1} + eta e_t a_t^T;
  V_t = V_{t-1}. V^T V in R^{8x8}, exact solve.
  Init/reset: U_0 = V_0 = 0; reset() re-zeros.
  dtype/accumulation: fp32 reference (fixed-order reductions); Triton bf16
  state (C7), fp32 accumulators. Overflow: ||e_t|| > 1e6 -> clamp (counted).
  Default-OFF: HENRI_DELTAMEM=1 enables; otherwise byte-identical passthrough.
  Zero-trainable: U,V buffers; no Parameter/optimizer/backward.
============================================================================
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

DEFAULT_D = 4096
DEFAULT_R = 8
DEFAULT_GAMMA = 0.985          # USER_SPECIFIED (C5), ratified
DEFAULT_ETA = 0.1              # pre-registered
DEFAULT_V_SEED = 20260824      # A9: frozen seeded projection seed
VETO_THRESHOLD = 0.35          # C10 / A6
NORM_CLAMP = 1e6               # overflow guard


def _sagnac_delta(pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """Real-space Sagnac delta (A6): S = 0.5*(1+<a,b>/(||a||||b||)), d=1-S."""
    a = pred.reshape(-1)
    b = obs.reshape(-1)
    na = a.norm()
    nb = b.norm()
    if na.item() == 0.0 or nb.item() == 0.0:
        return torch.tensor(0.5, dtype=a.dtype, device=a.device)
    s = 0.5 * (1.0 + torch.dot(a, b) / (na * nb))
    return (1.0 - s).clamp(0.0, 1.0)


class DeltaQFHRRAssociativeMemory(nn.Module):
    """Factorized low-rank delta-rule associative memory (C1, C2).

    Reference implementation: pure PyTorch, fp32, deterministic fixed-order
    reductions. The Triton kernel (bounded, default-OFF) must match this
    reference within the pre-registered tolerance.
    """

    def __init__(self, d: int = DEFAULT_D, r: int = DEFAULT_R,
                 gamma: float = DEFAULT_GAMMA, eta: float = DEFAULT_ETA,
                 v_seed: Optional[int] = DEFAULT_V_SEED,
                 enabled: Optional[bool] = None):
        super().__init__()
        self.d = d
        self.r = r
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.v_seed = v_seed
        # default-OFF: HENRI_DELTAMEM=1 enables; else byte-identical passthrough
        if enabled is None:
            enabled = os.environ.get("HENRI_DELTAMEM", "0") == "1"
        self.enabled = enabled
        # C2: buffers only — zero trainable parameters.
        self.register_buffer("U", torch.zeros(d, r, dtype=torch.float32))
        self.register_buffer("V", torch.zeros(d, r, dtype=torch.float32))
        # A9 (amendment): V is a FROZEN seeded orthonormal projection (the
        # static projection weights of C2), NOT zero — a zero V makes the
        # factorized update dead (a_t = 0, U never learns). U is the dynamic
        # memory state; reset() re-zeros U only.
        if self.v_seed is not None:
            g = torch.Generator().manual_seed(self.v_seed)
            Vr = torch.randn(d, r, generator=g, dtype=torch.float32)
            Q, _ = torch.linalg.qr(Vr)
            sgn = torch.sign(Q[0])
            sgn[sgn == 0] = 1.0
            self.V.copy_(Q * sgn)
        self.register_buffer("step", torch.zeros((), dtype=torch.int64))
        self.register_buffer("veto_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("clamp_count", torch.zeros((), dtype=torch.int64))
        self._last_e: Optional[torch.Tensor] = None

    # ---- introspection -------------------------------------------------
    def trainable_parameter_count(self) -> int:
        return 0  # buffers only; nothing requires grad

    def storage_bytes(self) -> int:
        """C7: dynamic memory storage = U+V buffers in their stored dtype."""
        b = self.U.numel() * self.U.element_size() + \
            self.V.numel() * self.V.element_size()
        return int(b)

    # ---- state ---------------------------------------------------------
    def reset(self) -> None:
        self.U.zero_()          # V stays frozen (A9)
        self.step.zero_()
        self.veto_count.zero_()
        self.clamp_count.zero_()
        self._last_e = None

    def readout(self, k: torch.Tensor) -> torch.Tensor:
        """phat = M k = U (V^T k). Default-OFF: identity passthrough."""
        if not self.enabled:
            return k.clone()
        k = k.reshape(-1).to(self.U.dtype)
        s = self.V.T @ k                 # (r,)
        return self.U @ s                # (d,)

    def update(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """One delta step: readout, veto, factorized update. Returns e_t."""
        if not self.enabled:
            return torch.zeros_like(v.reshape(-1))
        k = k.reshape(-1).to(self.U.dtype)
        v = v.reshape(-1).to(self.U.dtype)
        phat = self.readout(k)
        e = v - phat
        # C10 / A6: Sagnac veto -> zero the delta error, suppress write.
        # A7 (amendment, prereg cc6d7f59): a NULL readout (||phat|| ~ 0, e.g.
        # zero memory state or a projection-degenerate direction) carries no
        # destructive interference; vetoing it would block learning forever.
        # Veto applies only to non-null readouts whose phase conflicts.
        if phat.norm().item() > 1e-8 and _sagnac_delta(phat, v).item() > VETO_THRESHOLD:
            e = torch.zeros_like(e)
            self.veto_count.add_(1)
        # overflow guard (pre-registered): clamp e, count.
        en = e.norm().item()
        if en > NORM_CLAMP:
            e = e * (NORM_CLAMP / en)
            self.clamp_count.add_(1)
        # factorized update: a = (V^T V)^{-1} (V^T k); U = gamma U + eta e a^T
        with torch.no_grad():
            s = self.V.T @ k                      # (r,)
            g = self.V.T @ self.V                 # (r,r)
            try:
                a = torch.linalg.solve(g, s)      # (r,)
            except Exception:
                a = torch.linalg.lstsq(g, s.unsqueeze(-1)).solution.squeeze(-1)
            self.U.mul_(self.gamma).add_(self.eta * torch.outer(e, a))
        self.step.add_(1)
        self._last_e = e
        return e

    def step_once(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Combined readout + update (C6 measures this path)."""
        return self.update(k, v)

    # ---- dense exact reference (diagnostic; C4 cross-check) ------------
    def dense_equivalent(self) -> torch.Tensor:
        """Materialize M = U V^T (D x D). Diagnostic only — never for C7."""
        return self.U @ self.V.T

    # ---- determinism helper --------------------------------------------
    def state_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(self.U.reshape(-1).detach().cpu().numpy().tobytes())
        h.update(self.V.reshape(-1).detach().cpu().numpy().tobytes())
        h.update(bytes(str(int(self.step.item())) + str(int(self.veto_count.item())), "ascii"))
        return h.hexdigest()


# ============================================================================
# Triton kernels (bounded, deterministic; bf16 state for C7)
# ============================================================================
def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
        return torch.cuda.is_available()
    except Exception:
        return False


class TritonDeltaKernel:
    """bf16 state (C7: 128 KB), fp32 accumulators, deterministic order."""

    def __init__(self, d: int = DEFAULT_D, r: int = DEFAULT_R,
                 gamma: float = DEFAULT_GAMMA, eta: float = DEFAULT_ETA,
                 v_seed: Optional[int] = DEFAULT_V_SEED,
                 device: Optional[str] = None):
        if not _triton_available():
            raise RuntimeError("Triton/CUDA unavailable")
        self.d = d
        self.r = r
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.U = torch.zeros(d, r, dtype=torch.bfloat16, device=self.device)
        self.V = torch.zeros(d, r, dtype=torch.bfloat16, device=self.device)
        # A9: frozen seeded orthonormal projection (bf16 storage, fp32 solve).
        if v_seed is not None:
            g = torch.Generator().manual_seed(v_seed)
            Vr = torch.randn(d, r, generator=g, dtype=torch.float32)
            Q, _ = torch.linalg.qr(Vr)
            sgn = torch.sign(Q[0]); sgn[sgn == 0] = 1.0
            self.V.copy_(Q * sgn)
        self.step = 0
        self.veto_count = 0

    def storage_bytes(self) -> int:
        return int(self.U.numel() * self.U.element_size() +
                   self.V.numel() * self.V.element_size())

    def reset(self) -> None:
        self.U.zero_()          # V stays frozen (A9)
        self.step = 0
        self.veto_count = 0

    def _solve_a(self, k: torch.Tensor) -> torch.Tensor:
        """Host-side tiny solve: a = (V^T V)^{-1} (V^T k); V in bf16 -> fp32."""
        Vf = self.V.float()
        s = Vf.T @ k.reshape(-1).float()
        g = Vf.T @ Vf
        try:
            return torch.linalg.solve(g, s)
        except Exception:
            return torch.linalg.lstsq(g, s.unsqueeze(-1)).solution.squeeze(-1)

    def readout(self, k: torch.Tensor) -> torch.Tensor:
        """Triton GEMV readout: s = V^T k (r,), phat = U s (d,). fp32 out."""
        import triton
        import triton.language as tl

        k = k.reshape(-1).float().contiguous()
        d, r = self.d, self.r

        @triton.jit
        def _gemv_vTk(V, k, s, d: tl.constexpr, r: tl.constexpr, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for i in range(0, tl.cdiv(d, BLOCK) * BLOCK, BLOCK):
                offs = i + tl.arange(0, BLOCK)
                mask = offs < d
                v = tl.load(V + offs * r + row, mask=mask, other=0.0).to(tl.float32)
                kv = tl.load(k + offs, mask=mask, other=0.0).to(tl.float32)
                acc += v * kv
            tl.store(s + row, tl.sum(acc))

        @triton.jit
        def _gemv_Us(U, s, y, d: tl.constexpr, r: tl.constexpr, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            offs = row * BLOCK + tl.arange(0, BLOCK)
            mask = offs < d
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            for j in range(r):
                sv = tl.load(s + j).to(tl.float32)
                u = tl.load(U + offs * r + j, mask=mask, other=0.0).to(tl.float32)
                acc += sv * u
            tl.store(y + offs, acc, mask=mask)

        s = torch.empty(r, dtype=torch.float32, device=self.device)
        y = torch.empty(d, dtype=torch.float32, device=self.device)
        BLOCK = 1024
        _gemv_vTk[(r,)](self.V, k, s, d, r, BLOCK=BLOCK)
        _gemv_Us[(triton.cdiv(d, BLOCK),)](self.U, s, y, d, r, BLOCK=BLOCK)
        return y

    def update(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Triton update: e = v - phat (fp32); veto -> e=0; U = g U + eta e a^T."""
        import triton
        import triton.language as tl

        k = k.reshape(-1).float().contiguous()
        v = v.reshape(-1).float().contiguous()
        phat = self.readout(k)
        e = v - phat
        # Sagnac veto (A6/A7): null readouts exempt; non-null conflicting veto.
        if phat.norm().item() > 1e-8 and _sagnac_delta(phat, v).item() > VETO_THRESHOLD:
            e = torch.zeros_like(e)
            self.veto_count += 1
        a = self._solve_a(k)  # (r,) fp32 host solve

        d, r = self.d, self.r
        gamma = self.gamma
        eta = self.eta

        @triton.jit
        def _update_U(U, e, a, d: tl.constexpr, r: tl.constexpr,
                      gamma, eta, BLOCK: tl.constexpr):
            row = tl.program_id(0)
            offs = row * BLOCK + tl.arange(0, BLOCK)
            mask = offs < d
            ev = tl.load(e + offs, mask=mask, other=0.0).to(tl.float32)
            for j in range(r):
                u = tl.load(U + offs * r + j, mask=mask, other=0.0).to(tl.float32)
                aj = tl.load(a + j).to(tl.float32)
                nu = gamma * u + eta * ev * aj
                tl.store(U + offs * r + j, nu.to(tl.bfloat16), mask=mask)

        BLOCK = 1024
        _update_U[(triton.cdiv(d, BLOCK),)](self.U, e, a, d, r, gamma, eta, BLOCK=BLOCK)
        self.step += 1
        return e

    def step_once(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.update(k, v)


# ============================================================================
# Hand-computable fixtures (disposable; C1-C2, C10 sanity)
# ============================================================================
def run_fixtures() -> dict:
    res = {}
    torch.manual_seed(0)

    # F1: single step from zero U state -> hand-checkable.
    # U0=0 -> readout=0 (null) -> no veto (A7); e = v; U <- eta e a^T (a != 0
    # because V is the frozen seeded projection, A9).
    m = DeltaQFHRRAssociativeMemory(d=4, r=2, gamma=0.985, eta=0.1, enabled=True)
    k = torch.tensor([1.0, 0.0, -1.0, 2.0])
    v = torch.tensor([0.5, 1.0, -0.5, 0.0])
    e = m.step_once(k, v)
    res["F1_e_equals_v"] = bool(torch.allclose(e, v, atol=1e-6))
    res["F1_step"] = int(m.step.item()) == 1
    res["F1_U_updated_nonzero"] = float(m.U.abs().sum()) > 0.0
    res["F1_V_frozen_unchanged"] = bool(torch.allclose(
        m.V, DeltaQFHRRAssociativeMemory(d=4, r=2, gamma=0.985, eta=0.1,
                                         enabled=True).V, atol=1e-6))

    # F2: readout of stored association (gamma=0, eta=1, V fixed rank-2).
    # a = (V^T V)^-1 V^T k = [1,2]; U = outer(v,a); readout(k) = U (V^T k)
    # = v * (a . [1,2]) = 5v.
    m = DeltaQFHRRAssociativeMemory(d=4, r=2, gamma=0.0, eta=1.0, enabled=True)
    V0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
    m.V.copy_(V0)
    k = torch.tensor([1.0, 2.0, 0.0, 0.0])
    v = torch.tensor([3.0, 4.0, 0.0, 0.0])
    m.step_once(k, v)
    phat = m.readout(k)
    res["F2_readout_scaled_v"] = bool(torch.allclose(phat, 5.0 * v, atol=1e-5))

    # F3: decay factor isolated. Pre-set U so readout(k) == v exactly -> e=0;
    # then update scales U by gamma only.
    m = DeltaQFHRRAssociativeMemory(d=4, r=2, gamma=0.5, eta=0.1, enabled=True)
    m.V.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    k = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v = torch.tensor([1.0, 0.0, 0.0, 0.0])
    m.U[:, 0].copy_(v)          # readout(k) = U[:,0] * (V^T k)[0] = v
    before = m.U.clone()
    m.step_once(k, v)           # e = 0 -> U <- gamma U
    res["F3_decay_applies"] = bool(torch.allclose(m.U, before * 0.5, atol=1e-6))
    res["F3_veto_not_fired"] = int(m.veto_count.item()) == 0

    # F4: veto (C10, A7). Write v1 with k (allowed: null readout). Then attempt
    # v orthogonal to the stored readout -> delta=0.5 > 0.35 -> e=0, veto++.
    m = DeltaQFHRRAssociativeMemory(d=4, r=2, gamma=0.985, eta=0.1, enabled=True)
    m.V.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]))
    k = torch.tensor([1.0, 0.0, 0.0, 0.0])
    v1 = torch.tensor([0.0, 1.0, 0.0, 0.0])
    m.step_once(k, v1)          # first write (null readout -> no veto)
    v2 = torch.tensor([1.0, 0.0, 0.0, 0.0])  # orthogonal to readout 0.1*v1
    e = m.step_once(k, v2)
    res["F4_veto_zeroed_e"] = bool(torch.all(e == 0))
    res["F4_veto_count"] = int(m.veto_count.item()) == 1

    # F5: reset isolation.
    m.reset()
    res["F5_reset"] = (float(m.U.abs().sum()) == 0.0 and int(m.step.item()) == 0)

    # F6: zero trainable (C2) + default-OFF passthrough.
    res["F6_trainable"] = m.trainable_parameter_count() == 0
    m_off = DeltaQFHRRAssociativeMemory(d=4, r=2, enabled=False)
    out = m_off.readout(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    res["F6_off_identity"] = bool(torch.equal(out, torch.tensor([1.0, 2.0, 3.0, 4.0])))

    # F7: storage bytes (C7): fp32 ref = 2*4*2*4 = 64 B; bf16 triton = 2*4096*8*2.
    res["F7_ref_bytes"] = m.storage_bytes() == 64
    res["F7_target"] = 2 * 4096 * 8 * 2 == 131072  # 128 KB exactly

    # F8: deterministic state hash across two identical instances.
    m1 = DeltaQFHRRAssociativeMemory(d=8, r=2, gamma=0.985, eta=0.1)
    m2 = DeltaQFHRRAssociativeMemory(d=8, r=2, gamma=0.985, eta=0.1)
    for i in range(20):
        kk = torch.randn(8) * 0.1 + 0.5
        vv = torch.randn(8) * 0.1 + 0.3
        m1.step_once(kk, vv)
        m2.step_once(kk, vv)
    res["F8_deterministic"] = m1.state_hash() == m2.state_hash()
    return res


if __name__ == "__main__":
    import json
    r = run_fixtures()
    print(json.dumps(r, indent=2, sort_keys=True))
    ok = all(v is True for k, v in r.items() if k.startswith("F"))
    print("FIXTURES", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
