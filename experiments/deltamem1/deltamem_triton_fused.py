"""Fused deterministic Triton delta-memory step (C6/C7/C11 path).

Contract: delta_qfhrr_associative_memory.py prereg cc6d7f59 (A1-A10).
- Module-scope Triton kernels (no nested @triton.jit, no per-call recompile).
- SEALED A5 design: TWO-KERNEL fused step.
    K1 readout + reductions (_k_readout_us): each block recomputes
      s = V^T k (R dot products) redundantly, then y = U s, e = v - y,
      partial sums y^2, y*v, v^2. pid 0 persists s to global memory.
      (No Python lists inside the kernel — Triton tracer rejects append.)
    K2 finalize + update (_k_finalize_update): each block reduces the nblk
      partials locally (identical result), computes Sagnac delta + veto flag,
      applies veto (e -> 0, U <- gamma*U only), computes a = Ginv @ s in
      registers (s loaded from memory), updates U; pid 0 increments
      step/veto counters.
- No atomics; fixed-order per-block partial reductions (C11 determinism).
- Device-only step: no .item()/host sync in the timed path (C6).
- bf16 state (C7: 2*4096*8*2 = 131,072 B = 128 KB), fp32 accumulators.
- V frozen seeded orthonormal projection (A9); Ginv = (V^T V)^{-1} computed
  once at init (fp32 device).
- Sagnac veto (C10/A6/A7): null readout exempt; non-null conflict -> e := 0,
  write suppressed (U *= gamma only), device counter incremented.
- Non-contiguous input rejected explicitly (prereg #6).
- _BLOCK=256 -> nblk=16 blocks for D=4096 (RTX 5090 has ~148 SMs; a 4-block
  grid starves the GPU and dominates latency; implementation detail, not a
  contract change).
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

DEFAULT_D = 4096
DEFAULT_R = 8
DEFAULT_GAMMA = 0.985          # USER_SPECIFIED (C5), ratified
DEFAULT_ETA = 0.1              # pre-registered
DEFAULT_V_SEED = 20260824      # A9: frozen seeded projection seed
VETO_THRESHOLD = tl.constexpr(0.35)   # C10 / A6; constexpr for @jit access
_BLOCK = 256


# ---------------------------------------------------------------------------
# K1: readout + error + partial reductions (grid = nblk blocks)
# ---------------------------------------------------------------------------
@triton.jit
def _k_readout_us(U, V, k, v, s, y, e, vy2, yvdot, vv2,
                  D: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    """s = V^T k (recomputed per block, no Python lists); y = U s; e = v-y."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    acc_y = tl.zeros([BLOCK], dtype=tl.float32)
    for j in tl.static_range(R):
        # s_j = sum_d V[d,j] * k[d]  (full D scan; scalar at the end)
        s_acc = tl.zeros([BLOCK], dtype=tl.float32)
        for i in range(0, tl.cdiv(D, BLOCK) * BLOCK, BLOCK):
            o2 = i + tl.arange(0, BLOCK)
            m2 = o2 < D
            vv = tl.load(V + o2 * R + j, mask=m2, other=0.0).to(tl.float32)
            kv = tl.load(k + o2, mask=m2, other=0.0).to(tl.float32)
            s_acc += vv * kv
        s_j = tl.sum(s_acc)
        # y accumulation (in registers)
        u = tl.load(U + offs * R + j, mask=mask, other=0.0).to(tl.float32)
        acc_y += s_j * u
        # persist s (pid 0 only; identical value across blocks otherwise)
        if pid == 0:
            tl.store(s + j, s_j)
    yv = tl.load(v + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y + offs, acc_y, mask=mask)
    tl.store(e + offs, yv - acc_y, mask=mask)
    tl.store(vy2 + pid, tl.sum(acc_y * acc_y))
    tl.store(yvdot + pid, tl.sum(acc_y * yv))
    tl.store(vv2 + pid, tl.sum(yv * yv))


# ---------------------------------------------------------------------------
# K2: veto finalize + counters + update (grid = nblk blocks)
# ---------------------------------------------------------------------------
@triton.jit
def _k_finalize_update(U, Ginv, s, e, vy2, yvdot, vv2, step_cnt, veto_cnt,
                       D: tl.constexpr, R: tl.constexpr, NBLK: tl.constexpr,
                       GAMMA: tl.constexpr, ETA: tl.constexpr,
                       OUT: tl.constexpr, BLOCK: tl.constexpr):
    """Reduce partials -> Sagnac veto; a = Ginv s; U = gamma U + eta e a^T."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    # local reduction of partials (identical across blocks; race-free)
    y2 = 0.0
    yd = 0.0
    v2 = 0.0
    for i in tl.static_range(NBLK):
        y2 += tl.load(vy2 + i)
        yd += tl.load(yvdot + i)
        v2 += tl.load(vv2 + i)
    denom = tl.sqrt(y2 * v2)
    safe = denom > 1e-6
    sagnac = tl.where(safe, 1.0 - 0.5 * (1.0 + yd / denom), 0.0)
    sagnac = tl.minimum(tl.maximum(sagnac, 0.0), 1.0)
    vflag = tl.where((y2 > 1e-16) & (sagnac > VETO_THRESHOLD), 1, 0)
    ev = tl.load(e + offs, mask=mask, other=0.0).to(tl.float32)
    ev = tl.where(vflag.to(tl.float32) > 0.5, 0.0, ev)
    tl.store(e + offs, ev, mask=mask)
    # a = Ginv @ s in registers; U = gamma U + eta e a^T
    for j in tl.static_range(R):
        aj = 0.0
        for i in tl.static_range(R):
            aj += tl.load(Ginv + j * R + i) * tl.load(s + i)
        u = tl.load(U + offs * R + j, mask=mask, other=0.0).to(tl.float32)
        nu = GAMMA * u + ETA * ev * aj
        tl.store(U + offs * R + j, nu.to(OUT), mask=mask)
    # counters: single program increments (race-free, deterministic)
    if pid == 0:
        tl.store(step_cnt, tl.load(step_cnt) + 1)
        tl.store(veto_cnt, tl.load(veto_cnt) + vflag.to(tl.int64))


# ---------------------------------------------------------------------------
# Untimed readout path (telemetry only; not in C6)
# ---------------------------------------------------------------------------
@triton.jit
def _k_vtk(V, k, s, D: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, tl.cdiv(D, BLOCK) * BLOCK, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < D
        v = tl.load(V + offs * R + row, mask=mask, other=0.0).to(tl.float32)
        kv = tl.load(k + offs, mask=mask, other=0.0).to(tl.float32)
        acc += v * kv
    tl.store(s + row, tl.sum(acc))


@triton.jit
def _k_us(U, s, y, D: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < D
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for j in range(R):
        sv = tl.load(s + j).to(tl.float32)
        u = tl.load(U + offs * R + j, mask=mask, other=0.0).to(tl.float32)
        acc += sv * u
    tl.store(y + offs, acc, mask=mask)


class FusedTritonDeltaKernel:
    """Combined readout + delta-update step (A5: two-kernel fused), device-only.

    C6 measures step_once() between CUDA events (A5: 10k iters, mean < 15 us;
    p95 reported). C7: storage_bytes() == 131,072 for bf16 D=4096 r=8 (A1).
    """

    def __init__(self, d: int = DEFAULT_D, r: int = DEFAULT_R,
                 gamma: float = DEFAULT_GAMMA, eta: float = DEFAULT_ETA,
                 v_seed: Optional[int] = DEFAULT_V_SEED,
                 device: Optional[str] = None,
                 dtype: torch.dtype = torch.bfloat16):
        if not (torch.cuda.is_available() and triton):
            raise RuntimeError("Triton/CUDA unavailable")
        self.d, self.r = d, r
        self.gamma, self.eta = float(gamma), float(eta)
        self.device = device or "cuda"
        self.dtype = dtype  # bf16 for C7 path
        self._out_tl = tl.bfloat16 if dtype == torch.bfloat16 else tl.float32
        self.U = torch.zeros(d, r, dtype=dtype, device=self.device)
        self.V = torch.zeros(d, r, dtype=dtype, device=self.device)
        if v_seed is not None:  # A9 frozen seeded orthonormal projection
            g = torch.Generator(device="cpu").manual_seed(v_seed)
            Vr = torch.randn(d, r, generator=g, dtype=torch.float32)
            Q, _ = torch.linalg.qr(Vr)
            sgn = torch.sign(Q[0]); sgn[sgn == 0] = 1.0
            self.V.copy_(Q * sgn)
        Vf = self.V.float()
        G = Vf.T @ Vf                      # (r,r) fp32
        self._Ginv = torch.linalg.inv(G).to(self.device)   # (r,r) fp32
        # device-side counters (1-dim for Triton scalar load/store)
        self._step_dev = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._veto_dev = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._nblk = triton.cdiv(d, _BLOCK)
        # preallocated scratch (no per-step allocation in timed path)
        self._s = torch.empty(r, dtype=torch.float32, device=self.device)
        self._y = torch.empty(d, dtype=torch.float32, device=self.device)
        self._e = torch.empty(d, dtype=torch.float32, device=self.device)
        self._vy2 = torch.empty(self._nblk, dtype=torch.float32, device=self.device)
        self._yvdot = torch.empty(self._nblk, dtype=torch.float32, device=self.device)
        self._vv2 = torch.empty(self._nblk, dtype=torch.float32, device=self.device)

    # -- introspection ---------------------------------------------------
    def storage_bytes(self) -> int:
        return int(self.U.numel() * self.U.element_size() +
                   self.V.numel() * self.V.element_size())

    @property
    def step(self) -> int:
        return int(self._step_dev.item())

    @property
    def veto_count(self) -> int:
        return int(self._veto_dev.item())

    def reset(self) -> None:
        self.U.zero_()                      # V stays frozen (A9)
        self._step_dev.zero_()
        self._veto_dev.zero_()

    # -- combined step (C6 timed path; no host sync; 2 launches, A5) ------
    def step_once(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if (k.ndim != 1 or v.ndim != 1
                or not k.is_contiguous() or not v.is_contiguous()):
            raise ValueError("DeltaMem: non-contiguous input rejected (prereg #6)")
        k = k.reshape(-1).to(torch.float32)
        v = v.reshape(-1).to(torch.float32)
        d, r = self.d, self.r
        _k_readout_us[(self._nblk,)](self.U, self.V, k, v, self._s, self._y,
                                     self._e, self._vy2, self._yvdot, self._vv2,
                                     d, r, BLOCK=_BLOCK)
        _k_finalize_update[(self._nblk,)](self.U, self._Ginv, self._s, self._e,
                                          self._vy2, self._yvdot, self._vv2,
                                          self._step_dev, self._veto_dev,
                                          d, r, self._nblk,
                                          GAMMA=self.gamma, ETA=self.eta,
                                          OUT=self._out_tl, BLOCK=_BLOCK)
        return self._e

    # -- readout only (telemetry; not in timed path) ----------------------
    def readout(self, k: torch.Tensor) -> torch.Tensor:
        k = k.reshape(-1).to(torch.float32).contiguous()
        _k_vtk[(self.r,)](self.V, k, self._s, self.d, self.r, BLOCK=_BLOCK)
        _k_us[(self._nblk,)](self.U, self._s, self._y, self.d, self.r,
                             BLOCK=_BLOCK)
        return self._y

    # -- deterministic state hash (C11; bf16 cast to fp32 for numpy) -------
    def state_hash(self) -> str:
        import hashlib
        h = hashlib.sha256()
        h.update(self.U.detach().float().cpu().numpy().tobytes())
        h.update(self.V.detach().float().cpu().numpy().tobytes())
        h.update(bytes(str(self._step_dev.item()), "ascii"))
        h.update(bytes(str(self._veto_dev.item()), "ascii"))
        return h.hexdigest()
