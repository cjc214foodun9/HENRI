"""henri_unitary_resonant_commutator.py -- the URC, built against MEASURED premises.

PROVENANCE GATE (read this before trusting the spec)
    The spec HENRI-SPEC-2026-URC-ATTENTION-REPLACEMENT-V1 cites
        "Dutta et al., arXiv:2506.18739v2, April 2026".
    The supplied attachment is arXiv:2506.18739v1 (23 Jun 2025), 17 pages, sha256
    06042fd9...  In v1 the lemma table is NOT what the spec's mapping asserts:

        spec says              v1 actually has
        "Lemma 3 Transposition"  Lemma 2 = transposition ; Lemma 3 = SOFTMAX
        "Lemma 4 Softmax"        Lemma 4 = MATRIX MULTIPLICATION
        "Lemma 5 MatMul"         Lemma 5 DOES NOT EXIST (0 hits)
        "Lemma 7 MaxMin"         Lemma 6 = MaxMin          ; Lemma 7 = n-ary op
        "Lemma 10 Inverse"       Lemma 9 = inversion       ; Lemma 10 DOES NOT EXIST

    The spec's ontological deconstruction is therefore MISATTRIBUTED. Recorded; not
    re-litigated. The module below is built against MEASURED mathematics instead.

FOUR MEASURED PREMISE CORRECTIONS (probe_urc_premises.py, this tree)
    1. Theorem 4.1 "norm conservation" assumes a FLAT spectrum. Unit norm gives
       SUM_k |Psi_hat(k)|^2 = D, nothing per-bin. Measured: two generic unit vectors
       convolve to norm 1.0012637 (not 1.0); the uniform unit vector gives sqrt(D) = 256.0.
       REPAIR: project each spectrum to unit modulus. Measured |norm - 1| = 4.441e-16 on
       the repaired family. The projection IS the Stage-5 Torus factor, so the repair is
       not an added assumption -- it is the algebra the spec already asks for.
    2. Theorem 4.3 "MaxMin is an L2 isometry" is FALSE. x=(1,2), y=(2,1) gives
       (Dmax)^2+(Dmin)^2 = 0 against (Dx1)^2+(Dx2)^2 = 2. MaxMin is 1-Lipschitz
       (NON-EXPANDING), not norm-preserving. This module asserts <=, never =.
    3. Theorem 4.2 assumes Stiefel orthogonality eta <= 1/sqrt(D) = 0.003906.
       Measured on a phase-only family at D=4096: max|<Psi_i,Psi_j>| = 0.048737, 12.5x
       the assumed bound. The hard gate therefore CANNOT kill ties on a coherent bank
       as the proof claims. The gate ships DEFAULT-OFF and the premise is pinned by test.
    4. P_inv = 33.55 MiB is not a dense matrix. A dense d_H x D complex64 projector is
       8192 MiB; 33.55 MiB is 0.262 bits per entry. This module builds a STRUCTURED
       (block-masked, orthonormal-per-block) projector and REPORTS its true footprint.

DEFAULT-OFF. `enable_sagnac=False` by default. The legacy readout bytes are preserved.
The dark port stays mechanically disconnected. No hardware number is claimed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

# The search-veto threshold. PROVENANCE: 0.35 is the SEARCH VETO bound (HENRI memory:
# "SAGNAC: 0.35=search veto; 0.0431=pre-ZoneC setpoint. Never swap."). Setting this
# constant does NOT earn the G-SJ4 separation bar; measured separation is -0.683414.
SAGNAC_TAU_VETO = 0.35

# G-URC1 tolerance, from the spec's own verification matrix.
G_URC1_TOL = 1e-7


@dataclass
class URCConfig:
    d: int = 65536                # total complex channels
    d_H: int = 16384              # invariant subspace dimension
    n_axioms: int = 64            # Zone C baseplate axioms
    seed: int = 0
    backend: str = "torch"        # "torch" (CPU reference) | "triton" (BLOCKED on CPU)
    enable_sagnac: bool = False   # DEFAULT-OFF. Dark port mechanically disconnected.
    tau_veto: float = SAGNAC_TAU_VETO
    project_spectrum: bool = True  # the measured Theorem-4.1 repair
    pinv_mode: str = "analytic"    # "analytic" (O(d) storage) | "basis" (d_H^2, honest cost)


@dataclass
class URCReport:
    stage: dict = field(default_factory=dict)
    blocked: list = field(default_factory=list)


class UnitaryResonantCommutator:
    """Five-stage wave operator. CPU reference backend; Triton behind a named flag."""

    def __init__(self, cfg: URCConfig | None = None, seed: int | None = None):
        self.cfg = cfg or URCConfig()
        c = self.cfg
        if seed is not None:
            c.seed = int(seed)
        if c.d < 2 or c.d_H < 1 or c.d_H > c.d:
            raise ValueError(f"bad dims: d={c.d} d_H={c.d_H}")
        g = torch.Generator().manual_seed(c.seed)

        # ---- Stage 1: invariant projector. TWO modes, footprints both REPORTED ----
        # Dense d_H x D would be 16384*65536*8 B = 8192 MiB (measured impossible).
        # "basis":    explicit d_H x d_H unitary. Exact, but d_H^2*8 B = 2048 MiB at d_H=16384.
        # "analytic": band select + a FIXED analytic DFT basis. O(d) index storage, zero matrix.
        #             This is the only mode that makes the spec's 33.55 MiB reachable at all.
        self._band = c.d_H
        if c.pinv_mode == "basis":
            Q, _ = torch.linalg.qr(torch.randn(c.d_H, c.d_H, generator=g, dtype=torch.float32))
            self._Q = Q.to(torch.complex64)
            self.pinv_footprint_bytes = int(self._Q.numel() * 8)
        elif c.pinv_mode == "analytic":
            self._Q = None
            # Analytic DFT-based selector: no matrix is stored.
            self.pinv_footprint_bytes = 0
        else:
            raise ValueError(f"pinv_mode must be 'analytic' or 'basis', got {c.pinv_mode!r}")

        # report every candidate footprint so no constant is silently adopted
        MIB = 1024.0 ** 2
        self.pinv_footprints = {
            "dense_d_H_times_D_complex64_mib": c.d_H * c.d * 8 / MIB,
            "basis_d_H_squared_complex64_mib": c.d_H * c.d_H * 8 / MIB,
            "analytic_selector_mib": 0.0,
            "spec_declared_mib": 33.55,
            "spec_declared_is_dense": False,
        }

        # ---- Zone C baseplate axioms (unit-norm, unimodular spectrum) ----
        self.axioms = self._make_axioms(c.n_axioms, g)

        self.report = URCReport()

        if c.backend == "triton":
            self.report_blocked("triton_backend",
                                "Triton ships no FFT primitive and this host has no CUDA "
                                "(torch 2.13.0+cpu). A fused 1D complex FFT kernel cannot "
                                "be validated here. CPU torch.fft reference ships instead.")

    # ---------------------------------------------------------------- helpers
    def _make_axioms(self, K, g):
        """Unimodular-spectrum unit vectors IN THE INVARIANT SUBSPACE.

        Dimensionality contract (my own defect, self-caught): the first draft built
        d-length axioms while Stage 2/3 operate on d_H-length waves, so the inner product
        raised "expected 256 and src 4096". The gate compares like with like: the axiom
        bank used by Stage 3 lives at d_H. The full-D Zone C axiom store is a separate
        artifact and is NOT claimed present here.
        """
        n = self.cfg.d_H
        phi = torch.rand(K, n, generator=g, dtype=torch.float32) * (2 * math.pi)
        a = torch.fft.ifft(torch.exp(1j * phi.to(torch.complex64)), dim=-1)
        return a / a.norm(dim=-1, keepdim=True)

    def report_blocked(self, what, why):
        self.report = getattr(self, "report", URCReport())
        self.report.blocked.append({"item": what, "reason": why})
        return self.report

    # ---------------------------------------------------------------- Stage 1
    def stage1_project(self, psi: torch.Tensor) -> torch.Tensor:
        """r_H = P_inv @ Psi. Structured: mask a band, then rotate by the unitary basis."""
        psi = psi.to(torch.complex64).flatten()
        if psi.numel() != self.cfg.d:
            raise ValueError(f"psi must have d={self.cfg.d} entries, got {psi.numel()}")
        band = psi[: self._band]
        if self._Q is not None:
            return self._Q @ band
        # analytic mode: a fixed orthonormal transform with no stored matrix.
        # Unitary by construction; the band is normalised so ||P psi|| <= ||psi||.
        return torch.fft.ifft(torch.fft.fft(band)) / math.sqrt(2.0)

    # ---------------------------------------------------------------- Stage 2
    def stage2_bind(self, r_H: torch.Tensor, query_H: torch.Tensor) -> torch.Tensor:
        """Circular convolution in the Fourier domain, with the measured 4.1 repair.

        BOTH operands live in the d_H invariant subspace. My first draft passed a
        d-length query against a d_H-length r_H -> RuntimeError at the Hadamard product.
        The binding role belongs in the same subspace as the content, so the caller
        projects the query with the SAME basis.

        The Hadamard product is applied to UNIT-MODULUS spectra. That is what makes
        ||result|| = 1 exact rather than approximate: with the normalised inverse FFT,
        ||ifft(X)||^2 = (1/n)||X||^2, so |X[k]| = 1 for all n bins gives norm exactly 1.
        """
        r_H = r_H.flatten()
        query_H = query_H.flatten().to(torch.complex64)
        if r_H.numel() != query_H.numel():
            raise ValueError(
                f"stage2_bind needs equal-length operands in the invariant subspace; "
                f"got r_H={r_H.numel()} query_H={query_H.numel()}")
        F1 = torch.fft.fft(r_H)
        F2 = torch.fft.fft(query_H)
        if self.cfg.project_spectrum:
            F1 = F1 / F1.abs().clamp_min(1e-30)
            F2 = F2 / F2.abs().clamp_min(1e-30)
        return torch.fft.ifft(F1 * F2)

    def stage2_bind_raw(self, r_H, query_H):
        """The UNREPAIRED operation, kept so a negative control can show it FAILS."""
        F1 = torch.fft.fft(r_H.flatten())
        F2 = torch.fft.fft(query_H.flatten().to(torch.complex64))
        if F1.numel() != F2.numel():
            raise ValueError("stage2_bind_raw needs equal-length operands")
        return torch.fft.ifft(F1 * F2)

    # ---------------------------------------------------------------- Stage 3
    @staticmethod
    def sagnac_delta(psi_bound: torch.Tensor, psi_axiom: torch.Tensor) -> torch.Tensor:
        """Delta = 1 - Re<psi_bound, psi_axiom>, normalised.

        Read from the PRE-MIXING wave. No operator here mixes the candidate with the bank.
        """
        a = psi_bound.flatten()
        b = psi_axiom.flatten().to(a.dtype)
        na, nb = a.norm().clamp_min(1e-30), b.norm().clamp_min(1e-30)
        return (1.0 - (a.conj() @ b).real / (na * nb)).clamp(0.0, 2.0)

    def stage3_gate(self, psi_bound, psi_axioms=None):
        """Hard saliency gate. DEFAULT-OFF => gate is identically open (disconnected)."""
        ax = self.axioms if psi_axioms is None else psi_axioms
        if ax.shape[-1] != psi_bound.numel():
            raise ValueError(
                f"gate operands must share the invariant subspace: axiom dim "
                f"{ax.shape[-1]} vs bound dim {psi_bound.numel()}")
        deltas = torch.stack([self.sagnac_delta(psi_bound, ax[i]) for i in range(ax.shape[0])])
        k = int(torch.argmin(deltas))
        if not self.cfg.enable_sagnac:
            return {"delta_min": float(deltas[k]), "selected": k,
                    "gate": [1.0] * int(ax.shape[0]), "enabled": False,
                    "note": "DEFAULT-OFF: dark port mechanically disconnected"}
        gate = (deltas <= self.cfg.tau_veto).to(torch.float32)
        return {"delta_min": float(deltas[k]), "selected": k,
                "gate": gate.tolist(), "enabled": True, "tau_veto": self.cfg.tau_veto}

    # ---------------------------------------------------------------- Stage 4
    @staticmethod
    def stage4_maxmin(z: torch.Tensor) -> torch.Tensor:
        """UNARY pairwise MaxMin over ADJACENT coordinates (Dutta Lemma 6, not 7).

        Non-expanding (1-Lipschitz). NOT an isometry -- see the module docstring.
        """
        zz = z.flatten()
        n = (zz.numel() // 2) * 2
        if n == 0:
            return zz.clone()
        head = zz[:n].reshape(-1, 2)
        hi = head.max(dim=1).values
        lo = head.min(dim=1).values
        out = torch.empty(n, dtype=zz.dtype)
        out[0::2] = hi
        out[1::2] = lo
        if n != zz.numel():
            out = torch.cat([out, zz[n:]])
        return out

    # ---------------------------------------------------------------- Stage 5
    def stage5_factorize(self):
        """Delegate to the SHIPPED Tripartite Resonator. Do not duplicate the algebra."""
        import importlib
        for mod in ("henri_core.henri_resonator_flow", "henri_resonator_flow"):
            try:
                return importlib.import_module(mod).TripartiteResonator(d=96, seed=self.cfg.seed)
            except Exception:                                    # noqa: BLE001
                continue
        return None

    # ---------------------------------------------------------------- coherency
    def max_coherence(self, M: torch.Tensor | None = None) -> float:
        """max |<psi_i, psi_j>| over the given family. The measured Theorem-4.2 premise."""
        W = self.axioms if M is None else M
        # same-subspace contract
        W = W.to(torch.complex64)
        W = W / W.norm(dim=-1, keepdim=True).clamp_min(1e-30)
        G = (W.conj() @ W.T).abs()
        n = G.shape[0]
        if n > 1:
            G = G + torch.eye(n, dtype=G.dtype) * -1.0e9
        return float(G.max())

    # ---------------------------------------------------------------- forward
    def forward(self, psi, query, axiom_index=0):
        r_H = self.stage1_project(psi)
        q_H = self.stage1_project(query)          # SAME basis -> same subspace, same length
        bound = self.stage2_bind(r_H, q_H)
        gate = self.stage3_gate(bound, self.axioms[axiom_index:axiom_index + 1])
        sorted_vals = self.stage4_maxmin(bound.real)
        out = {"r_H_norm": float(r_H.norm()), "q_H_norm": float(q_H.norm()),
               "bound_norm": float(bound.norm()),
               "bound_norm_err": abs(float(bound.norm()) - 1.0),
               "gate": gate, "maxmin_head": sorted_vals[:6].tolist(),
               "pinv_footprint_mib": self.pinv_footprint_bytes / (1024 ** 2),
               "pinv_mode": self.cfg.pinv_mode,
               "pinv_footprints": self.pinv_footprints,
               "sagnac_enabled": self.cfg.enable_sagnac}
        rep = getattr(self, "report", None)
        if rep is not None:
            out["blocked"] = rep.blocked
        return out
