"""UHR-04 AMENDMENT 3 — observational readout (Psi (x) R_s^dag) in MCTS expansion.

THE AMENDMENT
    "Wire the observational readout (Psi (x) R_s^dag) directly into MCTS child
     branch expansion."

WHAT THAT MEANS, CONCRETELY
    `(x)` is circular convolution and `R_s^dag` is the adjoint (conjugate) of the
    state role. In vector-symbolic terms this is the UNBINDING operation: if a
    representation was formed by binding a state role R_s to a value V, then
    correlating with R_s^dag recovers V hat = Psi (x) R_s^dag. The amendment asks
    that this recovery be available AT EXPANSION TIME, so that a child branch is
    evaluated by what its wave actually READS OUT -- not only by a latent cosine
    against a reference wave.

WHY THIS IS THE RIGHT SHAPE OF CHANGE, AND WHAT IT CANNOT DO
    The latent waveform cosine is continuous and sensitive but scale-fragile; the
    discrete match-rate channel is interpretable but blind to magnitude. The
    readout is the bridge: it turns a wave into a recovered filler that can then be
    compared against an EXPECTED filler. It does NOT replace the Sagnac veto, does
    NOT score a task, and does NOT admit a candidate. It is a scoring INPUT.

HONEST LIMIT, MEASURED HERE AND NOT ASSERTED
    With FHRR binding, `unbind(bind(a, b), b) == a` holds EXACTLY only when the
    unbound-by vector is PHASE-ONLY, i.e. |fft(b)| == 1 elementwise. For a general
    real-valued role, unbinding factors through |fft(b)|^2 and recovery is
    approximate. This module therefore does not claim exactness in general:

      * `binding_roundtrip_error(role)` MEASURES how far a role is from phase-only.
      * `readout_is_exact(role)` reports whether the exact path applies.
      * `readout_delta(..., allow_approximate=False)` REFUSES to return a number
        for a non-phase-only role, rather than laundering an approximation as an
        exact readout. The caller passes allow_approximate=True to accept it
        knowingly, and the returned record is labelled.

DEFAULT OFF AT EVERY CALL SITE. Importing this module changes no production path.
No threshold from the Sagnac family (tau_veto = 0.3500) is reused or reinterpreted.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import torch

__all__ = [
    "READOUT_EXACT",
    "READOUT_APPROXIMATE",
    "READOUT_REFUSED_ROLE_NOT_UNITARY",
    "ReadoutResult",
    "binding_roundtrip_error",
    "bind",
    "circular_unbind",
    "phase_only_role",
    "readout_delta",
    "readout_is_exact",
    "verify_readout",
]

READOUT_EXACT = "EXACT_PHASE_ROLE"
READOUT_APPROXIMATE = "APPROXIMATE_REAL_ROLE"
READOUT_REFUSED_ROLE_NOT_UNITARY = "REFUSED_ROLE_NOT_PHASE_ONLY"

_EPS = 1e-12


@dataclass
class ReadoutResult:
    """Outcome of an observational readout at a single MCTS child."""

    delta: float                 # bounded [0, 2], the same convention as Sagnac
    cosine: float                # the underlying bounded similarity in [-1, 1]
    status: str                  # READOUT_EXACT | READOUT_APPROXIMATE | REFUSED...
    role_roundtrip_error: float  # how far the role is from phase-only
    n_rows: int
    reason: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def ok(self) -> bool:
        return self.status in (READOUT_EXACT, READOUT_APPROXIMATE)


# ---------------------------------------------------------------------------
# the readout itself
# ---------------------------------------------------------------------------
def _fft(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fft(x.to(torch.float64), dim=-1)


def circular_unbind(
    psi: torch.Tensor,
    role: torch.Tensor,
    *,
    renorm: bool = True,
) -> torch.Tensor:
    """Psi (x) R_s^dag along the LAST axis, in the frequency domain.

    Circular convolution does NOT preserve norm in the spatial domain (see the
    architecture catalog), and the DFT/IDFT pair carries a 1/n factor, so the
    result is renormalized per row by default. `renorm=False` returns the raw
    correlation for callers who need to measure the norm contract itself.
    """
    if psi.shape != role.shape:
        raise ValueError(
            f"psi and role must match: psi{tuple(psi.shape)} role{tuple(role.shape)}")
    out = torch.fft.ifft(_fft(psi) * torch.conj(_fft(role)), dim=-1).real
    if renorm:
        n = out.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        out = out / n
    return out.to(psi.dtype)


def phase_only_role(n_blocks: int, d_row: int, *, seed: int = 0) -> torch.Tensor:
    """A REAL role whose DFT has unit magnitude everywhere (exact unbinding).

    A real sequence supports |fft| == 1 only when its spectrum is
    Hermitian-symmetric. This builds such a spectrum from random phases, then
    inverse-transforms, so the returned role is real BY CONSTRUCTION and
    `binding_roundtrip_error` on it is at the fp floor.

    Why this helper exists: an earlier test built the role as
    `ifft(exp(1j*randn)).real`, which is NOT Hermitian-symmetric, so its spectrum
    magnitude was arbitrary (measured error 0.9993) and the "exact" arm was
    silently testing the APPROXIMATE path.
    """
    if d_row % 2 != 0:
        raise ValueError(f"d_row must be even for this construction; got {d_row}")
    g = torch.Generator().manual_seed(int(seed))
    half = d_row // 2
    phi = torch.randn(n_blocks, half + 1, dtype=torch.float64, generator=g)
    # DC (k=0) and Nyquist (k=n/2) are SELF-PAIRED under the Hermitian relation
    # X[n-k] = conj(X[k]), so both must be REAL or the inverse transform is not
    # real and `.real` destroys the unit modulus. Measured defect: leaving them as
    # exp(i*phi) gave roundtrip error 0.9979185909032822.
    bits = torch.randint(0, 2, (n_blocks, 2), generator=g).to(torch.float64)
    phi[:, 0] = bits[:, 0] * torch.pi
    phi[:, half] = bits[:, 1] * torch.pi
    X = torch.exp(1j * phi)
    # Hermitian completion: X[d-k] = conj(X[k]) for k = 1..half-1
    tail = torch.conj(X[:, 1:half]).flip(-1)
    Xfull = torch.cat([X, tail], dim=-1)
    assert Xfull.shape[-1] == d_row, f"{Xfull.shape[-1]} != {d_row}"
    # The construction is only correct if the spectrum really has unit modulus.
    # Assert it here so a future edit to this helper cannot silently degrade the
    # "exact" arm into the approximate one (which is what happened twice).
    mag_dev = float((Xfull.abs() - 1.0).abs().max())
    assert mag_dev < 1e-12, f"spectrum not unit modulus: max dev {mag_dev}"
    role = torch.fft.ifft(Xfull, dim=-1).real
    rt = float((_fft(role.to(torch.float64)).abs() - 1.0).abs().max())
    assert rt < 1e-9, f"role is not phase-only after ifft: err {rt}"
    return role.to(torch.float32)


def bind(value: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
    """Circular convolution binding: Psi = value (x) role, in the frequency domain."""
    return torch.fft.ifft(_fft(value) * _fft(role), dim=-1).real.to(value.dtype)


def binding_roundtrip_error(role: torch.Tensor) -> float:
    """How far `role` is from PHASE-ONLY (|fft(role)| == 1), in [0, inf).

    0.0 means unbinding with this role is algebraically exact. A real-valued role
    generally has |fft| far from 1, so its readout is approximate; measuring this
    is what lets the module label the path instead of guessing.
    """
    mag = _fft(role).abs()
    return float((mag - 1.0).abs().max())


def readout_is_exact(role: torch.Tensor, tol: float = 1e-6) -> bool:
    return binding_roundtrip_error(role) <= tol


def readout_delta(
    psi: torch.Tensor,
    role: torch.Tensor,
    value: torch.Tensor,
    *,
    allow_approximate: bool = False,
    exact_tol: float = 1e-6,
    reduce: str = "per_row_mean",
) -> ReadoutResult:
    """Recover the filler from `psi` using `role`, then compare to `value`.

    Returns delta = 1 - cos in [0, 2] -- the SAME bounded convention the Sagnac
    channel uses, so the two are commensurable. The Sagnac THRESHOLD is not
    imported, reused, or reinterpreted here; this is a scalar for ranking.

    Refuses (status = REFUSED_ROLE_NOT_PHASE_ONLY) when the role does not support
    an exact readout and `allow_approximate` is False.
    """
    err = binding_roundtrip_error(role)
    if err > exact_tol and not allow_approximate:
        return ReadoutResult(
            delta=float("nan"), cosine=float("nan"),
            status=READOUT_REFUSED_ROLE_NOT_UNITARY, role_roundtrip_error=err,
            n_rows=int(psi.reshape(-1, psi.shape[-1]).shape[0]),
            reason=(f"role roundtrip error {err:.6g} > tol {exact_tol:g}: unbinding "
                    f"with a non-phase-only role is APPROXIMATE (recovery factors "
                    f"through |fft(role)|^2). Pass allow_approximate=True to accept "
                    f"it knowingly."))

    if reduce not in ("per_row_mean", "global"):
        raise ValueError(f"reduce must be 'per_row_mean' or 'global'; got {reduce!r}")

    # per_row_mean is invariant to the per-row renormalization; global is only
    # meaningful without it (asserted below).
    rec = circular_unbind(psi, role, renorm=(reduce == "per_row_mean"))
    v = value.to(rec.dtype)
    if rec.shape != v.shape:
        raise ValueError(f"readout{tuple(rec.shape)} vs value{tuple(v.shape)}")

    if reduce == "per_row_mean":
        per = torch.nn.functional.cosine_similarity(rec, v, dim=-1)
        cos = float(per.mean().clamp(-1.0, 1.0))
    else:
        if float((rec.norm(dim=-1) - 1.0).abs().max()) < 1e-6:
            raise ValueError(
                "reduce='global' with a per-row renormalized readout mixes "
                "per-row scale factors into one angle; the statistic is then not "
                "the recovery error. Use reduce='per_row_mean', or obtain the "
                "readout with renorm=False. (This conflation produced a measured "
                "0.0321 'error' for an exact readout.)")
        a = rec.reshape(-1)
        b = v.reshape(-1)
        na = a.norm().clamp_min(_EPS)
        nb = b.norm().clamp_min(_EPS)
        cos = float(torch.dot(a, b) / (na * nb))
        cos = max(-1.0, min(1.0, cos))
    return ReadoutResult(
        delta=float(1.0 - cos), cosine=cos,
        status=READOUT_EXACT if err <= exact_tol else READOUT_APPROXIMATE,
        role_roundtrip_error=err,
        n_rows=int(rec.reshape(-1, rec.shape[-1]).shape[0]),
        reason=(f"phase-only role: exact unbinding ({reduce})" if err <= exact_tol
                else f"real-valued role: approximate (err {err:.6g}, {reduce})"))


# ---------------------------------------------------------------------------
# falsification
# ---------------------------------------------------------------------------
def verify_readout(n_blocks: int = 256, d_row: int = 8, seed: int = 0) -> Dict[str, Any]:
    """Falsify this module's claims, with GROUND TRUTH actually present.

    DESIGN FIX (measured): the first version of this function compared
    `unbind(random_noise, role)` against `role`, i.e. there was no bound value to
    recover, so every arm scored cosine ~ 0 and the exact arm reported the
    APPROXIMATE/REFUSED path. A readout test is only meaningful when a value was
    genuinely BOUND into the representation and the question is whether the
    readout recovers it.

    Checks (each can fail)
    ----------------------
    1. EXACT RECOVERY: bind(value, phase_only_role) -> unbind -> recovered == value
       at the fp floor, with status EXACT_PHASE_ROLE.
    2. WRONG-ROLE CONTROL: unbinding with an UNRELATED role must recover value far
       worse. Without this, a readout returning a constant would pass check 1.
    3. ROLE HONESTY: a real-valued (non-phase-only) role must be REFUSED unless the
       caller opts in, and the refusal must carry the measured error.
    4. NORM CONTRACT: raw circular correlation does not preserve norm (the catalog
       contract); the renormalized form has row norms 1 +- eps.
    5. ROW SENSITIVITY: permuting psi's rows changes the readout, so the statistic
       is not row-permutation-blind by construction.
    """
    out: Dict[str, Any] = {"n_blocks": n_blocks, "d_row": d_row}

    # --- 1. EXACT RECOVERY via a genuine phase-only role ---
    role = phase_only_role(n_blocks, d_row, seed=seed)
    out["phase_role_roundtrip_error"] = binding_roundtrip_error(role)
    out["phase_role_is_exact"] = readout_is_exact(role)

    g = torch.Generator().manual_seed(seed + 1)
    value = torch.randn(n_blocks, d_row, generator=g)
    psi = bind(value, role)

    # --- EXACTNESS: renorm=False. The recovered tensor equals `value` up to a
    # GLOBAL scale, so the flattened cosine must be 1.0.
    rec_raw = circular_unbind(psi, role, renorm=False)
    v = value.reshape(-1)
    rr = rec_raw.reshape(-1)
    cos_raw = float(torch.dot(rr, v) / (rr.norm().clamp_min(_EPS) * v.norm().clamp_min(_EPS)))
    out["exact_recovery_cosine_global"] = cos_raw

    # --- NORMALIZED BEHAVIOUR: renorm=True. Rows are unit-norm, so the PER-ROW
    # cosine must be 1.0. A FLATTENED cosine under this mode mixes per-row scale
    # factors and reads < 1 even when recovery is exact -- measuring only that
    # form conflated two different contracts and produced a false failure
    # (0.9679 reported against a 0.999999 requirement).
    rec = circular_unbind(psi, role, renorm=True)
    per_row = torch.nn.functional.cosine_similarity(rec, value, dim=-1)
    out["exact_recovery_cosine_perrow_min"] = float(per_row.min())
    out["exact_recovery_cosine_perrow_mean"] = float(per_row.mean())
    r = rec.reshape(-1)
    cos_rec = float(torch.dot(r, v) / (r.norm().clamp_min(_EPS) * v.norm().clamp_min(_EPS)))
    out["exact_recovery_cosine_flattened_renorm"] = cos_rec
    out["exact_recovery_delta"] = 1.0 - cos_raw
    res = readout_delta(psi, role, value, allow_approximate=True)
    out["exact_roundtrip_status"] = res.status
    out["exact_roundtrip_delta"] = res.delta

    # --- 2. WRONG-ROLE CONTROL (must be measurably worse) ---
    role_wrong = phase_only_role(n_blocks, d_row, seed=seed + 7)
    rec_wrong = circular_unbind(psi, role_wrong)
    w = rec_wrong.reshape(-1)
    cos_wrong = float(torch.dot(w, v) / (w.norm().clamp_min(_EPS) * v.norm().clamp_min(_EPS)))
    out["wrong_role_cosine"] = cos_wrong
    out["wrong_role_is_worse"] = bool(cos_wrong < cos_rec - 1e-3)

    # --- 3. ROLE HONESTY: a real-valued role must be refused ---
    gr = torch.Generator().manual_seed(seed + 2)
    role_real = torch.randn(n_blocks, d_row, generator=gr)
    out["real_role_roundtrip_error"] = binding_roundtrip_error(role_real)
    res_real = readout_delta(psi, role_real, value)
    out["real_role_status"] = res_real.status
    out["real_role_refused"] = (res_real.status == READOUT_REFUSED_ROLE_NOT_UNITARY)
    res_opt = readout_delta(psi, role_real, value, allow_approximate=True)
    out["real_role_optin_status"] = res_opt.status

    # --- 4. NORM CONTRACT ---
    raw = circular_unbind(psi, role, renorm=False)
    out["raw_row_norm_spread"] = float(raw.norm(dim=-1).std())
    rn = circular_unbind(psi, role, renorm=True)
    out["renorm_row_norms_dev"] = float((rn.norm(dim=-1) - 1.0).abs().max())

    # --- 5. ROW SENSITIVITY ---
    perm = torch.randperm(n_blocks, generator=g)
    rec_perm = circular_unbind(psi[perm], role)
    p = rec_perm.reshape(-1)
    cos_perm = float(torch.dot(p, v) / (p.norm().clamp_min(_EPS) * v.norm().clamp_min(_EPS)))
    out["row_permutation_changes_readout"] = bool(abs(cos_perm - cos_rec) > 1e-9)

    out["ALL_READOUT_CHECKS_PASS"] = bool(
        out["phase_role_is_exact"]
        and out["exact_recovery_cosine_global"] > 0.999999
        and out["exact_recovery_cosine_perrow_min"] > 0.999999
        and out["exact_roundtrip_status"] == READOUT_EXACT
        and out["exact_roundtrip_delta"] < 1e-6
        and out["wrong_role_is_worse"]
        and out["real_role_refused"]
        and out["real_role_optin_status"] == READOUT_APPROXIMATE
        and out["renorm_row_norms_dev"] < 1e-6
        and out["row_permutation_changes_readout"])
    return out
