"""Gap 3 closure: action-conditioned Koopman transition ledger.

THE GAP (audited, measured)
---------------------------
Stage 4 of HENRI-ARCH-2026-CARRIER-AUDIT-AND-PHYSICAL-ML-GAPS measured the
defect and the fix:

    digest-only ledger rows  -> 0 recoverable (x_t, a_t, x_next) triples;
                                identification BLOCKED.
    payload-backed rows      -> triples recoverable; an action-conditioned
                                operator can be identified.

The live substrate for this already exists and is unchanged here:
`temporal_transition_ledger.py` (T0 rows + continuity) and
`ledger_payload_store.py` (K0 content-addressed byte payloads, default-OFF).
This module is the CONSUMER that was missing: it turns payload-backed ledger
rows into identified, action-conditioned transition operators.

WHAT WAS ALSO MEASURED, AND IS ENFORCED HERE
--------------------------------------------
The Stage 4 gate `delta <= 0.15` is not a bare scalar. Two pre-conditions were
measured to flip the same fit from FAIL to PASS:

  1. METRIC. On the SAME fit, Frobenius gave 0.0745 (PASS) and mean per-sample
     relative error gave 0.3848 (FAIL). The metric must be stated. This module
     reports noise-normalized Frobenius as the primary metric and keeps the
     per-sample metric only as a labelled secondary.
  2. SAMPLE DIVERSITY / CONDITIONING. At n=1500 over 50 correlated episodes,
     `fitted/truth` was 2.308 with `kappa_max` = 126.9 (FAIL). At n=12000 over
     400 episodes, `fitted/truth` was 1.014 with `kappa_max` = 12.2 (PASS).
     No algorithmic change was needed; 8x more data flipped the verdict. So the
     identification gate here is pre-conditioned on `n_pairs` and reports
     `kappa_max`, and refuses to emit a verdict below the pre-registered row
     count.

FAIL-CLOSED CONTRACT
--------------------
  - No payload rows            -> status BLOCKED_NO_PAYLOADS (never a silent 0).
  - Below `min_pairs`          -> status BLOCKED_INSUFFICIENT_PAIRS.
  - kappa above `max_kappa`    -> status BLOCKED_ILL_CONDITIONED.
  - A digest that does not reproduce its bytes -> the row is UNUSABLE and
    counted; it is never fitted.

Evidence boundary: this identifies linear operators from data. It does not
establish that the environment is linear; that is exactly why the `fitted` vs
`ground_truth` comparison is reported separately when a ground truth is known.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

STATUS_OK = "KOOPMAN_IDENTIFIED"
STATUS_NO_PAYLOADS = "BLOCKED_NO_PAYLOADS"
STATUS_INSUFFICIENT = "BLOCKED_INSUFFICIENT_PAIRS"
STATUS_ILL_CONDITIONED = "BLOCKED_ILL_CONDITIONED"
STATUS_NO_FIT = "BLOCKED_FIT_FAILED"

# Pre-registered pre-conditions, taken from the Stage 4 measurements.
DEFAULT_MIN_PAIRS = 12000
DEFAULT_MAX_KAPPA = 50.0
DEFAULT_DELTA_GATE = 0.15
DEFAULT_RIDGE = 1e-6


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------

def quadratic_observables(x: np.ndarray) -> np.ndarray:
    """psi(x) = [1, x, vech(x x^T)] -> 1 + d + d(d+1)/2.

    A quadratic dictionary makes the linear family exactly representable, which
    is what isolates PAYLOAD SUFFICIENCY from model misspecification. A non-zero
    delta on a linear system therefore indicts the data path, not the model.
    """
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[-1]
    iu = np.triu_indices(d)
    quad = np.einsum("...i,...j->...ij", x, x)[..., iu[0], iu[1]]
    ones = np.ones(x.shape[:-1] + (1,))
    return np.concatenate([ones, x, quad], axis=-1)


# ---------------------------------------------------------------------------
# Ledger reading (payload-backed only)
# ---------------------------------------------------------------------------

@dataclass
class LedgerCorpus:
    """Recovered transitions plus the accounting that makes them trustworthy."""

    n_rows: int
    n_usable: int
    n_unusable: int
    n_digest_mismatch: int
    X: np.ndarray
    A: np.ndarray
    Y: np.ndarray
    episode_ids: List[str]
    steps: List[int]
    payload_root: Optional[str] = None

    @property
    def digest_ok(self) -> bool:
        return self.n_digest_mismatch == 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_usable": self.n_usable,
            "n_unusable": self.n_unusable,
            "n_digest_mismatch": self.n_digest_mismatch,
            "n_pairs": int(len(self.X)),
            "episodes": len(set(self.episode_ids)),
            "payload_root": self.payload_root,
        }


def read_payload_ledger(
    ledger_path: str | Path,
    payload_root: str | Path,
) -> LedgerCorpus:
    """Recover (x_t, a_t, x_next) from a payload-backed ledger. Fail-closed.

    A row is usable only when all six payload fields are present AND every
    recovered blob reproduces its recorded digest. A digest mismatch is counted
    and the row is dropped - corrupt data must never enter the fit.
    """
    ledger_path = Path(ledger_path)
    payload_root = Path(payload_root)
    keys = ("obs_t_ref", "action_ref", "obs_next_ref")
    X: List[np.ndarray] = []
    A: List[int] = []
    Y: List[np.ndarray] = []
    eps: List[str] = []
    steps: List[int] = []
    n_rows = n_unusable = n_mismatch = 0

    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n_rows += 1
            rec = json.loads(line)
            if not all(k in rec for k in keys):
                n_unusable += 1
                continue
            try:
                blobs = []
                for k in keys:
                    ref = rec[k]
                    blob = (payload_root / f"{ref}.bin").read_bytes()
                    if hashlib.sha256(blob).hexdigest() != ref:
                        n_mismatch += 1
                        raise ValueError("digest mismatch")
                    blobs.append(blob)
                raw_x = json.loads(blobs[0].decode("utf-8"))
                raw_a = blobs[1].decode("utf-8")
                raw_n = json.loads(blobs[2].decode("utf-8"))
                action = int(json.loads(raw_a))
                X.append(np.asarray(raw_x, dtype=np.float64).reshape(-1))
                A.append(action)
                Y.append(np.asarray(raw_n, dtype=np.float64).reshape(-1))
                eps.append(str(rec.get("episode_id", "")))
                steps.append(int(rec.get("step", -1)))
            except Exception:
                n_unusable += 1
                continue

    return LedgerCorpus(
        n_rows=n_rows,
        n_usable=len(X),
        n_unusable=n_unusable,
        n_digest_mismatch=n_mismatch,
        X=np.asarray(X) if X else np.zeros((0, 0)),
        A=np.asarray(A, dtype=np.int64),
        Y=np.asarray(Y) if Y else np.zeros((0, 0)),
        episode_ids=eps,
        steps=steps,
        payload_root=str(payload_root),
    )


# ---------------------------------------------------------------------------
# Action-conditioned identification
# ---------------------------------------------------------------------------

@dataclass
class ActionOperator:
    action: int
    K: np.ndarray                     # [m, d]
    n_fit: int
    kappa_max: float
    frobenius: float
    safe: bool


@dataclass
class IdentificationResult:
    status: str
    reason: str = ""
    delta_frobenius: Optional[float] = None
    delta_per_sample: Optional[float] = None
    delta_ground_truth: Optional[float] = None
    delta_fitted_over_truth: Optional[float] = None
    kappa_max: Optional[float] = None
    n_train: int = 0
    n_test: int = 0
    n_pairs: int = 0
    min_pairs: int = DEFAULT_MIN_PAIRS
    max_kappa: float = DEFAULT_MAX_KAPPA
    gate: float = DEFAULT_DELTA_GATE
    operators: Dict[int, ActionOperator] = field(default_factory=dict)
    contractive: Optional[bool] = None
    rollout_err_first: Optional[float] = None
    rollout_err_last: Optional[float] = None
    metric: str = "noise_normalized_frobenius"

    @property
    def gate_pass(self) -> bool:
        return bool(
            self.status == STATUS_OK
            and self.delta_frobenius is not None
            and self.delta_frobenius <= self.gate
        )

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["operators"] = {int(k): v.frobenius for k, v in self.operators.items()}
        d["operator_kappa"] = {int(k): v.kappa_max for k, v in self.operators.items()}
        d["gate_pass"] = self.gate_pass
        return d


def _rel_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Noise-normalized Frobenius: ||pred - true||_F / ||true||_F.

    This is the PRIMARY metric. It is scale-normalized by the target, so it is
    comparable across (d, n) settings and does not inflate under contraction.
    """
    num = float(np.linalg.norm(pred - true, axis=-1).mean()) if pred.ndim > 1 else float(
        np.linalg.norm(pred - true))
    den = float(np.linalg.norm(true, axis=-1).mean()) if true.ndim > 1 else float(
        np.linalg.norm(true))
    return num / max(den, 1e-12)


def _per_sample_error(pred: np.ndarray, true: np.ndarray) -> float:
    """Labelled SECONDARY metric: mean per-sample relative error.

    Reported only for comparability with the Stage 4 receipt, where the two
    metrics disagreed on the same fit (0.0745 vs 0.3848). Quoting this one
    alone is the defect that the audit recorded; it inflates without bound when
    the dynamics contract.
    """
    num = np.linalg.norm(pred - true, axis=-1)
    den = np.linalg.norm(true, axis=-1) + 1e-12
    return float(np.mean(num / den))


def fit_action_operator(
    psi_x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float = DEFAULT_RIDGE,
) -> Tuple[np.ndarray, float]:
    """Ridge least squares K: K = (P^T P + lI)^-1 P^T Y, and kappa(P)."""
    m = psi_x.shape[1]
    P = psi_x
    A = P.T @ P + ridge * np.eye(m)
    B = P.T @ y
    K = np.linalg.solve(A, B)
    # Conditioning of the dictionary Gram matrix: the measured cause of the
    # n=1500 failure (kappa_max 126.9).
    kappa = float(np.linalg.cond(P.T @ P))
    return K, kappa


def identify_action_conditioned(
    corpus: LedgerCorpus,
    *,
    n_actions: Optional[int] = None,
    n_test_episodes: int = 10,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    max_kappa: float = DEFAULT_MAX_KAPPA,
    gate: float = DEFAULT_DELTA_GATE,
    ridge: float = DEFAULT_RIDGE,
    ground_truth: Optional[Sequence[np.ndarray]] = None,
    rollout_len: int = 25,
) -> IdentificationResult:
    """Identify one Koopman operator per action from payload-backed rows.

    Order of checks is the contract: payload presence, then row count, then
    conditioning, then the fit. Checking the fit before the pre-conditions is
    how a gate becomes uninterpretable.
    """
    if len(corpus.X) == 0 or corpus.n_usable == 0:
        return IdentificationResult(
            status=STATUS_NO_PAYLOADS,
            reason=(f"{corpus.n_rows} ledger rows carried no usable payload "
                    f"({corpus.n_unusable} unusable, "
                    f"{corpus.n_digest_mismatch} digest mismatches); "
                    "no (state, action, next_state) triple is reconstructable"),
            n_pairs=0, min_pairs=min_pairs, max_kappa=max_kappa, gate=gate,
        )

    n_pairs = int(len(corpus.X))
    if n_pairs < min_pairs:
        return IdentificationResult(
            status=STATUS_INSUFFICIENT,
            reason=(f"{n_pairs} pairs < pre-registered minimum {min_pairs}; "
                    "the fit is reachable but not interpretable at this "
                    "conditioning (measured: n=1500 gave kappa 126.9 and "
                    "fitted/truth 2.308)"),
            n_pairs=n_pairs, min_pairs=min_pairs, max_kappa=max_kappa, gate=gate,
        )

    d_state = int(corpus.X.shape[-1])
    actions = sorted(set(int(a) for a in corpus.A.tolist()))
    if n_actions is not None:
        actions = [a for a in actions if a < int(n_actions)]

    episode_ids = list(dict.fromkeys(corpus.episode_ids))
    if len(episode_ids) <= n_test_episodes:
        return IdentificationResult(
            status=STATUS_NO_FIT,
            reason=(f"{len(episode_ids)} episodes cannot be split into train "
                    f"and {n_test_episodes} held-out test episodes"),
            n_pairs=n_pairs, min_pairs=min_pairs, max_kappa=max_kappa, gate=gate,
        )
    train_eps = set(episode_ids[:-n_test_episodes])
    tr = np.array([i for i, e in enumerate(corpus.episode_ids) if e in train_eps])
    te = np.array([i for i, e in enumerate(corpus.episode_ids) if e not in train_eps])

    psi_all = quadratic_observables(corpus.X)
    operators: Dict[int, ActionOperator] = {}
    kappas: List[float] = []
    for a in actions:
        mask = tr[corpus.A[tr] == a]
        if len(mask) < 2:
            continue
        K, kappa = fit_action_operator(psi_all[mask], corpus.Y[mask], ridge=ridge)
        operators[a] = ActionOperator(
            action=a, K=K, n_fit=int(len(mask)), kappa_max=kappa,
            frobenius=float(np.linalg.norm(K)), safe=bool(kappa <= max_kappa),
        )
        kappas.append(kappa)

    if not operators:
        return IdentificationResult(
            status=STATUS_NO_FIT, reason="no action had enough rows to fit",
            n_pairs=n_pairs, min_pairs=min_pairs, max_kappa=max_kappa, gate=gate,
        )

    kappa_max = float(max(kappas))
    if kappa_max > max_kappa:
        return IdentificationResult(
            status=STATUS_ILL_CONDITIONED,
            reason=(f"kappa_max {kappa_max:.1f} exceeds the pre-registered "
                    f"maximum {max_kappa}; the fit is not interpretable "
                    "(measured cause of the Stage 4 n=1500 failure)"),
            kappa_max=kappa_max, n_train=int(len(tr)), n_test=int(len(te)),
            n_pairs=n_pairs, min_pairs=min_pairs, max_kappa=max_kappa, gate=gate,
            operators=operators,
        )

    # -- held-out scoring ---------------------------------------------------
    preds: List[np.ndarray] = []
    trues: List[np.ndarray] = []
    for i in te:
        a = int(corpus.A[i])
        if a not in operators:
            continue
        preds.append(quadratic_observables(corpus.X[i]) @ operators[a].K)
        trues.append(corpus.Y[i])
    if not preds:
        return IdentificationResult(
            status=STATUS_NO_FIT, reason="no held-out row had a fitted action",
            kappa_max=kappa_max, n_pairs=n_pairs, min_pairs=min_pairs,
            max_kappa=max_kappa, gate=gate, operators=operators,
        )
    P = np.asarray(preds)
    T = np.asarray(trues)
    delta_f = _rel_error(P, T)
    delta_p = _per_sample_error(P, T)

    # -- ground-truth comparison (isolates conditioning from estimation) ----
    delta_truth = ratio = None
    if ground_truth is not None:
        # Ground truth must be lifted into the SAME observable space as the
        # fit, or the comparison is a dimension error rather than a measurement.
        gt = _as_observable_operators(
            ground_truth, d_state=d_state, m=psi_all.shape[1]
        )
        gt_preds = [quadratic_observables(corpus.X[i]) @ gt[int(corpus.A[i])]
                    for i in te if int(corpus.A[i]) < len(gt)]
        if gt_preds:
            gt_arr = np.asarray(gt_preds)
            delta_truth = _rel_error(gt_arr, T)
            if delta_truth > 1e-12:
                ratio = delta_f / delta_truth

    # -- contraction check on a held-out episode ---------------------------
    first_ep = episode_ids[-n_test_episodes]
    ep_rows = sorted(
        [i for i, e in enumerate(corpus.episode_ids) if e == first_ep],
        key=lambda i: corpus.steps[i],
    )[:rollout_len]
    contractive = None
    err_first = err_last = None
    if len(ep_rows) >= 3 and operators:
        x_p = corpus.X[ep_rows[0]].copy()
        growth: List[float] = []
        for j, i in enumerate(ep_rows):
            a = int(corpus.A[ep_rows[j - 1]]) if j > 0 else int(corpus.A[i])
            if a not in operators:
                break
            x_p = quadratic_observables(x_p) @ operators[a].K
            growth.append(_rel_error(np.asarray(x_p), np.asarray(corpus.Y[i])))
        if growth:
            err_first, err_last = growth[0], growth[-1]
            contractive = bool(growth[-1] <= 3.0 * max(growth[0], 1e-9))

    return IdentificationResult(
        status=STATUS_OK,
        reason="identified per-action operators on held-out episodes",
        delta_frobenius=delta_f,
        delta_per_sample=delta_p,
        delta_ground_truth=delta_truth,
        delta_fitted_over_truth=ratio,
        kappa_max=kappa_max,
        n_train=int(len(tr)),
        n_test=int(len(te)),
        n_pairs=n_pairs,
        min_pairs=min_pairs,
        max_kappa=max_kappa,
        gate=gate,
        operators=operators,
        contractive=contractive,
        rollout_err_first=err_first,
        rollout_err_last=err_last,
    )


def predict_action_conditioned(
    result: IdentificationResult,
    state: np.ndarray,
    action: int,
) -> Optional[np.ndarray]:
    """One-step prediction through the identified per-action operator."""
    op = result.operators.get(int(action))
    if op is None:
        return None
    return quadratic_observables(np.asarray(state, dtype=np.float64)) @ op.K


def lift_linear_koopman(A: np.ndarray, d_state: int) -> np.ndarray:
    """Exact Koopman operator on [1, x, vech(x x^T)] for dynamics x' = A x.

    psi(x) @ K must equal A @ x. Only the linear block can contribute to a
    linear observable, so:

        K[1 : 1 + d, :] = A^T      (all other rows are zero)

    This is required to compare a FITTED operator against a known ground truth
    on the same observable space. Comparing a raw [d, d] matrix against an
    [m, d] observable operator is a dimension error, not a measurement.
    """
    A = np.asarray(A, dtype=np.float64)
    d = int(d_state)
    if A.shape != (d, d):
        raise ValueError(f"expected a [{d}, {d}] operator; got {A.shape}")
    m = 1 + d + d * (d + 1) // 2
    K = np.zeros((m, d), dtype=np.float64)
    K[1:1 + d, :] = A.T
    return K


def _as_observable_operators(
    ground_truth: Sequence[np.ndarray],
    d_state: int,
    m: int,
) -> List[np.ndarray]:
    """Normalize ground-truth operators into observable space.

    Accepts either an already-lifted [m, d] Koopman operator or a raw [d, d]
    linear operator, which is lifted exactly. Anything else is a typed error:
    silently reshaping would corrupt the comparison this exists to make.
    """
    out: List[np.ndarray] = []
    for g in ground_truth:
        g = np.asarray(g, dtype=np.float64)
        if g.shape == (m, d_state):
            out.append(g)
        elif g.shape == (d_state, d_state):
            out.append(lift_linear_koopman(g, d_state))
        else:
            raise ValueError(
                f"ground-truth operator has shape {g.shape}; expected "
                f"[{m}, {d_state}] (lifted) or [{d_state}, {d_state}] (linear)"
            )
    return out


def n_min_from_kappa(
    n_params: int,
    target_kappa: float = DEFAULT_MAX_KAPPA,
    safety: float = 8.0,
) -> int:
    """Rough pre-registration helper: rows required for `n_params` dictionary
    entries at a target conditioning.

    A ridge least-squares fit of `n_params` parameters needs an overdetermined
    system; the measured Stage 4 point (45 parameters per action, ~375 rows per
    action, kappa 127 -> FAIL; ~3000 rows/action, kappa 12 -> PASS) suggests an
    overdetermination factor near `safety`. This is a PLANNING aid, not a
    guarantee - always report the measured kappa.
    """
    return int(math.ceil(float(n_params) * float(safety)))
