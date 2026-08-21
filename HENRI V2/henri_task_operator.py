"""Path A in-context task operator — factorized least-squares R-EDMD (Class 4 accuracy remediation).

Spec: HENRI-ASSESSMENT-PATH-A-INFERENCE-2026 (drive inbox, 2026-08-20).
Mechanism (5 steps, per spec):
  Step 1  Ingress: authorized in-context (X_i, Y_i) pairs -> S^{D-1} waves
          via ASTDiscriminativeEncoder (structural phase encoding; the
          IDF-weighted variant requires an MBPP corpus file that is not
          staged in this worktree — recorded as a bounded deviation).
  Step 2  Cross-covariance: least-squares R-EDMD operator
          W_task = argmin_W sum_i ||W x_i - y_i||^2, solved in closed form
          as W = Y X^+ (thin SVD; no [D, D], no O(D^3)).
  Step 3  Polar/Procrustes structure: on the demo span the operator is the
          composition of two orthogonal factors (input basis U, output
          factor Q = Y V S^{-1} normalized); this is a partial isometry on
          span(X), NOT a globally unitary map. No unitary overclaim.
  Step 4  Inference: Psi_goal = W_task x_query (factorized: A (U^T x)).
  Step 5  Egress: rank candidates by phase alignment <Psi_c, Psi_goal>.

FACTORIZATION CONTRACT (architecture invariant): no [D, D] tensor may be
materialized (34 GiB at D=65,536). Storage is [D, r] + [r] factors only;
apply() allocates [D, r] / [r] / [D] intermediates at most. Asserted at
every allocation site by _assert_factor_shape.

DEMO-SOURCE BOUNDARY: the only authorized demo source on the HumanEval
path is the prompt's own docstring ">>>" examples — part of the task
specification the model receives, never the `test` field and never the
reference answer. Zero-pretraining invariant preserved: everything is
compiled 100% online at test time from in-context pairs.

Kill/accept gates are pre-registered in
experiments/verification/class4_path_a_operator_design.md.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from qfhrr_ast_discriminative_kernel import ASTDiscriminativeEncoder

# Docstring example extraction: ">>> call(args)" lines inside the prompt
# docstring. The first line after ">>> " is the call; subsequent lines up
# to a blank line are the expected output (multi-line outputs supported).
DOC_EXAMPLE_RE = re.compile(r"^\s*>>>\s+(.+)$")


def extract_docstring_examples(prompt: str, entry: str) -> List[Tuple[str, str]]:
    """Extract authorized (call_src, output_src) pairs from a prompt docstring.

    Returns [] when the prompt has no ">>>" examples. Only the prompt text
    is read — never the `test` field.
    """
    pairs: List[Tuple[str, str]] = []
    lines = prompt.splitlines()
    i = 0
    while i < len(lines):
        m = DOC_EXAMPLE_RE.match(lines[i])
        if m:
            call_src = m.group(1).strip()
            # sanity: the example must reference the entry function
            if not re.match(rf"^{re.escape(entry)}\s*\(", call_src):
                i += 1
                continue
            out_lines = []
            j = i + 1
            # Stop at: blank line, next ">>>" example, or the docstring's
            # closing triple-quote fence (preflight caught `"""` being
            # captured as output text -> "unterminated triple-quoted string").
            while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith(">>>") \
                    and not lines[j].strip().startswith('"""'):
                out_lines.append(lines[j].strip())
                j += 1
            if out_lines:
                out_src = "\n".join(out_lines)
                pairs.append((call_src, out_src))
            i = j
        else:
            i += 1
    return pairs


@dataclass
class FactorizedTaskOperator:
    """W_task in factorized least-squares form: y_hat = A (U^T x).

    U: [D, r] orthonormal columns (input basis of the demo span).
    A: [D, r] output factor (A = Y V S^{-1}).
    S: [r] singular values of the demo input matrix.
    rank: effective rank r = min(r_rank, M).
    orth_error: ||U^T U - I_r||_F after refinement.
    a_orth_error: ||A^T A - I_r||_F (near-isometry of the output factor).
    demo_mse: mean squared error on ALL demo pairs, batched.
    singular_values: S.tolist() (telemetry).
    per_pair_mse: [M] per-demo reconstruction errors (telemetry).
    """

    U: torch.Tensor  # [D, r]
    A: torch.Tensor  # [D, r]
    S: torch.Tensor  # [r]
    rank: int
    orth_error: float  # ||U^T U - I_r||_F
    a_orth_error: float  # ||A^T A - I_r||_F
    demo_mse: float   # mean squared error over ALL demo pairs (batched)
    singular_values: List[float]
    per_pair_mse: List[float]

    def device(self) -> torch.device:
        return self.U.device


def _assert_factor_shape(t: torch.Tensor, r: int, label: str) -> None:
    """Reject any dense [D, D] (or otherwise non-factor) allocation."""
    if t.dim() == 1:
        return
    if t.dim() == 2:
        # allowed: [D, r] (first dim == D), [r, r] Gram core, [M, r]
        if t.shape[0] == t.shape[1] and t.shape[0] > 64:
            raise AssertionError(f"{label}: dense square intermediate [{t.shape[0]}, {t.shape[1]}]")
        if t.shape[0] > 1_000_000 or t.shape[1] > 65536:
            raise AssertionError(f"{label}: oversized factor [{t.shape[0]}, {t.shape[1]}]")
        return
    raise AssertionError(f"{label}: unexpected dim {t.dim()}")


def _encode_source(encoder: ASTDiscriminativeEncoder, src: str) -> Optional[torch.Tensor]:
    """Encode a source string to a real unit [D] wave (production mapping)."""
    q = encoder.encode_code_string(src)
    if q is None:
        return None
    real = (q.to(torch.float32) / 255.0 * 2.0 - 1.0).to(encoder.device)
    return torch.nn.functional.normalize(real.view(-1), p=2, dim=0)


def compile_task_operator(
    demo_pairs: List[Tuple[str, str]],
    *,
    d_model: int = 65536,
    r_rank: int = 16,
    newton_schulz_iters: int = 3,
    device: Optional[str] = None,
) -> Optional[FactorizedTaskOperator]:
    """Compile the factorized least-squares R-EDMD task operator from (in, out) source pairs.

    Contract: W = A U^T with U [D, r] orthonormal and A = Y V S^{-1} [D, r].
    This is the least-squares operator on the demo span (a partial isometry
    when both factors are column-orthonormal). It is NOT globally unitary —
    no unitary overclaim is made.

    Returns None when fewer than 2 valid pairs encode (caller must mark the
    item NOT_EXPRESSIBLE — never fall back silently).
    """
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ASTDiscriminativeEncoder(d_model=d_model, device=dev)
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for in_src, out_src in demo_pairs:
        x = _encode_source(encoder, in_src)
        y = _encode_source(encoder, out_src)
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    if len(xs) < 2:
        return None
    X = torch.stack(xs, dim=1)  # [D, M]
    Y = torch.stack(ys, dim=1)  # [D, M]
    _assert_factor_shape(X, r_rank, "X")
    _assert_factor_shape(Y, r_rank, "Y")
    M = X.shape[1]
    r = min(r_rank, M)

    # Thin SVD of the demo input matrix: X = U S V^T, U [D, r], S [r], V [M, r].
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    U = U[:, :r].contiguous()
    S = S[:r].clamp_min(1e-12)
    V = Vh[:r, :].t().contiguous()  # [M, r]
    _assert_factor_shape(U, r, "U")
    _assert_factor_shape(V, r, "V")

    # Newton-Schulz orthonormalization on the [D, r] left factor (r x r Gram core).
    # Only the input factor is refined; the output factor A is NOT globally
    # unitarized (that would change the least-squares solution).
    for _ in range(max(0, newton_schulz_iters)):
        G = U.t() @ U  # [r, r] Gram core
        _assert_factor_shape(G, r, "G")
        U = U @ (1.5 * torch.eye(r, device=U.device) - 0.5 * G)
    U = torch.nn.functional.normalize(U, p=2, dim=0)
    orth_error = float(torch.norm(U.t() @ U - torch.eye(r, device=U.device)).item())

    # Output factor: A = Y V S^{-1}  ([D, M] @ [M, r] @ [r] -> [D, r]).
    A = (Y @ V) @ torch.diag(1.0 / S)
    _assert_factor_shape(A, r, "A")
    a_orth_error = float(torch.norm(A.t() @ A - torch.eye(r, device=A.device)).item())

    # Demo reconstruction MSE over ALL demos batched (no broadcast bug):
    # Y_hat = A (U^T X), MSE = mean((Y_hat - Y)^2).
    proj_all = U.t() @ X  # [r, M]
    _assert_factor_shape(proj_all, r, "proj_all")
    y_hat_all = A @ proj_all  # [D, M]
    _assert_factor_shape(y_hat_all, r, "y_hat_all")
    per_pair = torch.mean((y_hat_all - Y).pow(2), dim=0)  # [M]
    demo_mse = float(torch.mean(per_pair).item())

    return FactorizedTaskOperator(
        U=U, A=A, S=S, rank=r, orth_error=orth_error, a_orth_error=a_orth_error,
        demo_mse=demo_mse, singular_values=S.tolist(),
        per_pair_mse=[float(v) for v in per_pair.tolist()])


def apply_task_operator(op: FactorizedTaskOperator, x: torch.Tensor) -> torch.Tensor:
    """Project a query wave through W_task: y_hat = A (U^T x). Returns [D] raw."""
    _assert_factor_shape(x, op.rank, "x")
    x = x.view(-1).to(op.device())
    proj = op.U.t() @ x  # [r]
    _assert_factor_shape(proj, op.rank, "proj")
    y_hat = op.A @ proj  # [D]
    _assert_factor_shape(y_hat, op.rank, "y_hat")
    return y_hat


def rank_by_goal(
    candidates: List[Tuple[str, dict]],
    op: FactorizedTaskOperator,
    query_src: str,
    *,
    top_k: Optional[int] = None,
) -> List[Tuple[str, dict]]:
    """Reorder the candidate SET (membership unchanged) by phase alignment with Psi_goal.

    Psi_goal = W_task x_query, where x_query is encoded from query_src with
    the SAME ASTDiscriminativeEncoder family used to compile the operator
    (representation-consistent by construction — never a random-ring wave).
    score(c) = <normalize(c_wave), normalize(Psi_goal)>.
    Candidate set membership is never altered; the runner slices by its own
    attempt budget. top_k (when given) only bounds the returned list length.
    """
    d_model = op.U.shape[0]
    enc = ASTDiscriminativeEncoder(d_model=d_model, device=str(op.device()))
    query_wave = _encode_source(enc, query_src)
    if query_wave is None:
        return list(candidates)
    goal = apply_task_operator(op, query_wave)
    goal = torch.nn.functional.normalize(goal, p=2, dim=0)
    scored: List[Tuple[float, int, str, dict]] = []
    for ci, (src, meta) in enumerate(candidates):
        v = _encode_source(enc, src)
        sim = float(torch.dot(torch.nn.functional.normalize(v, p=2, dim=0), goal).item()) if v is not None else -1e9
        scored.append((sim, ci, src, meta))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out = [(src, meta) for _, _, src, meta in scored]
    if top_k is not None:
        out = out[:top_k]
    return out
