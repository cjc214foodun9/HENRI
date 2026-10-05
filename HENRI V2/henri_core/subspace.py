"""Universal Weight Subspace extraction and drift audit.

Authority: the daydream document, sections 3.2 and 4.4. The document attributes
the hypothesis to Kaushik et al. (arXiv:2512.05117, verified 2026-10-04: title,
authors, and the >1100-model / few-principal-directions claim are confirmed in
the abstract; the theorem numbering and the Davis-Kahan constant in the document
are the authors' rendering and are NOT verified).

HONEST SCOPE -- do not launder the paper's claim
    The paper's evidence is 1,100+ models. We have ONE decoder. We therefore do
    NOT claim the Kaushik universality result. We test only the consequence the
    daydream engine depends on: that THIS decoder's weight trajectory occupies a
    low-dimensional subspace with sharp eigenvalue decay. If it does, the
    retraction W <- U_k U_k^T W is nearly lossless. If it does not, the
    retraction costs capacity and the gate says so.

One term per meaning:
    update vector   delta_theta, the flattened parameter displacement between
                    two checkpoints of one model
    trajectory      the matrix of update vectors over T checkpoints
    U_k             the top-k right singular directions of the centred trajectory
    projector       P_k = U_k U_k^T
"""
from __future__ import annotations

import torch


class UniversalSubspace:
    """Extract U_k from a weight-update trajectory and audit its stability.

    Args:
        k: number of retained directions (the document uses k <= 32)
    """

    def __init__(self, k: int = 32):
        self.k = int(k)
        self._rows: list[torch.Tensor] = []
        self.U: torch.Tensor | None = None
        self.singular: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None

    # ------------------------------------------------------------------ collect
    @torch.no_grad()
    def observe(self, vector: torch.Tensor) -> None:
        """Append one update vector (flattened parameter displacement)."""
        self._rows.append(vector.detach().reshape(-1).float().clone())

    @torch.no_grad()
    def observe_params(self, model: torch.nn.Module, prev: dict | None) -> dict:
        """Snapshot the model. Returns the snapshot to pass in as `prev` next time.

        The first call records no update, because there is no previous state.
        """
        flat = torch.cat([p.detach().reshape(-1).float()
                          for p in model.parameters()])
        if prev is not None and "flat" in prev:
            self.observe(flat - prev["flat"])
        return {"flat": flat}

    @property
    def n_observations(self) -> int:
        return len(self._rows)

    # -------------------------------------------------------------------- fit
    @torch.no_grad()
    def fit(self) -> "UniversalSubspace":
        if len(self._rows) < 2:
            raise RuntimeError("need at least 2 update vectors to fit")
        D = torch.stack(self._rows)                        # [T, P]
        self.mean = D.mean(0, keepdim=True)
        Dc = D - self.mean
        # D95 (self-caught): Dc is [T, P] with T << P, so torch.linalg.svd
        # returns U [T, T], S [T], Vh [T, P]. The basis vectors that live in
        # PARAMETER space are the ROWS of Vh (i.e. Vh.T), NOT the columns of U.
        # The first draft used U[:, :k], shape [T, k], so project() would have
        # multiplied a [P] vector by a [T, k] matrix and failed -- or worse,
        # silently produced a wrong result when T == P.
        U, S, Vh = torch.linalg.svd(Dc, full_matrices=False)
        self.singular = S
        k = min(self.k, Vh.shape[0])
        self.U = Vh.T[:, :k].contiguous()                  # [P, k]
        return self

    def explained_variance(self, k: int | None = None) -> float:
        s2 = self.singular ** 2
        k = int(k or self.k)
        return float(s2[:k].sum() / s2.sum().clamp_min(1e-30))

    def project(self, delta: torch.Tensor) -> torch.Tensor:
        """Retract a vector onto the subspace: U_k U_k^T v."""
        v = delta.reshape(-1).float()
        v = v - self.mean.squeeze(0)
        return self.U @ (self.U.T @ v)

    def project_inplace(self, model: torch.nn.Module) -> float:
        """Project the model's current parameters onto the subspace anchor."""
        if self.U is None:
            raise RuntimeError("fit() first")
        with torch.no_grad():
            off = 0
            moved = 0.0
            for p in model.parameters():
                n = p.numel()
                v = p.detach().reshape(-1).float() - self.mean.squeeze(0)[off:off + n]
                proj = self.U @ (self.U.T @ v)
                new = proj + self.mean.squeeze(0)[off:off + n]
                moved += float((new - p.detach().reshape(-1).float()).abs().sum())
                p.copy_(new.reshape(p.shape).to(p.dtype))
                off += n
        return moved

    # ------------------------------------------------------------------ drift
    def half_split_drift(self) -> dict:
        """Davis-Kahan sinTheta audit: projectors from the two trajectory halves.

        Returns the operator-norm distance between the two empirical projectors,
        the estimated eigengap gamma_k, and the document's bound 0.05 / gamma_k.
        A small distance means the subspace is STABLE across the trajectory, which
        is the precondition for retracting updates onto it without drift.
        """
        if len(self._rows) < 4:
            raise RuntimeError("need at least 4 update vectors for a drift audit")
        D = torch.stack(self._rows)
        h = len(self._rows) // 2
        U1 = self._basis(D[:h])
        U2 = self._basis(D[h:])
        P1 = U1 @ U1.T
        P2 = U2 @ U2.T
        dist = float(torch.linalg.matrix_norm(P1 - P2, ord=2))
        s = self.singular
        k = self.U.shape[1]
        gamma = float(s[k - 1] ** 2 - s[k] ** 2) if len(s) > k else float(s[k - 1] ** 2)
        bound = 0.05 / max(gamma, 1e-12)
        return {"op_drift": dist, "gamma_k": gamma, "bound": bound,
                "passes": dist <= bound, "n_points": len(self._rows)}

    @staticmethod
    @torch.no_grad()
    def _basis(D: torch.Tensor, k: int = 32) -> torch.Tensor:
        """Right singular vectors of a [T, P] trajectory block, as [P, k]."""
        Dc = D - D.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(Dc, full_matrices=False)
        kk = min(k, Vh.shape[0])
        return Vh.T[:, :kk].contiguous()
