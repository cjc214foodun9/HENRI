"""
Project HENRI: FunctorFlow — categorical composition engine for wave states.

Thin wrapper exposing existing Clifford/Hopfield/EFE primitives under
categorical names. Every operation maps directly onto a verified codebase
primitive; this file adds the naming layer, not new math.

FunctorFlow operations:
    KET       — Left Kan extension: attention / aggregation
    DB        — Diagonal Bridge: commutativity check
    GT        — Geometric Transport: neighborhood consistency
    OBSTRUCTION — off-manifold residual (constraint penalty)
    BASKET    — Repair pipeline: preference-guided completion (Right Kan)
    ROCKET    — Workflow pipeline: compose operations with obstruction gates

References:
    - FunctorFlow design doc (user-provided, 2026-07-22)
    - product_clifford_product_kernel.py (composition engine)
    - efe_planner.py (constraint / obstruction)
    - hopfield_cleanup.py (retrieval / colimit)
    - zone_c_segment_cache.py (GRM / sheaf)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from product_clifford_product_kernel import ProductCliffordAlgebra3D
from hopfield_cleanup import ContinuousHopfieldCleanup


class FunctorFlow(nn.Module):
    """Categorical composition engine over Clifford wave states.

    Wraps a Clifford algebra instance + optional EFE planner reference for
    operations that need the constraint subspace or preference store.
    All operations are pure functions of their wave inputs — no hidden state
    beyond the algebra tables and the planner handle.
    """

    def __init__(self, num_blocks: int = 8192, planner=None):
        super().__init__()
        self.num_blocks = num_blocks
        self.clifford = ProductCliffordAlgebra3D(num_blocks=num_blocks)
        self.planner = planner  # optional: needed for obstruction/basket

    # ------------------------------------------------------------------
    # KET: Left Kan extension — attention / aggregation
    # ------------------------------------------------------------------

    def ket(self, wave_a: torch.Tensor, wave_b: torch.Tensor) -> torch.Tensor:
        """Left Kan: scalar-part projection of geometric product.

        In categorical terms, the scalar part of ab is the "hom-object"
        measuring compatibility of morphisms a and b — the attention weight.
        Returns [B, K] scalar similarity per block.

        wave_a, wave_b: [B, K, 8] or [K, 8] Clifford multivectors.
        """
        if wave_a.dim() == 2:
            wave_a = wave_a.unsqueeze(0)
        if wave_b.dim() == 2:
            wave_b = wave_b.unsqueeze(0)
        gp = self.clifford.geometric_product(wave_a, wave_b)
        return gp[..., 0]  # grade-0 scalar part

    def ket_attention(self, query: torch.Tensor, keys: torch.Tensor,
                      temperature: float = 0.1) -> torch.Tensor:
        """Multi-key attention: KET over each key, softmax-gated.

        query: [K, 8], keys: [M, K, 8]. Returns [M] attention weights.
        """
        q = query.unsqueeze(0).expand(keys.shape[0], -1, -1)
        sim = self.ket(q, keys)  # [M, K]
        # Aggregate over blocks: mean scalar similarity
        scores = sim.mean(dim=-1)  # [M]
        return torch.softmax(scores / temperature, dim=0)

    # ------------------------------------------------------------------
    # DB: Diagonal Bridge — commutativity check
    # ------------------------------------------------------------------

    def db(self, wave_a: torch.Tensor, wave_b: torch.Tensor) -> torch.Tensor:
        """Diagonal Bridge: commutator norm of geometric product.

        [a, b] = ab - ba. Zero when a and b commute (same grade, aligned).
        Non-zero measures structural obstruction — the failure of a diagram
        to commute. Returns [B, K, 8] commutator per block.

        wave_a, wave_b: [B, K, 8] or [K, 8].
        """
        if wave_a.dim() == 2:
            wave_a = wave_a.unsqueeze(0)
        if wave_b.dim() == 2:
            wave_b = wave_b.unsqueeze(0)
        ab = self.clifford.geometric_product(wave_a, wave_b)
        ba = self.clifford.geometric_product(wave_b, wave_a)
        return ab - ba

    def db_score(self, wave_a: torch.Tensor, wave_b: torch.Tensor) -> float:
        """Scalar commutativity score: mean commutator norm over blocks.

        0 = fully commutative (compatible morphisms).
        Large = structural obstruction.
        """
        comm = self.db(wave_a, wave_b)
        return float(comm.norm(p=2, dim=-1).mean())

    # ------------------------------------------------------------------
    # GT: Geometric Transport — neighborhood consistency
    # ------------------------------------------------------------------

    def gt(self, wave: torch.Tensor) -> torch.Tensor:
        """Geometric Transport: toroidal 2D discrete Laplacian.

        Measures spatial consistency of the wave across the block grid.
        High GT = rough, spatially incoherent; low GT = smooth, coherent.
        Returns [K, 8] Laplacian per block.

        wave: [K, 8] or [num_blocks, 8].
        """
        psi = wave
        K = psi.shape[0]
        import math
        side = int(math.isqrt(K))
        if side * side != K:
            # Non-square grid: fall back to 1D Laplacian along block axis
            lap = (torch.roll(psi, 1, 0) + torch.roll(psi, -1, 0) - 2.0 * psi)
            return lap
        grid = psi.view(side, side, 8)
        lap = (torch.roll(grid, 1, 0) + torch.roll(grid, -1, 0)
               + torch.roll(grid, 1, 1) + torch.roll(grid, -1, 1)
               - 4.0 * grid)
        return lap.view(K, 8)

    def gt_stress(self, wave: torch.Tensor) -> float:
        """Scalar geometric transport stress: 0.5 * ||GT(wave)||^2."""
        lap = self.gt(wave)
        return float(0.5 * (lap ** 2).sum())

    # ------------------------------------------------------------------
    # OBSTRUCTION: off-manifold residual (constraint penalty)
    # ------------------------------------------------------------------

    def obstruction(self, wave: torch.Tensor) -> float:
        """Obstruction loss: how far the wave lies from the learned invariant
        subspace. Delegates to the planner's constraint_penalty when available;
        returns 0.0 when no planner is attached (no constraint to violate).

        wave: [num_blocks, 8] Clifford wave.
        """
        if self.planner is None or self.planner.axiom_constraint.numel() == 0:
            return 0.0
        return self.planner.constraint_penalty(wave)

    def obstruction_reject(self, wave: torch.Tensor, threshold: float = None) -> bool:
        """Binary veto: True if the wave is off-manifold beyond threshold."""
        if threshold is None and self.planner is not None:
            threshold = self.planner.constraint_reject_thresh
        elif threshold is None:
            threshold = 0.5
        obs = self.obstruction(wave)
        return obs > threshold

    # ------------------------------------------------------------------
    # BASKET: Repair pipeline — Right Kan extension (preference completion)
    # ------------------------------------------------------------------

    def basket(self, wave: torch.Tensor, alpha: float = 0.3,
               max_attempts: int = 3) -> tuple:
        """PEARL repair: preference-guided completion of off-manifold wave.

        Right Kan extension: given a partial wave (rejected by obstruction),
        complete it by blending toward the preference store — the catalog
        of historically favorable transition waves. This is the categorical
        dual of KET: while KET aggregates, BASKET completes.

        Returns (repaired_wave, residual_type, repair_cost):
            residual_type: 'ACCEPTED_CLEAN', 'ACCEPTED_PEARL_REPAIRED',
                           'REJECTED_REPAIR_FAILED', 'REJECTED_NO_PREFS'
            repair_cost: final obstruction score after repair
        """
        obs = self.obstruction(wave)
        if obs == 0.0:
            # No constraint subspace yet — accept as clean
            return wave, "ACCEPTED_CLEAN", 0.0

        reject_thresh = (self.planner.constraint_reject_thresh
                         if self.planner is not None else 0.5)

        if obs <= reject_thresh:
            return wave, "ACCEPTED_CLEAN", obs

        # Need preferences to repair
        if (self.planner is None
                or self.planner.preference_store.num_engrams() == 0):
            return wave, "REJECTED_NO_PREFS", obs

        prefs = self.planner.preference_store.engrams
        flat = wave.detach().reshape(-1)
        flat = flat / (torch.norm(flat) + 1e-12)

        best_wave = wave
        best_penalty = obs
        best_alpha = 0.0

        for attempt in range(max_attempts):
            # Retrieve most resonant preference
            sim = flat @ prefs.T
            top_idx = sim.argmax()
            pref_flat = prefs[top_idx]
            pref_wave = pref_flat.view_as(wave)

            # Phase blend: steer toward favorable basin
            alpha_eff = alpha * (1.0 - attempt / max_attempts)
            repaired = (1 - alpha_eff) * best_wave + alpha_eff * pref_wave
            repaired = repaired / (torch.norm(repaired, p=2, dim=-1, keepdim=True) + 1e-9)

            new_penalty = self.obstruction(repaired)
            if new_penalty <= reject_thresh:
                return repaired, "ACCEPTED_PEARL_REPAIRED", new_penalty

            if new_penalty < best_penalty:
                best_wave, best_penalty, best_alpha = repaired, new_penalty, alpha_eff

        return best_wave, "REJECTED_REPAIR_FAILED", best_penalty

    # ------------------------------------------------------------------
    # ROCKET: Workflow pipeline — compose operations with obstruction gates
    # ------------------------------------------------------------------

    def rocket(self, wave: torch.Tensor,
               operations: list,
               obstruction_threshold: float = None) -> list:
        """ROCKET workflow: compose a sequence of transformations with
        obstruction checks. Each operation is a callable(wave) -> wave.
        After each step, checks obstruction; on violation, attempts BASKET
        repair before proceeding.

        Returns list of (step_name, output_wave, residual_type, obstruction_score).
        """
        if obstruction_threshold is None and self.planner is not None:
            obstruction_threshold = self.planner.constraint_reject_thresh
        elif obstruction_threshold is None:
            obstruction_threshold = 0.5

        log = []
        current = wave

        for i, (name, op) in enumerate(operations):
            current = op(current)
            obs = self.obstruction(current)

            if obs > obstruction_threshold:
                repaired, rtype, cost = self.basket(current)
                log.append((f"{name}", current, "REJECTED", obs))
                if rtype.startswith("ACCEPTED"):
                    current = repaired
                    log.append((f"{name}_repair", current, rtype, cost))
                else:
                    log.append((f"{name}_repair_failed", current, rtype, cost))
            else:
                log.append((name, current, "ACCEPTED_CLEAN", obs))

        return log

    # ------------------------------------------------------------------
    # Type classifier: wave grade distribution
    # ------------------------------------------------------------------

    def typeof(self, wave: torch.Tensor) -> dict:
        """Classify a wave by its Clifford grade power distribution.

        Returns dict with per-grade L2 norm and dominant grade.
        wave: [K, 8] — each row is a multivector [s, e1, e2, e3, e12, e23, e31, e123].
        """
        grades = ["scalar", "e1", "e2", "e3", "e12", "e23", "e31", "e123"]
        power = (wave ** 2).mean(dim=0)  # [8] mean squared amplitude per grade
        total = power.sum() + 1e-12
        dist = {g: float(p / total) for g, p in zip(grades, power)}
        dominant = max(dist, key=dist.get)
        return {"distribution": dist, "dominant_grade": dominant,
                "total_power": float(total)}

    # ------------------------------------------------------------------
    # Composition: sequence two waves through the geometric product
    # ------------------------------------------------------------------

    def compose(self, wave_a: torch.Tensor, wave_b: torch.Tensor) -> torch.Tensor:
        """Categorical composition: a ∘ b = geometric_product(a, b).

        The Clifford geometric product IS the composition operator in this
        category. Grade-tracking is automatic — the product of a grade-m
        and grade-n blade produces grades |m−n| through m+n.
        """
        if wave_a.dim() == 2:
            wave_a = wave_a.unsqueeze(0)
        if wave_b.dim() == 2:
            wave_b = wave_b.unsqueeze(0)
        return self.clifford.geometric_product(wave_a, wave_b)

    def dagger(self, wave: torch.Tensor) -> torch.Tensor:
        """Adjoint (Clifford reversion): flips bivectors and trivector.

        Grade-k reversion sign: (-1)^{k(k-1)/2}.
        Vectors (grade 1): no flip. Bivectors (grade 2): flip.
        Trivector (grade 3): flip.
        """
        out = wave.clone()
        out[..., [4, 5, 6, 7]] *= -1.0
        return out

    # ------------------------------------------------------------------
    # Diagram: check commutativity of a square
    # ------------------------------------------------------------------

    def diagram_commutes(self, top: torch.Tensor, right: torch.Tensor,
                         bottom: torch.Tensor, left: torch.Tensor,
                         threshold: float = 0.1) -> tuple:
        """Check if a categorical square commutes: top∘right ≈ bottom∘left.

        Returns (commutes: bool, obstruction: float).
        """
        path1 = self.compose(top, right)
        path2 = self.compose(bottom, left)
        obs = self.db_score(path1, path2)
        return obs < threshold, obs

    # ------------------------------------------------------------------
    # Colimit: weighted superposition (Hopfield softmax retrieval)
    # ------------------------------------------------------------------

    @staticmethod
    def colimit(query: torch.Tensor, engrams: torch.Tensor,
                beta: float = None, temperature: float = None) -> torch.Tensor:
        """Categorical colimit: weighted superposition over engrams.

        This IS the Hopfield retrieval operation — the universal property
        of the colimit: the "best approximation" to the query from the
        diagram of stored engrams.

        query: [D], engrams: [M, D]. Returns [D] colimit wave.

        When temperature is provided, uses soft-temperature weighting
        (epistemic readout); otherwise uses engram-store beta (hard snap).
        """
        q = F.normalize(query.reshape(-1), p=2, dim=-1)
        e = engrams.to(q.device)
        sim = q @ e.T
        if temperature is not None:
            w = torch.softmax(sim / temperature, dim=-1)
        elif beta is not None:
            w = torch.softmax(beta * sim, dim=-1)
        else:
            # Hard snap: one-hot at argmax
            idx = sim.argmax()
            w = torch.zeros_like(sim)
            w[idx] = 1.0
        colim = w @ e
        return F.normalize(colim, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Smoke: verify all operations run without error
# ---------------------------------------------------------------------------

def _smoke():
    """Verify every FunctorFlow operation executes without error."""
    ff = FunctorFlow(num_blocks=64)
    a = torch.randn(64, 8)
    b = torch.randn(64, 8)
    a = a / torch.norm(a, p=2, dim=-1, keepdim=True)
    b = b / torch.norm(b, p=2, dim=-1, keepdim=True)

    # KET
    k = ff.ket(a, b)
    assert k.shape == (1, 64), f"ket shape {k.shape}"

    # DB
    comm = ff.db(a, b)
    assert comm.shape == (1, 64, 8), f"db shape {comm.shape}"
    dbs = ff.db_score(a, b)
    assert isinstance(dbs, float), f"db_score type {type(dbs)}"

    # GT
    lap = ff.gt(a)
    assert lap.shape == (64, 8), f"gt shape {lap.shape}"
    gts = ff.gt_stress(a)
    assert isinstance(gts, float), f"gt_stress type {type(gts)}"

    # OBSTRUCTION
    obs = ff.obstruction(a)
    assert obs == 0.0, "obstruction should be 0.0 without planner"

    # BASKET
    repaired, rtype, cost = ff.basket(a)
    assert rtype == "ACCEPTED_CLEAN", f"basket type {rtype}"

    # ROCKET
    ops = [("ident", lambda x: x), ("scale", lambda x: 0.5 * x)]
    log = ff.rocket(a, ops)
    assert len(log) == 2, f"rocket log length {len(log)}"

    # typeof
    t = ff.typeof(a)
    assert "dominant_grade" in t, f"typeof keys {t.keys()}"

    # compose
    c = ff.compose(a, b)
    assert c.shape == (1, 64, 8), f"compose shape {c.shape}"

    # dagger (reversion)
    d = ff.dagger(a)
    assert d.shape == a.shape, f"dagger shape {d.shape}"
    assert torch.allclose(d[..., :4], a[..., :4]), "dagger should preserve vectors"
    assert torch.allclose(d[..., 4:7], -a[..., 4:7]), "dagger should flip bivectors"

    # diagram_commutes (identity square)
    ok, obs = ff.diagram_commutes(a, b, a, b)
    assert ok, f"identity diagram should commute, obs={obs}"

    # colimit: query [D=512], engrams [M=8, D=512]
    D = 64 * 8
    eng = torch.randn(8, D)
    eng = F.normalize(eng, p=2, dim=-1)
    col = ff.colimit(a.reshape(-1), eng, temperature=0.1)
    assert col.shape == (D,), f"colimit shape {col.shape}"

    print("[FunctorFlow] All smoke checks passed.")


if __name__ == "__main__":
    _smoke()
