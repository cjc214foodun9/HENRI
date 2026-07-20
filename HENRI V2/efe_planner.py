"""
Project HENRI: Expected Free Energy (EFE) Action Planner.

Implements Friston's active-inference action selection over the swarm's
continuous wave states. For each candidate action, the planner propagates the
current state wave through a unitary transition operator (the 65k-dim analog
of WaveJEPA), then scores the predicted next wave by Expected Free Energy:

    EFE(a) = pragmatic_value(a) - epistemic_value(a)

    pragmatic_value = Sagnac delta of predicted wave vs boundary axioms
                      (expected surprise / violation of prior preferences)

    epistemic_value = information gain, measured as the entropy of the
                      Hopfield retrieval distribution over attractor engrams
                      induced by the predicted wave (uncertainty reduction)

Action selection: a* = argmin_a EFE(a).

References: FEP synthesis (nlm_fep.md), JEPA transition spec (nlm_jepa.md),
Hopfield cleanup (nlm_hopfield.md). Circular convolution binds state and
action into a fused intent wave before the unitary transition, matching the
FHRR binding algebra used across the rest of the stack.
"""

import math
import torch
import torch.nn as nn
import torch.fft as fft

from hopfield_cleanup import ContinuousHopfieldCleanup


class UnitaryWaveTransition(nn.Module):
    """
    Action-conditioned forward dynamics in latent wave space.

    predicted = U * (state ⊗ action), where ⊗ is circular convolution
    (FHRR binding) and U is a Stiefel-constrained complex matrix acting on
    the flattened wave. To keep memory tractable at d=65536, U is factored
    as a block-diagonal operator over the [num_blocks, 8] Clifford grid:
    one 8x8 complex matrix per block, shared structure via a low-rank core.
    """

    def __init__(self, num_blocks: int = 8192, block_dim: int = 8):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        # Per-block complex transition matrices, initialized near-unitary
        real = torch.eye(block_dim) + 0.01 * torch.randn(num_blocks, block_dim, block_dim)
        imag = 0.01 * torch.randn(num_blocks, block_dim, block_dim)
        self.transition = nn.Parameter(torch.complex(real, imag))
        self._retract()

    @torch.no_grad()
    def _retract(self):
        """Project each block matrix toward unitarity via QR of the complex matrix."""
        Q, _ = torch.linalg.qr(self.transition)
        self.transition.copy_(Q)

    def bind(self, state_wave: torch.Tensor, action_wave: torch.Tensor) -> torch.Tensor:
        """
        FHRR circular convolution binding over the Clifford grid.
        state_wave, action_wave: [num_blocks, 8] real. Returns complex
        [num_blocks, 8] fused intent (unit modulus per block row).
        """
        s = torch.complex(state_wave[..., :4], state_wave[..., 4:])  # [blocks, 4] complex
        a = torch.complex(action_wave[..., :4], action_wave[..., 4:])
        fs = fft.fft(s, dim=-1)
        fa = fft.fft(a, dim=-1)
        bound = fft.ifft(fs * fa, dim=-1)
        bound = bound / (torch.norm(bound, p=2, dim=-1, keepdim=True) + 1e-9)
        # Embed back to 8-dim: 4 complex -> interleaved re/im
        return torch.cat([bound.real, bound.imag], dim=-1).to(torch.complex64)

    def forward(self, state_wave: torch.Tensor, action_wave: torch.Tensor) -> torch.Tensor:
        """
        state_wave, action_wave: [num_blocks, 8] real Clifford waves.
        Returns predicted next wave [num_blocks, 8] real, unit-norm per block.
        """
        fused = self.bind(state_wave, action_wave)  # [blocks, 8] complex
        # Block-diagonal unitary transition: einsum over per-block 8x8 matrices
        predicted = torch.einsum('bij,bj->bi', self.transition, fused)
        predicted_real = torch.cat([predicted.real, predicted.imag], dim=-1)[..., :8] \
            if predicted.shape[-1] != 8 else predicted.real
        # Normalize per block to stay on the manifold
        out = predicted_real / (torch.norm(predicted_real, p=2, dim=-1, keepdim=True) + 1e-9)
        return out


class EFEPlanner(nn.Module):
    """
    Scores candidate actions by Expected Free Energy and selects a*.

    Holds the swarm's action engram store (shared with the Hopfield decoder),
    the transition operator, and the scoring weights.
    """

    def __init__(
        self,
        num_blocks: int = 8192,
        d_model: int = 65536,
        action_engrams: torch.Tensor = None,
        epistemic_weight: float = 1.0,
        pragmatic_weight: float = 1.0,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.epistemic_weight = epistemic_weight
        self.pragmatic_weight = pragmatic_weight

        self.transition = UnitaryWaveTransition(num_blocks=num_blocks)
        # Retrieval store over predicted waves (real Clifford waves of width
        # d_model); engrams registered externally (decoder action basis, Zone C
        # attractors).
        self.cleanup = ContinuousHopfieldCleanup(dim=d_model)
        if action_engrams is not None:
            self.cleanup.store_engrams(action_engrams)

    # -- value terms ------------------------------------------------------

    def pragmatic_value(self, predicted_wave: torch.Tensor, boundary_axioms: torch.Tensor) -> torch.Tensor:
        """
        Expected surprise: normalized Sagnac delta of the predicted wave
        against boundary axioms. boundary_axioms: [N, num_blocks, 8].
        Returns scalar in [0, 2].
        """
        p = predicted_wave.view(-1)
        p = p / (torch.norm(p) + 1e-12)
        deltas = []
        for axiom in boundary_axioms:
            a = axiom.view(-1)
            a = a / (torch.norm(a) + 1e-12)
            inner = torch.dot(p, a)  # real waves: plain inner product
            deltas.append(1.0 - inner)
        return torch.stack(deltas).min()  # closest axiom governs

    def epistemic_value(self, predicted_wave: torch.Tensor) -> torch.Tensor:
        """
        Information gain: entropy of the Hopfield retrieval distribution over
        attractor engrams induced by the predicted wave. High entropy = the
        prediction is spread across many attractors = high uncertainty about
        outcome = potentially informative action. Returns scalar >= 0.
        """
        if self.cleanup.num_engrams() == 0:
            return torch.tensor(0.0, device=predicted_wave.device)
        flat = predicted_wave.view(-1)
        flat = flat / (torch.norm(flat) + 1e-12)
        _, weights = self.cleanup.retrieve(flat, return_weights=True)
        w = weights.clamp(min=1e-12)
        return -(w * torch.log(w)).sum()

    # -- planning ---------------------------------------------------------

    def score_actions(self, state_wave: torch.Tensor, candidate_actions: list, boundary_axioms: torch.Tensor):
        """
        candidate_actions: list of (action_id, action_wave[num_blocks, 8]).
        Returns list of dicts sorted by EFE ascending (best first).
        """
        results = []
        for action_id, action_wave in candidate_actions:
            predicted = self.transition(state_wave, action_wave)
            pragmatic = self.pragmatic_value(predicted, boundary_axioms)
            epistemic = self.epistemic_value(predicted)
            efe = self.pragmatic_weight * pragmatic - self.epistemic_weight * epistemic
            results.append({
                "action": action_id,
                "efe": efe.item(),
                "pragmatic": pragmatic.item(),
                "epistemic": epistemic.item(),
                "predicted_wave": predicted,
            })
        results.sort(key=lambda r: r["efe"])
        return results

    def select_action(self, state_wave: torch.Tensor, candidate_actions: list, boundary_axioms: torch.Tensor,
                      explore_threshold: float = None):
        """
        Returns (best_action_id, predicted_wave, scores_table).

        T4 — calibrated exploration: when the planner's uncertainty (the EFE
        spread across candidates) exceeds explore_threshold, the model's
        dynamics are unreliable, so we select the highest-EPISTEMIC-value
        action (max information gain) instead of the lowest-EFE one. When
        confident, we exploit (min EFE). Default threshold: adaptive, the
        median of the observed spread.
        """
        results = self.score_actions(state_wave, candidate_actions, boundary_axioms)
        best = results[0]
        spread = results[-1]["efe"] - results[0]["efe"]

        if explore_threshold is None:
            # Default: explore when the spread is a large fraction of the
            # mean |EFE| — i.e. candidates are poorly separated.
            mean_abs = sum(abs(r["efe"]) for r in results) / max(len(results), 1)
            explore_threshold = 0.5 * mean_abs

        if spread > explore_threshold and len(results) > 1:
            # Confused: take the most informative action instead
            epistemic_best = max(results, key=lambda r: r["epistemic"])
            best = epistemic_best
            best = dict(best, explored=True)
        else:
            best = dict(best, explored=False)

        best["spread"] = spread
        best["explore_threshold"] = explore_threshold
        # Annotate which table entry was actually chosen so callers can see
        # whether the returned action was the exploit or explore pick.
        chosen = dict(best)
        results = [dict(r, chosen=(r["action"] == chosen["action"])) for r in results]
        return best["action"], best["predicted_wave"], results, chosen

    def train_transition_step(
        self,
        state_wave: torch.Tensor,
        action_wave: torch.Tensor,
        observed_next_wave: torch.Tensor,
        lr: float = 0.05,
    ) -> float:
        """
        Online latent-space dynamics learning (T1 + T2).

        Trains the transition operator on the Sagnac signal itself: the loss
        is the normalized coherence delta between the model's prediction and
        the observed next wave — the exact quantity the swarm measures.

            predicted = transition(state ⊗ action)
            loss      = 1 - Re(<predicted, observed>) / (||predicted|| ||observed||)

        The (state, action) pair is the EXECUTED action's canonical wave, so
        the model learns action-conditioned dynamics rather than an
        action-averaged blur (T2). After the Wirtinger step the transition
        matrices are re-retracted to unitarity.

        Returns the pre-update loss (the Sagnac delta this step).
        """
        with torch.enable_grad():
            predicted = self.transition(state_wave.detach(), action_wave.detach())
            p = predicted.view(-1)
            o = observed_next_wave.detach().view(-1)
            inner = torch.dot(p, o)
            denom = (torch.norm(p) * torch.norm(o)).clamp(min=1e-12)
            loss = 1.0 - inner / denom  # normalized Sagnac delta, differentiable
            g = torch.autograd.grad(loss, self.transition.transition)[0]
        with torch.no_grad():
            self.transition.transition -= lr * g
            self.transition._retract()
        return float(loss.detach())

    @torch.no_grad()
    def apply_creep(self, predicted_wave: torch.Tensor, observed_wave: torch.Tensor, lr: float = 0.01):
        """Deprecated stub retained for API compatibility; use train_transition_step."""
        return float(torch.mean((predicted_wave - observed_wave) ** 2))


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    NB = 64  # small blocks for CPU smoke test
    D = NB * 8

    planner = EFEPlanner(num_blocks=NB, d_model=D).to(device)

    # Two candidate actions with distinct waves
    def mk_wave(seed):
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(NB, 8, generator=g)
        return (w / torch.norm(w, p=2, dim=-1, keepdim=True)).to(device)

    state = mk_wave(1)
    act_a, act_b = mk_wave(2), mk_wave(3)
    boundary = torch.stack([mk_wave(4), mk_wave(5)])

    # Seed the retrieval store with a few attractors (real waves of width d_model)
    attractors = torch.stack([mk_wave(10), mk_wave(11), mk_wave(12)]).view(3, -1).to(device)
    planner.cleanup.store_engrams(attractors)

    scores = planner.score_actions(state, [("A", act_a), ("B", act_b)], boundary)
    for s in scores:
        print(f"  action {s['action']}: EFE={s['efe']:+.4f} "
              f"(pragmatic={s['pragmatic']:.4f}, epistemic={s['epistemic']:.4f})")

    best, pred, table, chosen = planner.select_action(state, [("A", act_a), ("B", act_b)], boundary)
    print(f"selected: {best}")
    assert best in ("A", "B")
    assert all(math.isfinite(s["efe"]) for s in table)
    assert pred.shape == (NB, 8)
    # predicted wave stays on manifold
    norms = torch.norm(pred, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), norms
    print("EFE planner smoke test PASSED")
