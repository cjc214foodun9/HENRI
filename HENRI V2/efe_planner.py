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

    Low-rank coupled operator (ephaptic field + gap-junction residual):

        predicted = V (W^dag fused)   [global field channel, rank r]
                  + R_block fused     [local block-diagonal residual]

    The global field channel integrates the whole wave into an r-dim
    bottleneck (W^dag: r x d) and broadcasts it back (V: d x r), giving every
    block access to every other block's state — the cross-block coupling a
    pure block-diagonal operator structurally cannot represent. The local
    residual R_block preserves the per-block unitary wiring.

    Both channels keep Stiefel/unitary structure: V is column-semi-unitary,
    R_block is per-block unitary, so predictions stay near the manifold.
    Sample complexity to identify the field channel is O(r), independent of
    d — learnable online within a single env episode.
    """

    def __init__(self, num_blocks: int = 8192, block_dim: int = 8, rank: int = 64):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.rank = rank
        self.d = num_blocks * block_dim

        # Global field channel: V [d, r] and W [d, r], near-semi-unitary.
        scale = 1.0 / math.sqrt(self.d)
        self.field_V = nn.Parameter(torch.randn(self.d, rank) * scale)
        self.field_W = nn.Parameter(torch.randn(self.d, rank) * scale)

        # Local residual: per-block near-unitary 8x8 matrices (the gap wiring).
        real = torch.eye(block_dim) + 0.01 * torch.randn(num_blocks, block_dim, block_dim)
        imag = 0.01 * torch.randn(num_blocks, block_dim, block_dim)
        self.block_residual = nn.Parameter(torch.complex(real, imag))

        self._retract()

    @torch.no_grad()
    def _retract(self):
        """Project parameters toward their manifold constraints."""
        # V -> column-orthonormal (semi-unitary): QR of the [d, r] real matrix.
        Qv, _ = torch.linalg.qr(self.field_V, mode="reduced")
        self.field_V.copy_(Qv)
        # Residual -> per-block unitary.
        Qb, _ = torch.linalg.qr(self.block_residual)
        self.block_residual.copy_(Qb)

    # Backward-compat alias used by EFEPlanner tests / training hooks.
    @property
    def transition(self):
        return self.block_residual

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

        # Local gap-junction residual: per-block unitary transform.
        local = torch.einsum('bij,bj->bi', self.block_residual, fused)  # complex [B, 8]

        # Global ephaptic field channel: integrate then broadcast (real part
        # of the fused wave drives the field; field adds a real global mode).
        fused_flat = fused.real.reshape(-1)  # [d]
        field_mode = self.field_W.T @ fused_flat          # [r]
        field = (self.field_V @ field_mode).view(self.num_blocks, self.block_dim)

        predicted_real = local.real + field
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

        # T4: model-accuracy tracking. EMA of the transition loss; the
        # exploration gate keys off how wrong the dynamics model is, so
        # exploitation kicks in as the model improves (not just on spread).
        self.loss_ema = 1.0  # start fully uncertain
        self.loss_ema_beta = 0.95
        # Slow tracker of the worst (initial) loss for the adaptive floor.
        self.loss_ema_peak = 1.0

        # Epistemic novelty memory: a small Hopfield store of recently
        # predicted outcome waves. Actions whose predictions land near an
        # already-visited outcome yield less information (already explored).
        self.novelty_memory = ContinuousHopfieldCleanup(dim=d_model, beta=8.0)
        self.novelty_capacity = 256

    def update_model_accuracy(self, transition_loss: float):
        """EMA update of the dynamics model's observed error (T4)."""
        self.loss_ema = self.loss_ema_beta * self.loss_ema + (1 - self.loss_ema_beta) * transition_loss
        # Peak tracks the highest error seen (starts at the initial error).
        self.loss_ema_peak = max(self.loss_ema_peak, self.loss_ema)

    def _accuracy_floor(self) -> float:
        """Adaptive exploitation threshold: exploit once the model's error has
        dropped ~10% below the worst error seen in this session."""
        return self.loss_ema_peak - 0.1

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
        Information gain with novelty discounting.

        Two terms:
          (a) Retrieval entropy: uncertainty over which attractor the
              prediction lands in (informative when spread).
          (b) Novelty bonus: how far the prediction is from the novelty
              memory of already-visited outcomes. Repeated predictions
              (same action, same outcome) are discounted toward zero, so
              exploration stops rewarding loops like RESET-spam.
        Returns scalar >= 0.
        """
        flat = predicted_wave.view(-1)
        flat = flat / (torch.norm(flat) + 1e-12)

        entropy = torch.tensor(0.0, device=predicted_wave.device)
        if self.cleanup.num_engrams() > 1:
            # Soft-temperature retrieval for a meaningful entropy readout:
            # the cleanup store's beta (sqrt d) is tuned for hard snapping and
            # collapses the distribution to one-hot; epistemic uncertainty
            # needs a spread, so recompute weights at a fixed soft temperature.
            r = flat / (torch.norm(flat) + 1e-12)
            sim = r @ self.cleanup.engrams.T
            w = torch.softmax(sim / 0.1, dim=-1).clamp(min=1e-12)
            entropy = -(w * torch.log(w)).sum()

        novelty = torch.tensor(1.0, device=predicted_wave.device)
        if self.novelty_memory.num_engrams() > 0:
            # Distance to nearest remembered outcome: 1 - max raw cosine sim
            r = flat / (torch.norm(flat) + 1e-12)
            sim = (r @ self.novelty_memory.engrams.T).max()
            novelty = (1.0 - sim).clamp(min=0.0)

        return entropy * novelty

    def remember_outcome(self, predicted_wave: torch.Tensor):
        """Record a visited outcome wave so future identical predictions are
        discounted as non-novel. Caps the memory at novelty_capacity by
        rebuilding from the most recent entries (ring behavior via clear +
        restore is handled by the Hopfield store's append)."""
        flat = predicted_wave.view(-1)
        flat = flat / (torch.norm(flat) + 1e-12)
        self.novelty_memory.store_engrams(flat.unsqueeze(0))
        # Enforce capacity: drop oldest by clearing and keeping the tail
        if self.novelty_memory.num_engrams() > self.novelty_capacity:
            self.novelty_memory.engrams = self.novelty_memory.engrams[-self.novelty_capacity:]

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

        # T4 accuracy-gated exploration: explore iff the dynamics model is
        # still too inaccurate to trust its min-EFE ranking. The floor is
        # adaptive: exploit once the model has improved ~10% from its initial
        # error (tracked as a slow EMA of the worst-seen loss), so the gate
        # is reachable during a session instead of demanding an absolute
        # accuracy the operator may take thousands of steps to reach.
        if explore_threshold is not None:
            accuracy_floor = explore_threshold
        else:
            accuracy_floor = self._accuracy_floor()
        if self.loss_ema > accuracy_floor and len(results) > 1:
            epistemic_best = max(results, key=lambda r: r["epistemic"])
            best = dict(epistemic_best, explored=True)
        else:
            best = dict(best, explored=False)
        explore_threshold = accuracy_floor

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
            params = [self.transition.field_V, self.transition.field_W,
                      self.transition.block_residual]
            gV, gW, gR = torch.autograd.grad(loss, params)
        with torch.no_grad():
            self.transition.field_V -= lr * gV
            self.transition.field_W -= lr * gW
            self.transition.block_residual -= lr * gR
            self.transition._retract()
        # T4: track model accuracy so the exploration gate tightens as we learn.
        self.update_model_accuracy(float(loss.detach()))
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
