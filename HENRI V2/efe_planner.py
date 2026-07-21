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

    def __init__(self, num_blocks: int = 8192, block_dim: int = 8, rank: int = 128):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_dim = block_dim
        self.rank = rank
        self.d = num_blocks * block_dim

        # Global field channel: W [2d, r] reads the full complex fused wave
        # (Re ‖ Im), V [d, r] broadcasts the r-dim global mode back onto the
        # real block grid. This lets the FHRR phase content drive the field
        # while the prediction stays a real, on-manifold wave.
        scale = 1.0 / math.sqrt(2 * self.d)
        self.field_V = nn.Parameter(torch.randn(self.d, rank) * scale)
        self.field_W = nn.Parameter(torch.randn(2 * self.d, rank) * scale)

        # Local residual: per-block near-unitary 8x8 matrices (the gap wiring).
        real = torch.eye(block_dim) + 0.01 * torch.randn(num_blocks, block_dim, block_dim)
        imag = 0.01 * torch.randn(num_blocks, block_dim, block_dim)
        self.block_residual = nn.Parameter(torch.complex(real, imag))

        self._retract()

    @torch.no_grad()
    def _retract(self, residual_only: bool = False):
        """Project parameters toward their manifold constraints.

        residual_only=True skips the field_V QR: the batched EDMD fit stores
        singular-value magnitude in field_V (V·√S), and re-orthonormalizing
        it discards the solved amplitude. The residual-only path is used by
        train_transition_batch's residual refit loop.
        """
        if not residual_only:
            # V -> column-orthonormal (semi-unitary): QR of [d, r] real matrix.
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

        # Global ephaptic field channel: integrate the full complex fused wave
        # (Re ‖ Im, 2d wide) into the r-dim global mode, then broadcast onto
        # the real block grid (d wide). FHRR phase content drives the field.
        fused_flat = torch.cat([fused.real.reshape(-1), fused.imag.reshape(-1)])  # [2d]
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

        # Wire A (pragmatic prior): Hopfield store of waves from historically
        # favorable transitions (valence v > 0). p(o|m) = exp(V(s)) in the
        # FEP formulation — resonance with this store is the prior-preference
        # term of the pragmatic value, warping EFE drift toward verified
        # basins. Real waves, ring-capped like the novelty memory.
        self.preference_store = ContinuousHopfieldCleanup(dim=d_model, beta=8.0)
        self.preference_capacity = 256
        self.beta_pragmatic = 1.0

        # EDMD fit diagnostics (Phase 0: cd82 L2 instability characterization).
        # Populated by train_transition_batch on every fit; read-only record
        # of the solved spectrum and Gram conditioning, no behavior change.
        self.last_edmd_diagnostics = {}

    def update_model_accuracy(self, transition_loss: float):
        """EMA update of the dynamics model's observed error (T4)."""
        self.loss_ema = self.loss_ema_beta * self.loss_ema + (1 - self.loss_ema_beta) * transition_loss
        # Peak tracks the highest error seen (starts at the initial error).
        self.loss_ema_peak = max(self.loss_ema_peak, self.loss_ema)

    def _accuracy_floor(self) -> float:
        """Adaptive exploitation threshold: exploit once the model's error has
        dropped ~10% below the worst error seen in this session."""
        return self.loss_ema_peak - 0.1

    # -- deep consolidation (NL Level 3: field channel persistence) --------

    def field_channel_wave(self) -> torch.Tensor:
        """Pack the transition operator (field_W, field_V, residual) into a
        single wave-shaped tensor for Zone C engram storage."""
        t = self.transition
        return torch.cat([
            t.field_W.detach().reshape(-1).cpu(),
            t.field_V.detach().reshape(-1).cpu(),
            t.block_residual.detach().real.reshape(-1).cpu(),
            t.block_residual.detach().imag.reshape(-1).cpu(),
        ])

    @torch.no_grad()
    def load_field_channel_wave(self, wave: torch.Tensor):
        """Inverse of field_channel_wave: restore the operator from a wave."""
        t = self.transition
        d, r, B, b = t.d, t.rank, t.num_blocks, t.block_dim
        nW, nV, nR = 2 * d * r, d * r, B * b * b
        wave = wave.detach().cpu().float()
        assert wave.numel() >= nW + nV + 2 * nR, (
            f"field channel wave too short: {wave.numel()} < {nW + nV + 2 * nR}")
        dev = t.field_V.device
        t.field_W.copy_(wave[:nW].reshape(2 * d, r).to(dev))
        t.field_V.copy_(wave[nW:nW + nV].reshape(d, r).to(dev))
        re = wave[nW + nV:nW + nV + nR].reshape(B, b, b)
        im = wave[nW + nV + nR:nW + nV + 2 * nR].reshape(B, b, b)
        t.block_residual.copy_(torch.complex(re, im).to(dev))

    # -- value terms ------------------------------------------------------

    def pragmatic_value(self, predicted_wave: torch.Tensor, boundary_axioms: torch.Tensor) -> torch.Tensor:
        """Pragmatic term of the EFE: surprise minus prior-preference resonance.

        FEP decomposition (bank-conformant): the pragmatic value of a policy
        is the KL divergence between predicted outcomes and prior preferences
        p(o|m) = exp(V(s)). Implemented as

            pragmatic = min_a sagnac_delta(predicted, axiom_a)
                        - beta_pragmatic * max_resonance(predicted, prefs)

        The first term is expected surprise against boundary axioms [0, 2];
        the second is geometric resonance with the preference store of
        historically favorable transition waves [0, 1]. Resonance with a
        verified-successful outcome LOWERS the score (argmin wins), pulling
        the drift toward favorable basins without touching the epistemic
        term. boundary_axioms: [N, num_blocks, 8]. Returns scalar.
        """
        p = predicted_wave.view(-1)
        p = p / (torch.norm(p) + 1e-12)
        deltas = []
        for axiom in boundary_axioms:
            a = axiom.view(-1)
            a = a / (torch.norm(a) + 1e-12)
            inner = torch.dot(p, a)  # real waves: plain inner product
            deltas.append(1.0 - inner)
        surprise = torch.stack(deltas).min()  # closest axiom governs

        resonance = torch.zeros((), device=predicted_wave.device)
        if self.preference_store.num_engrams() > 0:
            sim = p @ self.preference_store.engrams.T
            resonance = sim.max().clamp(min=0.0)
        return surprise - self.beta_pragmatic * resonance

    def register_preference(self, predicted_wave: torch.Tensor):
        """Consolidate a favorable transition's predicted wave into the
        preference store (called by the orchestrator when valence > 0).
        Ring-capped at preference_capacity, oldest dropped first."""
        flat = predicted_wave.view(-1)
        flat = flat / (torch.norm(flat) + 1e-12)
        self.preference_store.store_engrams(flat.unsqueeze(0))
        if self.preference_store.num_engrams() > self.preference_capacity:
            self.preference_store.engrams = self.preference_store.engrams[-self.preference_capacity:]

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

    # -- transition training (single-step + batched EDMD) -----------------

    def train_transition_step(
        self,
        state_wave: torch.Tensor,
        action_wave: torch.Tensor,
        observed_next_wave: torch.Tensor,
        lr: float = 0.05,
        surprise_modulate: bool = True,
        valence: float = 0.0,
    ) -> float:
        """
        Online latent-space dynamics learning (T1 + T2), fast NL level.

        Trains the transition operator on the Sagnac signal itself: the loss
        is the normalized coherence delta between the model's prediction and
        the observed next wave — the exact quantity the swarm measures.

            predicted = transition(state ⊗ action)
            loss      = 1 - Re(<predicted, observed>) / (||predicted|| ||observed||)

        The (state, action) pair is the EXECUTED action's canonical wave, so
        the model learns action-conditioned dynamics rather than an
        action-averaged blur (T2). After the Wirtinger step the transition
        matrices are re-retracted to unitarity.

        Surprise modulation (Titans fast-memory analog): the normalized
        Sagnac delta IS an associative-surprise signal, so the effective
        learning rate is scaled by it — lr_eff = lr * (0.25 + delta/2),
        bounded to [0.25x, 1.25x] lr. High-surprise transitions get
        high-plasticity updates; already-predictable transitions barely
        move the weights (matching the paper's gradient-magnitude gate).

        Wire B (valence precision gate, dopaminergic polarity): valence
        nu in [-1, 1] scales precision (inverse temperature) of the
        prediction error. Success (nu > 0) cools the thermostat — the
        update crystallizes the verified trajectory with a DAMPED rate.
        Failure (nu < 0) keeps the surprise-gated rate plastic (the system
        must NOT consolidate the failed trajectory, but per the Titans
        saturation warning it must NOT freeze either — the heat lives in
        the swarm's Langevin schedule, which receives the same valence).
        Neutral (nu = 0) leaves the surprise gate untouched.

            lr_eff = lr * (0.25 + delta/2) * 1/(1 + nu)   for nu > 0
                   = lr * (0.25 + delta/2) * (1 + nu)^2   for nu < 0
        clamped to (0, 1.25x lr]. The (1+nu)^2 failure branch damps
        consolidation quadratically without a hard zero-halt.

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
            if surprise_modulate:
                delta = float(loss.detach())
                lr_eff = lr * min(1.25, 0.25 + 0.5 * delta)
            else:
                lr_eff = lr
            if valence > 0.0:
                lr_eff /= (1.0 + valence)       # crystallize: damped update
            elif valence < 0.0:
                lr_eff *= (1.0 + valence) ** 2  # failed trajectory: damp
                lr_eff = max(lr_eff, 1e-4 * lr)  # never fully freeze
            self.transition.field_V -= lr_eff * gV
            self.transition.field_W -= lr_eff * gW
            self.transition.block_residual -= lr_eff * gR
            self.transition._retract()
        # T4: track model accuracy so the exploration gate tightens as we learn.
        self.update_model_accuracy(float(loss.detach()))
        return float(loss.detach())

    @torch.no_grad()
    def train_transition_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        observed_nexts: torch.Tensor,
        iters: int = 3,
        ridge: float = 1e-4,
        update_residual: bool = True,
    ) -> float:
        """
        Batched EDMD-style transition training over a Koopman dictionary.

        Instead of one stochastic SGLD step per observed transition, this
        collects a BATCH of (state, action) -> next-wave triples, lifts each
        through the planner's fixed nonlinear dictionary

            Psi(s, a) = concat(Re(bind), Im(bind))  in R^{2d}

        (FHRR circular-convolution binding — the Koopman eigenfunctions of
        this architecture), and solves for the global field channel by
        regularized least squares (extended DMD / kernel ridge regression,
        solved in the N-dim sample dual — never forming the 2d x 2d primal
        Gram, which is 64 GiB at production scale):

            K = Psi_batch Psi_batch^T + ridge * N * I      [N x N]
            W = Psi_batch^T K^{-1} Y_batch                 [2d x d]

        The field channel is then set as the rank-r SVD truncation
        field_W @ field_V^T ~= W with V column-orthonormalized by QR —
        exactly the EDMD Galerkin projection of the Koopman operator onto
        the span of the observed dictionary outputs. The per-block unitary
        residual is optionally refit afterward with a few projected-gradient
        steps on the batch Sagnac loss (kept on the Stiefel manifold by
        retraction).

        All tensors are stacked [N, num_blocks, 8] real Clifford waves.
        Returns the mean pre-fit batch Sagnac loss (the quantity minimized).
        """
        N = states.shape[0]
        device = self.transition.field_V.device
        states = states.detach().to(device)
        actions = actions.detach().to(device)
        observed_nexts = observed_nexts.detach().to(device)

        preds = torch.stack(
            [self.transition(states[i], actions[i]) for i in range(N)]
        )
        p = preds.reshape(N, -1)
        o = observed_nexts.reshape(N, -1)
        pre_losses = 1.0 - (
            (p * o).sum(-1)
            / (p.norm(dim=-1) * o.norm(dim=-1)).clamp(min=1e-12)
        )
        pre_loss = float(pre_losses.mean())

        # Lift through the Koopman dictionary: fused intent waves.
        fused = torch.stack(
            [self.transition.bind(states[i], actions[i]) for i in range(N)]
        )  # [N, blocks, 8] complex
        d = self.transition.d
        X = torch.cat(
            [fused.real.reshape(N, d), fused.imag.reshape(N, d)], dim=-1
        )  # [N, 2d]
        Y = observed_nexts.reshape(N, d)  # [N, d]

        # EDMD / ridge least-squares for the linear readout on the dictionary.
        # DUAL (Woodbury) form: W = X^T (X X^T + ridge*N*I)^{-1} Y = X^T C.
        # The full W is [2d, d] = 32 GiB at d=65536 — never form it. Since
        # W = X^T (Uc Sc Vc^T), its rank-r truncation follows from the thin
        # SVD of the N x d coefficient matrix C = Uc Sc Vc^T alone:
        #   field_V = Vc[:, :r]           (orthonormal right singular vecs)
        #   field_W = X^T Uc[:, :r] Sc[:r]  (one N-row contraction, [2d, r])
        # Memory: O(N*d) for C plus O(2d*r) for the product — ~130 MB total
        # at production scale instead of 32 GiB.
        K = X @ X.T + ridge * N * torch.eye(N, device=device, dtype=X.dtype)
        # Gram conditioning BEFORE any jitter escalation: the ratio of the
        # largest to smallest eigenvalue of the raw Gram. Near-duplicate
        # buffer rows (RESET loops) collapse the small eigenvalues; the log10
        # condition number is the instability leading indicator for cd82.
        gram_eigs = torch.linalg.eigvalsh(K)
        gram_cond = float(gram_eigs.max() / gram_eigs.min().clamp(min=1e-30))
        # fp32 Gram matrices from near-duplicate buffer rows (RESET loops)
        # can be rank-deficient past the ridge lift; escalate jitter on
        # failure instead of dying. K is N x N (<= 4096), retries are free.
        C = None
        jitter_tier = -1
        for jitter_mult in (1.0, 10.0, 100.0, 1000.0):
            jitter_tier += 1
            try:
                L = torch.linalg.cholesky(
                    K + (jitter_mult - 1.0) * ridge * N
                    * torch.eye(N, device=device, dtype=X.dtype))
                C = torch.cholesky_solve(Y, L)  # [N, d]
                break
            except torch._C._LinAlgError:
                continue
        if C is not None:
            Uc, Sc, Vch = torch.linalg.svd(C, full_matrices=False)
            # Available rank is min(N, d) — with a small buffer (N < r) the
            # truncated operator is genuinely rank-N, not rank-r. Keep the V
            # side at its solved width and zero the unused field_V columns;
            # field_W matches column-for-column so the product is exact.
            k = min(self.transition.rank, Sc.numel())
            # Rank-k truncation of W = X^T Uc Sc Vc^T.
            self.transition.field_V.zero_()
            self.transition.field_V[:, :k].copy_(Vch[:k, :].T.contiguous())
            self.transition.field_W.zero_()
            self.transition.field_W[:, :k].copy_((X.T @ Uc[:, :k]) * Sc[:k])

        # Spectral diagnostics: the solved coefficient spectrum (Sc near 1
        # marks near-invariant modes — candidate axioms, Phase 1), Gram
        # conditioning, jitter tier, and the pre/post-fit loss delta — the
        # cd82 instability leading indicators.
        if C is not None:
            self.last_edmd_diagnostics = {
                "n_samples": int(N),
                "gram_cond_log10": round(math.log10(max(gram_cond, 1e-30)), 3),
                "jitter_tier": jitter_tier,
                "cholesky_failed": False,
                "sc_top8": [round(float(s), 4) for s in Sc[:8]],
                "sc_rank": int((Sc > 1e-6 * Sc.max()).sum()),
                "pre_loss": round(pre_loss, 6),
            }
        else:
            self.last_edmd_diagnostics = {
                "n_samples": int(N),
                "gram_cond_log10": round(math.log10(max(gram_cond, 1e-30)), 3),
                "jitter_tier": jitter_tier,
                "cholesky_failed": True,
                "pre_loss": round(pre_loss, 6),
            }

        # Residual: absorb what the field channel cannot, staying per-block
        # unitary via projected gradient + retraction. The step must be small:
        # the field channel is already solved to near-optimality on the batch,
        # and an aggressive residual refit drags the coupled prediction off
        # the manifold faster than retraction can restore it (verified: lr
        # 0.05 for 3 steps degraded the batch loss from 0.31 back to 0.89).
        if update_residual:
            for _ in range(iters):
                with torch.enable_grad():
                    loss = self._batch_sagnac_loss(states, actions, observed_nexts)
                    (gR,) = torch.autograd.grad(loss, [self.transition.block_residual])
                with torch.no_grad():
                    self.transition.block_residual -= 0.005 * gR
                    # Residual-only retraction: a full _retract() would QR
                    # field_V back to orthonormal columns and discard the
                    # singular-value amplitudes the EDMD solve just stored.
                    self.transition._retract(residual_only=True)

        post_loss = self._batch_sagnac_loss(states, actions, observed_nexts)
        self.last_edmd_diagnostics["post_loss"] = round(float(post_loss), 6)
        self.update_model_accuracy(pre_loss)
        return pre_loss

    def _batch_sagnac_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        observed_nexts: torch.Tensor,
    ) -> torch.Tensor:
        """Mean normalized Sagnac delta of predictions over a batch."""
        preds = torch.stack(
            [self.transition(states[i], actions[i]) for i in range(states.shape[0])]
        )
        p = preds.reshape(states.shape[0], -1)
        o = observed_nexts.reshape(states.shape[0], -1)
        return (
            1.0 - (p * o).sum(-1) / (p.norm(dim=-1) * o.norm(dim=-1)).clamp(min=1e-12)
        ).mean()

    @torch.no_grad()
    def apply_creep(self, predicted_wave: torch.Tensor, observed_wave: torch.Tensor, lr: float = 0.01):
        """Deprecated stub retained for API compatibility; use train_transition_step."""
        return float(torch.mean((predicted_wave - observed_wave) ** 2))

