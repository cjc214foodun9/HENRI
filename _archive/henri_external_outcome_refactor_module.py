import math
from typing import Sequence, Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F

# ==============================================================================
# P0: EXTERNAL-OUTCOME ACTION EFFECT POSTERIOR
# ==============================================================================

class ActionEffectPosterior:
    """
    Tracks action-conditioned external state change probabilities via Beta-Bernoulli posteriors.
    
    Prior: Theta_a ~ Beta(alpha_0=1, beta_0=1)
    Updates:
      - changed=True  -> alpha_a += 1
      - changed=False -> beta_a += 1 (if valid=True)
    """
    
    # Exact analytical normalization factor for Beta(1,1) Bernoulli observation EIG
    # I(1,1) = H_B(0.5) - [digamma(3) - digamma(2)] = ln(2) - (1.5 - 1.0) = ln(2) - 0.5 ≈ 0.1931471805599453
    I_1_1_NORM: float = 0.1931471805599453

    def __init__(self, action_ids: Optional[Sequence[int]] = None):
        self.alpha: Dict[int, float] = {}
        self.beta: Dict[int, float] = {}
        if action_ids is not None:
            self.reset(action_ids)

    def reset(self, action_ids: Sequence[int]) -> None:
        """Reset posteriors for a new environment instance."""
        self.alpha = {int(a): 1.0 for a in action_ids}
        self.beta = {int(a): 1.0 for a in action_ids}

    def observe(self, action_id: int, *, changed: bool, valid: bool = True) -> None:
        """
        Update Beta posterior based on environmental feedback.
        Execution only happens after environmental step.
        """
        a_id = int(action_id)
        if a_id not in self.alpha:
            self.alpha[a_id] = 1.0
            self.beta[a_id] = 1.0

        if not valid:
            return

        if changed:
            self.alpha[a_id] += 1.0
        else:
            self.beta[a_id] += 1.0

    def raw_information_gain(self, action_id: int) -> float:
        """
        Calculates exact analytical Expected Information Gain (EIG) for one future Bernoulli trial:
          I_a = H_B(q_a) - [ psi(a + b + 1) - (a*psi(a+1) + b*psi(b+1))/(a + b) ]
        where q_a = alpha / (alpha + beta) and psi is the digamma function.
        """
        a_id = int(action_id)
        a = self.alpha.get(a_id, 1.0)
        b = self.beta.get(a_id, 1.0)

        total = a + b
        q = a / total

        # Binary Entropy H_B(q) in nats
        eps = 1e-15
        q_clamped = max(eps, min(1.0 - eps, q))
        h_b = -(q_clamped * math.log(q_clamped) + (1.0 - q_clamped) * math.log(1.0 - q_clamped))

        # Expected posterior entropy via Digamma (psi)
        psi_total_plus_1 = torch.special.digamma(torch.tensor(total + 1.0, dtype=torch.float64)).item()
        psi_a_plus_1 = torch.special.digamma(torch.tensor(a + 1.0, dtype=torch.float64)).item()
        psi_b_plus_1 = torch.special.digamma(torch.tensor(b + 1.0, dtype=torch.float64)).item()

        exp_post_entropy = psi_total_plus_1 - (a * psi_a_plus_1 + b * psi_b_plus_1) / total
        eig = h_b - exp_post_entropy
        return max(0.0, float(eig))

    def information_gain(self, action_id: int) -> float:
        """
        Normalized Information Gain I_a^norm in [0, 1].
        """
        raw_eig = self.raw_information_gain(action_id)
        norm_eig = raw_eig / self.I_1_1_NORM
        return max(0.0, min(1.0, norm_eig))


# ==============================================================================
# P0: EXTERNAL TASK PREFERENCE STORE & EFE PLANNER EXTENSION
# ==============================================================================

class TaskPreferenceStore:
    """
    Isolated store for waves associated with externally verified success (level completion or WIN).
    Decoupled from internal PROGRESS_VALENCE to prevent solipsistic loop contamination.
    """
    def __init__(self, dim: int = 65536):
        self.dim = dim
        self.verified_waves: List[torch.Tensor] = []

    def reset(self) -> None:
        self.verified_waves.clear()

    def add_verified_wave(self, wave: torch.Tensor) -> None:
        """Store externally verified next-state wave."""
        if wave is not None:
            # Store normalized copy
            norm_wave = F.normalize(wave.detach().clone().to(torch.float32), dim=-1)
            self.verified_waves.append(norm_wave)

    def task_resonance(self, predicted_wave: torch.Tensor) -> float:
        """
        Calculates cosine resonance R_task with stored success waves.
        Returns 0.0 if store is empty.
        """
        if not self.verified_waves or predicted_wave is None:
            return 0.0

        p_norm = F.normalize(predicted_wave.to(torch.float32), dim=-1)
        resonances = [torch.sum(p_norm * w).item() for w in self.verified_waves]
        return max(resonances)


class EFEPlannerWithExternalOutcome:
    """
    Extended Active Inference Planner incorporating External-Outcome EFE Channel.
    """
    def __init__(
        self,
        dim: int = 65536,
        external_outcome_efe: bool = False,
        w_eig: float = 0.25,
        w_task: float = 1.0,
    ):
        self.dim = dim
        self.external_outcome_efe = external_outcome_efe
        self.w_eig = min(0.5, max(0.0, w_eig))
        self.w_task = min(2.0, max(0.0, w_task))

        self.posterior = ActionEffectPosterior()
        self.task_store = TaskPreferenceStore(dim=dim)

    def reset_environment_context(self, valid_action_ids: Sequence[int]) -> None:
        """Reset posterior and task store on new environment instantiation."""
        self.posterior.reset(valid_action_ids)
        self.task_store.reset()

    def observe_external_outcome(
        self,
        action_id: int,
        *,
        frame_changed: bool,
        task_progressed: bool,
        observed_next_wave: Optional[torch.Tensor],
        valid: bool = True,
    ) -> None:
        """
        Updates action posterior and task preference store AFTER environment execution.
        """
        if not self.external_outcome_efe:
            return

        # 1. Update Beta Posterior
        self.posterior.observe(action_id, changed=frame_changed, valid=valid)

        # 2. Add to Task Store only on verified level completion or WIN
        if task_progressed and observed_next_wave is not None:
            self.task_store.add_verified_wave(observed_next_wave)

    def score_candidate_action(
        self,
        action_id: int,
        base_g_score: float,
        predicted_next_wave: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Applies external EFE correction:
          G'(a) = G_current(a) - w_EIG * I_a^norm - w_task * R_task(psi_hat_{t+1}^a)
        """
        if not self.external_outcome_efe:
            return base_g_score

        i_norm = self.posterior.information_gain(action_id)
        r_task = self.task_store.task_resonance(predicted_next_wave) if predicted_next_wave is not None else 0.0

        external_contribution = (self.w_eig * i_norm) + (self.w_task * r_task)
        g_prime = base_g_score - external_contribution
        return g_prime


# ==============================================================================
# P1: RESIDUAL ERROR-TARGETED LANGEVIN MASK
# ==============================================================================

def build_prediction_error_mask(
    observed_wave: torch.Tensor,
    predicted_prior: torch.Tensor,
    *,
    cap: float = 4.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Computes bounded diagonal stochastic temperature modulation mask:
      e = psi_observed - psi_hat_prior
      m_i = min(m_max, |e_i| / sqrt( (1/D) * sum_j(e_j^2) + eps ))
    
    Applied only to stochastic SGLD term: eta_i' = m_i * eta_i
    """
    assert observed_wave.shape == predicted_prior.shape, "Wave dimension mismatch in residual mask build."
    
    # 1. Compute elementwise residual
    e = observed_wave - predicted_prior

    # 2. Root Mean Square Error scaling denominator
    rmse = torch.sqrt(torch.mean(e ** 2) + eps)

    # 3. Calculate dimension-wise relative error mask
    m = torch.abs(e) / rmse

    # 4. Cap upper bound
    m_capped = torch.clamp(m, max=cap)
    return m_capped


# ==============================================================================
# DENSE VSA RECONSTRUCTION KILL-TEST HARNESS
# ==============================================================================

def run_vsa_reconstruction_kill_test(
    m_superpositions: Sequence[int] = (64, 256, 900),
    num_trials: int = 32,
    dim: int = 65536,
    num_colors: int = 16,
    device: str = "cpu",
) -> Dict[int, float]:
    """
    Production-scale kill-test evaluating full-grid superposition crosstalk noise floor.
    Pre-registered verdicts:
      - Acc @ M=900 >= 99%: Reject Slot refactor
      - Acc @ M=900 < 95% or drop from M=256 to M=900 > 5%: Authorize Slot Prototype
    """
    results = {}
    torch.manual_seed(42)

    # Generate orthogonal role hypervectors and color filler hypervectors
    roles = F.normalize(torch.randn(max(m_superpositions), dim, device=device), dim=-1)
    colors = F.normalize(torch.randn(num_colors, dim, device=device), dim=-1)

    for m in m_superpositions:
        correct_retrievals = 0
        total_retrievals = m * num_trials

        for trial in range(num_trials):
            # Sample random grid color assignments
            color_indices = torch.randint(0, num_colors, (m,), device=device)
            
            # Superpose bound pairs: S = sum_i (Role_i * Color_i)
            bound_pairs = roles[:m] * colors[color_indices]
            s_wave = torch.sum(bound_pairs, dim=0, keepdim=True)

            # Unbind roles and decode colors: Color_hat_i = S * Role_i
            unbound_queries = s_wave * roles[:m] # [M, D]
            
            # Cosine similarity against 16 color bases
            unbound_norm = F.normalize(unbound_queries, dim=-1)
            similarities = torch.matmul(unbound_norm, colors.T) # [M, 16]
            decoded_colors = torch.argmax(similarities, dim=-1)

            correct_retrievals += (decoded_colors == color_indices).sum().item()

        accuracy = correct_retrievals / total_retrievals
        results[m] = accuracy

    return results


# ==============================================================================
# RED TEST SUITE
# ==============================================================================

def run_red_tests():
    print("Executing HENRI Refactor RED Test Suite...")

    # Test 1: EIG Exactness & Bounds
    post = ActionEffectPosterior()
    post.reset([1, 2])
    raw_eig_1_1 = post.raw_information_gain(1)
    assert abs(raw_eig_1_1 - ActionEffectPosterior.I_1_1_NORM) < 1e-6, f"EIG exactness failed: {raw_eig_1_1}"
    assert abs(post.information_gain(1) - 1.0) < 1e-6, "Normalized EIG at (1,1) must be 1.0"

    # Digamma decay at high counts: I(100, 100) approx 0.00249375 nats
    post.alpha[2] = 100.0
    post.beta[2] = 100.0
    raw_eig_100_100 = post.raw_information_gain(2)
    assert abs(raw_eig_100_100 - 0.00249375) < 1e-4, f"EIG high-count decay mismatch: {raw_eig_100_100}"
    assert 0.0 <= post.information_gain(2) <= 1.0, "EIG bound violated"

    # Test 2: External Counterfactual Ranking Shift
    planner = EFEPlannerWithExternalOutcome(external_outcome_efe=True, w_eig=0.5, w_task=1.0)
    planner.reset_environment_context([1, 2])
    
    # Base scores equal
    score_a1_init = planner.score_candidate_action(1, base_g_score=10.0)
    score_a2_init = planner.score_candidate_action(2, base_g_score=10.0)
    assert score_a1_init == score_a2_init, "Initial candidate ranking should be equal"

    # Observe change on Action 1
    planner.observe_external_outcome(1, frame_changed=True, task_progressed=False, observed_next_wave=None)
    score_a1_post = planner.score_candidate_action(1, base_g_score=10.0)
    score_a2_post = planner.score_candidate_action(2, base_g_score=10.0)
    assert score_a1_post < score_a2_post, "EIG reduction must prioritize Action 1 (lower G' score)"

    # Test 3: Residual Mask Locality & Bounds
    obs = torch.tensor([1.0, 0.0, -1.0, 0.0])
    prior = torch.tensor([1.0, 0.0, 0.0, 0.0])
    mask = build_prediction_error_mask(obs, prior, cap=4.0)
    assert mask[0].item() == 0.0 and mask[1].item() == 0.0, "Zero residual must yield zero mask value"
    assert torch.all(torch.isfinite(mask)), "Mask must contain finite values"

    print("All RED tests passed successfully.")


if __name__ == "__main__":
    run_red_tests()
    print("\nRunning VSA Reconstruction Kill-Test Benchmark...")
    kill_test_results = run_vsa_reconstruction_kill_test(m_superpositions=[64, 256, 900], num_trials=16)
    for M, acc in kill_test_results.items():
        print(f"  M = {M:3d} Cell Unbinding Accuracy: {acc * 100.6:.2f}%")
```e-o-f