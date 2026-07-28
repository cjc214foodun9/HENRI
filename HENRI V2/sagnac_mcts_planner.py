"""
Sagnac-Guided EFE-MCTS Planner for Project HENRI V2.

Combines Expected Free Energy (EFE), PUCT prior, and Sagnac Homodyne Physical Vetoing
for zero-entropy branch pruning during programmatic search.

Branches violating Dirichlet boundary axioms (Delta_Sagnac > tau_veto) are physically
annihilated, preventing local attractor traps and search landscape flattening.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class MCTSNode:
    """Node in the Sagnac-Guided EFE-MCTS search tree."""

    def __init__(self, state_wave: torch.Tensor, prior: float = 1.0):
        self.state_wave = state_wave
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.children: Dict[int, "MCTSNode"] = {}
        self.sagnac_deltas: Dict[int, float] = {}

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count


class SagnacMCTSPlanner(nn.Module):
    """
    Sagnac-Guided Monte Carlo Tree Search Planner.
    Prunes search branches using the physical Sagnac homodyne delta.
    """

    def __init__(
        self,
        d_model: int = 4096,
        num_blocks: int = 512,
        num_morphisms: int = 8,
        tau_veto: float = 0.70,
        lambda_sagnac: float = 5.0,
        c_puct: float = 1.414,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_morphisms = num_morphisms
        self.tau_veto = tau_veto
        self.lambda_sagnac = lambda_sagnac
        self.c_puct = c_puct

        # Precompute quasi-orthogonal qFHRR key waves for morphisms
        g = torch.Generator(device="cpu").manual_seed(123)
        keys = torch.randn(num_morphisms, num_blocks, 8, generator=g)
        self.register_buffer("morphism_keys", F.normalize(keys, p=2, dim=-1))

    def compute_sagnac_delta(self, child_wave: torch.Tensor, boundary_axiom: torch.Tensor) -> float:
        """
        Calculates Sagnac homodyne phase obstruction between child wave and Dirichlet boundary axiom.
        Delta_Sagnac in [0, 1].
        """
        c_flat = child_wave.view(-1)
        b_flat = boundary_axiom.view(-1)
        cos_sim = float(F.cosine_similarity(c_flat.unsqueeze(0), b_flat.unsqueeze(0)).item())
        sagnac_delta = 1.0 - abs(cos_sim)
        return float(np_clip(sagnac_delta, 0.0, 1.0))

    def select_child(self, node: MCTSNode, boundary_axiom: torch.Tensor) -> Tuple[int, MCTSNode]:
        """
        Selects next morphism branch maximizing PUCT + Sagnac Veto score.
        Physically prunes branches exceeding tau_veto.
        """
        best_score = -1e9
        best_morphism = 0

        for m_idx in range(self.num_morphisms):
            if m_idx in node.children:
                child = node.children[m_idx]
                s_delta = node.sagnac_deltas.get(m_idx, 0.0)
            else:
                # Unexpanded node: compute predictive child wave
                m_key = self.morphism_keys[m_idx].to(node.state_wave.device)
                child_wave = F.normalize(node.state_wave + 0.3 * m_key, p=2, dim=-1)
                child = MCTSNode(state_wave=child_wave, prior=1.0 / self.num_morphisms)
                s_delta = self.compute_sagnac_delta(child_wave, boundary_axiom)
                node.children[m_idx] = child
                node.sagnac_deltas[m_idx] = s_delta

            # PHYSICAL VETO PRUNING GATE
            if s_delta > self.tau_veto:
                score = -1e9  # Branch physically vetoed
            else:
                q_val = child.value
                puct = self.c_puct * child.prior * (math.sqrt(node.visit_count + 1e-8) / (1 + child.visit_count))
                veto_penalty = self.lambda_sagnac * s_delta
                score = q_val + puct - veto_penalty

            if score > best_score:
                best_score = score
                best_morphism = m_idx

        return best_morphism, node.children[best_morphism]

    def search_rollout(
        self, root_wave: torch.Tensor, boundary_axiom: torch.Tensor, num_simulations: int = 20
    ) -> Dict[str, float]:
        """Executes N simulations of Sagnac-Guided MCTS search."""
        root = MCTSNode(state_wave=root_wave)
        pruned_branches = 0

        for _ in range(num_simulations):
            m_idx, child = self.select_child(root, boundary_axiom)
            s_delta = root.sagnac_deltas[m_idx]

            if s_delta > self.tau_veto:
                pruned_branches += 1
                value = -1.0
            else:
                # Evaluate EFE value (negative Sagnac delta)
                value = 1.0 - s_delta

            # Backpropagate visit counts and values
            root.visit_count += 1
            root.total_value += value
            child.visit_count += 1
            child.total_value += value

        pruning_efficiency = (pruned_branches / max(1, num_simulations)) * 100.0
        return {
            "num_simulations": num_simulations,
            "pruned_branches": pruned_branches,
            "pruning_efficiency_pct": pruning_efficiency,
            "root_visits": root.visit_count,
        }


def np_clip(v: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, v))
