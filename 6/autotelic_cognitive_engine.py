import torch
import torch.nn as torch_nn
import torch.nn.functional as F
import collections
import numpy as np

class VygotskianImagination(torch_nn.Module):
    """
    Vygotskian Language Internalization.
    Internalizes linguistic primitives (e.g., physical concepts like "amplify", 
    "orthogonal", "phase-shift", "topology") and recombines them to "imagine" 
    novel Out-Of-Distribution (OOD) target states in a continuous latent space.
    """
    def __init__(self, vocab_size, embed_dim, state_dim):
        super().__init__()
        # Embed discrete concepts into a continuous topological space
        self.concept_embedding = torch_nn.Embedding(vocab_size, embed_dim)
        
        # Recombinator: Fuses multiple concepts into a single physical goal state
        self.recombinator = torch_nn.Sequential(
            torch_nn.Linear(embed_dim * 2, embed_dim * 2),
            torch_nn.GELU(),
            torch_nn.LayerNorm(embed_dim * 2),
            torch_nn.Linear(embed_dim * 2, state_dim)
        )

    def forward(self, concept_idx_1, concept_idx_2):
        # Imagine a new state by combining two abstract concepts 
        # (e.g., Concept 1: "Entropy Reduction", Concept 2: "Wavefront Variance")
        emb1 = self.concept_embedding(concept_idx_1)
        emb2 = self.concept_embedding(concept_idx_2)
        
        combined = torch.cat([emb1, emb2], dim=-1)
        imagined_goal_state = self.recombinator(combined)
        return imagined_goal_state


class ForwardWorldModel(torch_nn.Module):
    """
    Knowledge-Based Motivation (Surprise / Prediction Error).
    Predicts the next physical state given the current state and an action.
    High prediction error = High Surprise = High Intrinsic Reward for Exploration.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.dynamics = torch_nn.Sequential(
            torch_nn.Linear(state_dim + action_dim, state_dim * 2),
            torch_nn.GELU(),
            torch_nn.Linear(state_dim * 2, state_dim)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        predicted_next_state = state + self.dynamics(x) # Residual prediction
        return predicted_next_state


class IMGEP_Manager(torch_nn.Module):
    """
    Intrinsically Motivated Goal Exploration Process.
    Balances Knowledge-based motivation (seeking surprise) and 
    Competence-based motivation (seeking states where learning progress is highest).
    """
    def __init__(self, state_dim, action_dim, vocab_size, embed_dim):
        super().__init__()
        self.state_dim = state_dim
        
        # Sub-modules
        self.imagination = VygotskianImagination(vocab_size, embed_dim, state_dim)
        self.world_model = ForwardWorldModel(state_dim, action_dim)
        
        # Inverse Dynamics: Maps (Current State, Goal State) -> Action needed to reach it
        self.inverse_dynamics = torch_nn.Sequential(
            torch_nn.Linear(state_dim * 2, state_dim),
            torch_nn.GELU(),
            torch_nn.Linear(state_dim, action_dim),
            torch_nn.Tanh() # Assuming normalized continuous control actions
        )

        # Competence Memory Buffer (Tracks learning progress)
        # Maps imagined goal hashes to a history of their prediction errors
        self.competence_history = collections.defaultdict(lambda: collections.deque(maxlen=10))
        
        # Optimizers for self-supervised learning
        self.wm_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=1e-3)
        self.id_optimizer = torch.optim.Adam(self.inverse_dynamics.parameters(), lr=1e-3)

    def generate_goal(self, concept_1, concept_2, exploration_noise=0.1):
        """Generates a goal and computes its intrinsic motivation score."""
        with torch.no_grad():
            goal_state = self.imagination(concept_1, concept_2)
            
            # Add noise for continuous exploration around the discrete linguistic anchor
            goal_state += torch.randn_like(goal_state) * exploration_noise
            
        return goal_state

    def select_action(self, current_state, goal_state):
        """Uses inverse dynamics to attempt to reach the imagined goal."""
        x = torch.cat([current_state, goal_state], dim=-1)
        action = self.inverse_dynamics(x)
        return action

    def internalize_experience(self, state, action, next_state, goal_state, concept_key):
        """
        The core Autotelic Learning Loop. No external rewards are passed here.
        Updates internal models based on prediction error and learning progress.
        """
        # 1. Update Forward World Model (Knowledge Motivation)
        predicted_next_state = self.world_model(state, action)
        cos_sim = F.cosine_similarity(predicted_next_state, next_state, dim=-1).mean()
        surprise_loss = 1.0 - cos_sim # 0 means perfect prediction, 2 means complete anti-phase
        
        self.wm_optimizer.zero_grad()
        surprise_loss.backward(retain_graph=True)
        self.wm_optimizer.step()

        # 2. Update Inverse Dynamics (Competence Motivation)
        # We perform "Hindsight Experience Replay" (HER) - even if we didn't reach 
        # the intended goal, we definitely reached 'next_state', so train on that.
        pred_action = self.inverse_dynamics(torch.cat([state, next_state], dim=-1))
        competence_loss = F.mse_loss(pred_action, action)

        self.id_optimizer.zero_grad()
        competence_loss.backward()
        self.id_optimizer.step()

        # 3. Track Competence Progress (Derivative of Error)
        error_val = surprise_loss.item()
        history = self.competence_history[concept_key]
        history.append(error_val)
        
        # Calculate Learning Progress (Absolute derivative of error over time)
        # If error is dropping quickly, learning progress is high.
        learning_progress = 0.0
        if len(history) >= 2:
            learning_progress = abs(history[-1] - history[0])

        return {
            "surprise": error_val,
            "learning_progress": learning_progress
        }