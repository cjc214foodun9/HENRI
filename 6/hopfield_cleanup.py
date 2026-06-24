import torch
import numpy as np
import math

class HopfieldSemanticCleanup:
    """
    Zone C Receiver chiplet: Dense Associative Memory (Modern Hopfield Network).
    Digitizes and cleans up blurry/noisy 4096-D complex waves from Zone B
    back to pristine orthogonal semantic vectors from the database or lexicon.
    """
    def __init__(self, dim=4096, beta=20.0, max_iterations=5, tolerance=1e-5):
        self.dim = dim
        self.beta = beta  # Inverse temperature (selectivity)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.vocabulary = {}  # Map: label -> complex tensor (shape: [dim])
        self.repellers = []  # List of complex tensors (shape: [dim])

    def register_concept(self, label: str, vector: torch.Tensor):
        """Registers a pristine vector in the Hopfield vocabulary."""
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension must be {self.dim}, got {vector.shape[0]}")
        # Ensure unit magnitude
        mags = torch.abs(vector)
        mags = torch.clamp(mags, min=1e-8)
        self.vocabulary[label] = vector / mags

    def register_repeller(self, vector: torch.Tensor):
        """Registers a repeller vector in the Hopfield memory to form topological hills."""
        if vector.shape[0] != self.dim:
            raise ValueError(f"Repeller dimension must be {self.dim}, got {vector.shape[0]}")
        mags = torch.abs(vector)
        mags = torch.clamp(mags, min=1e-8)
        self.repellers.append(vector / mags)
        # Bounded size to prevent memory bloat
        if len(self.repellers) > 50:
            self.repellers.pop(0)

    def clear_repellers(self):
        """Clears the repellers list."""
        self.repellers = []

    def register_lexicon(self, lexicon):
        """Loads all concepts from a ZoneCOrthogonalLexicon instance."""
        for label, vector in lexicon.vocabulary.items():
            self.register_concept(label, vector)

    def cleanup(self, noisy_wave: torch.Tensor, gamma: float = 0.5) -> tuple:
        """
        Executes continuous state evolution of the Modern Hopfield Network
        incorporating dynamic repellers as a negative exponential interaction term:
        x = sum_mu e^{beta * sim_mu} * v_mu - gamma * sum_nu e^{beta * sim_nu} * r_nu
        Returns:
            clean_wave: The converged unit-magnitude complex tensor [dim]
            best_label: The closest semantic label in the vocabulary
            confidence: Cosine similarity of the clean wave to the registered vector
        """
        if not self.vocabulary:
            raise ValueError("[Hopfield] Vocabulary is empty. Cannot perform cleanup.")

        # Reshape input to [dim] if necessary
        s = noisy_wave.clone().view(-1)
        
        # Ensure input is unit magnitude
        s_mags = torch.abs(s)
        s_mags = torch.clamp(s_mags, min=1e-8)
        s = s / s_mags

        labels = list(self.vocabulary.keys())
        v_matrix = torch.stack([self.vocabulary[lbl] for lbl in labels]).to(s.device)

        last_s = s.clone()
        
        for iteration in range(self.max_iterations):
            # 1. Attractor updates
            dot_products_a = torch.sum(v_matrix * torch.conj(s).unsqueeze(0), dim=-1)
            similarities_a = torch.real(dot_products_a) / float(self.dim)
            
            max_sim = torch.max(similarities_a)
            weights_a = torch.exp(self.beta * (similarities_a - max_sim))
            weights_a = weights_a / (torch.sum(weights_a) + 1e-8)
            
            x_attractors = torch.sum(weights_a.unsqueeze(1) * v_matrix, dim=0)
            x = x_attractors
            
            # 2. Repeller updates (negative exponential interaction term)
            if self.repellers:
                rep_matrix = torch.stack(self.repellers).to(s.device)
                dot_products_r = torch.sum(rep_matrix * torch.conj(s).unsqueeze(0), dim=-1)
                similarities_r = torch.real(dot_products_r) / float(self.dim)
                
                weights_r = torch.exp(self.beta * (similarities_r - max_sim))
                weights_r = weights_r / (torch.sum(weights_r) + 1e-8)
                
                x_repellers = torch.sum(weights_r.unsqueeze(1) * rep_matrix, dim=0)
                
                # Apply the negative interaction update
                x = x - gamma * x_repellers
            
            # Apply complex-valued sgn projection
            x_mags = torch.abs(x)
            x_mags = torch.clamp(x_mags, min=1e-8)
            s = x / x_mags
            
            # Convergence check
            diff = torch.sum(torch.abs(s - last_s)).item()
            if diff < self.tolerance:
                break
            last_s = s.clone()

        # Final match lookup (against attractors only)
        dot_products = torch.sum(v_matrix * torch.conj(s).unsqueeze(0), dim=-1)
        similarities = torch.real(dot_products) / float(self.dim)
        
        best_idx = torch.argmax(similarities).item()
        best_label = labels[best_idx]
        confidence = similarities[best_idx].item()
        
        # Contrastive Egress Snapping: cleanly snap continuous phase angles onto the closest canonical vector
        canonical_vector = v_matrix[best_idx]
        s = torch.polar(torch.ones_like(s.real), torch.angle(canonical_vector))
        
        return s, best_label, confidence
