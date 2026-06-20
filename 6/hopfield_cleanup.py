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

    def register_concept(self, label: str, vector: torch.Tensor):
        """Registers a pristine vector in the Hopfield vocabulary."""
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension must be {self.dim}, got {vector.shape[0]}")
        # Ensure unit magnitude
        mags = torch.abs(vector)
        mags = torch.clamp(mags, min=1e-8)
        self.vocabulary[label] = vector / mags

    def register_lexicon(self, lexicon):
        """Loads all concepts from a ZoneCOrthogonalLexicon instance."""
        for label, vector in lexicon.vocabulary.items():
            self.register_concept(label, vector)

    def cleanup(self, noisy_wave: torch.Tensor) -> tuple:
        """
        Executes continuous state evolution of the Modern Hopfield Network:
        s_new = sgn_complex( sum_mu e^{beta * sim_mu} * v_mu )
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
        # Stack vocabulary vectors: shape [M, dim]
        v_matrix = torch.stack([self.vocabulary[lbl] for lbl in labels]).to(s.device)

        last_s = s.clone()
        
        for iteration in range(self.max_iterations):
            # Compute complex cosine similarities: Re(v_mu * s^*)
            # v_matrix is [M, dim], s is [dim]
            # dot product: sum_j v_{mu, j} * conj(s_j)
            dot_products = torch.sum(v_matrix * torch.conj(s).unsqueeze(0), dim=-1)
            similarities = torch.real(dot_products) / float(self.dim)
            
            # Apply exponential activation function (Modern Hopfield Network)
            # We subtract max similarity for numerical stability in softmax/exp
            max_sim = torch.max(similarities)
            weights = torch.exp(self.beta * (similarities - max_sim))
            weights = weights / (torch.sum(weights) + 1e-8)  # normalize weights

            # Form state superposition: shape [dim]
            x = torch.sum(weights.unsqueeze(1) * v_matrix, dim=0)
            
            # Apply complex-valued sgn projection (project elements back to unit circle)
            x_mags = torch.abs(x)
            x_mags = torch.clamp(x_mags, min=1e-8)
            s = x / x_mags
            
            # Convergence check
            diff = torch.sum(torch.abs(s - last_s)).item()
            if diff < self.tolerance:
                break
            last_s = s.clone()

        # Final match lookup
        dot_products = torch.sum(v_matrix * torch.conj(s).unsqueeze(0), dim=-1)
        similarities = torch.real(dot_products) / float(self.dim)
        
        best_idx = torch.argmax(similarities).item()
        best_label = labels[best_idx]
        confidence = similarities[best_idx].item()
        
        return s, best_label, confidence
