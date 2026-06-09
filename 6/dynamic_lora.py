import os
import torch
import numpy as np

class DynamicLoraManager:
    """
    Manages dynamic LoRA updates in the continuous-time latent space.
    Accepts rehypothecated reflection delta tensors from the Zone B physical model
    to update Rank-16 LoRA adapters (A and B matrices), saving the updated state
    into a dynamic weight file to guide speculative reasoning.
    """
    def __init__(self, gemma_dim=2048, rank=16, lora_path="archive/dynamic_lora_weights.bin"):
        self.gemma_dim = gemma_dim
        self.rank = rank
        self.lora_path = lora_path
        
        # Initialize LoRA weight matrices
        # A: projects from gemma_dim to rank (initialized with random normal)
        self.lora_A = torch.randn(gemma_dim, rank) * 0.02
        # B: projects from rank to gemma_dim (initialized with zeros)
        self.lora_B = torch.zeros(rank, gemma_dim)
        
        self.load_weights()

    def load_weights(self):
        """Loads LoRA weights from disk if they exist."""
        if os.path.exists(self.lora_path):
            try:
                state = torch.load(self.lora_path, map_location="cpu")
                loaded_A = state.get("lora_A")
                loaded_B = state.get("lora_B")
                if loaded_A is not None and loaded_A.shape == (self.gemma_dim, self.rank):
                    self.lora_A = loaded_A
                else:
                    print(f"[LoRA ENGINE] Warning: Shape mismatch for lora_A in {self.lora_path}. Expected {(self.gemma_dim, self.rank)}, got {loaded_A.shape if loaded_A is not None else None}. Reinitializing weights.")
                if loaded_B is not None and loaded_B.shape == (self.rank, self.gemma_dim):
                    self.lora_B = loaded_B
                else:
                    print(f"[LoRA ENGINE] Warning: Shape mismatch for lora_B in {self.lora_path}. Expected {(self.rank, self.gemma_dim)}, got {loaded_B.shape if loaded_B is not None else None}. Reinitializing weights.")
                print(f"[LoRA ENGINE] Loaded dynamic LoRA weights from {self.lora_path}")
            except Exception as e:
                print(f"[LoRA ENGINE] Warning: Failed to load LoRA weights: {e}")

    def save_weights(self):
        """Saves current LoRA weights to disk."""
        try:
            os.makedirs(os.path.dirname(self.lora_path), exist_ok=True)
            torch.save({
                "lora_A": self.lora_A,
                "lora_B": self.lora_B
            }, self.lora_path)
            print(f"[LoRA ENGINE] Saved updated dynamic LoRA weights to {self.lora_path}")
        except Exception as e:
            print(f"[LoRA ENGINE] Warning: Failed to save LoRA weights: {e}")

    def update_with_rehypothecated_tensors(self, delta_projected, alignment_score: float):
        """
        Dynamically updates the LoRA weights using the pre-projected error vector.
        """
        with torch.no_grad():
            if isinstance(delta_projected, np.ndarray):
                delta_tensor = torch.tensor(delta_projected, dtype=torch.float32)
            else:
                delta_tensor = delta_projected.to(torch.float32)
                
            if len(delta_tensor) < self.gemma_dim:
                delta_tensor = torch.nn.functional.pad(delta_tensor, (0, self.gemma_dim - len(delta_tensor)))
            elif len(delta_tensor) > self.gemma_dim:
                delta_tensor = delta_tensor[:self.gemma_dim]
            
            # Calculate update step proportional to Sagnac misalignment
            learning_rate = 0.05 * (1.0 - alignment_score)
            
            # Calculate update steps
            update_A = torch.outer(delta_tensor, delta_tensor[:self.rank])
            update_B = torch.outer(delta_tensor[:self.rank], delta_tensor)
            
            # Normalize updates to have unit Frobenius norm to prevent vanishing updates at scale
            norm_A = torch.norm(update_A)
            norm_B = torch.norm(update_B)
            
            if norm_A > 1e-8:
                update_A = (update_A / norm_A) * learning_rate * 5.0  # Scale up for measurable bending
            if norm_B > 1e-8:
                update_B = (update_B / norm_B) * learning_rate * 5.0
                
            self.lora_A += update_A
            self.lora_B += update_B
            
            # Clamp weights to prevent gradient explosion
            self.lora_A = torch.clamp(self.lora_A, -1.0, 1.0)
            self.lora_B = torch.clamp(self.lora_B, -1.0, 1.0)
            
        print(f"[LoRA ENGINE] Rehypothecation applied. Updated weights with error energy (Alignment: {alignment_score:.4f}).")
        self.save_weights()

    def apply_lora(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Applies the dynamic LoRA weight transformations to model activations/embeddings.
        Formula: h_new = h + (h @ A) @ B
        """
        with torch.no_grad():
            lora_step = torch.matmul(torch.matmul(activations, self.lora_A), self.lora_B)
            return activations + lora_step
