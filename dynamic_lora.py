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
                self.lora_A = state.get("lora_A", self.lora_A)
                self.lora_B = state.get("lora_B", self.lora_B)
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

    def update_with_rehypothecated_tensors(self, reflection_delta: np.ndarray, alignment_score: float):
        """
        Dynamically updates the LoRA weights using the rehypothecated reflection delta
        (error energy vector) from the Zone B physical emulator.
        """
        # Convert reflection delta complex array to real features
        delta_real = np.real(reflection_delta)
        delta_imag = np.imag(reflection_delta)
        
        # Merge parts and project to gemma_dim size (4096 -> 2048)
        delta_combined = (delta_real[:self.gemma_dim] + delta_imag[:self.gemma_dim]) / 2.0
        if len(delta_combined) < self.gemma_dim:
            delta_combined = np.pad(delta_combined, (0, self.gemma_dim - len(delta_combined)))
            
        delta_tensor = torch.tensor(delta_combined, dtype=torch.float32)
        
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
            
        with torch.no_grad():
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
