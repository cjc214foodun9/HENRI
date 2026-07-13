import torch
import torch.nn as nn
import math

# Enforce strict float32 and complex64 logic matching the continuous physics engine
torch.set_default_dtype(torch.float32)

class HolographicActionCrystallizer(nn.Module):
    """
    Motor Egress module. Translates a continuous 4096-D complex wave into a strictly 
    validated, discrete JSON tool-call payload by leveraging a GBNF logical sieve.
    """
    def __init__(self, dim: int = 4096, vocab_size: int = 32000):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Linear projection mapping 4096 real + 4096 imaginary continuous 
        # components back into discrete target vocabulary logit space
        self.projection = nn.Linear(dim * 2, vocab_size, dtype=torch.float32)
        
    def forward(self, psi: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Projects the stabilized thermodynamic wave into discrete space and 
        crystallizes it against the FSM environmental boundaries.
        """
        # Deinterleave the complex wave into distinct real and imaginary feature blocks
        # Shape projection: [batch, dim * 2]
        wave_features = torch.cat([psi.real, psi.imag], dim=-1)
        
        # Project structural wave to raw continuous logit space
        logits = self.projection(wave_features)
        
        # Collapse the wave physically against the FSM boundaries
        probabilities, selection = self.GBNF_Logit_Sieve(logits, mask)
        
        return probabilities, selection

    def GBNF_Logit_Sieve(self, logits: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Non-autoregressive structural constraint gate.
        Mathematically multiplies violating state probabilities by exactly 0,
        ensuring syntactical output guarantees.
        """
        # Ensure mask is exactly binary float format (0.0 or 1.0)
        mask = mask.to(torch.float32)
        
        # To prevent inf * 0 = NaN when the network predicts massive confidences for 
        # invalid tokens, we anchor the invalid logits safely before applying the mathematical mask.
        safe_logits = logits.masked_fill(mask == 0.0, -1e9)
        
        # Apply standard numerical stability offset
        max_allowed_logits = torch.max(safe_logits, dim=-1, keepdim=True).values
        stable_logits = safe_logits - max_allowed_logits
        
        # Calculate wave intensities
        exp_logits = torch.exp(stable_logits)
        
        # Physical boundary collision (Eradicate impossible branches)
        # We multiply by mask to strictly enforce the mathematical prompt requirement
        masked_exp = exp_logits * mask
        
        # Constrained normalization (Softmax equivalent over valid topological boundaries only)
        probabilities = masked_exp / (torch.sum(masked_exp, dim=-1, keepdim=True) + 1e-16)
        
        # Final physical crystallization collapses into a singular guaranteed action
        selection = torch.argmax(probabilities, dim=-1)
        
        return probabilities, selection


if __name__ == "__main__":
    print("Initializing Holographic Tool-Use Crystallization (Motor Egress)...")
    
    dim = 4096
    vocab = 5  # Small test schema vocabulary [ "{" , "API" , "CALL" , "}" , "MALFORMED_JSON" ]
    batch = 1
    
    crystallizer = HolographicActionCrystallizer(dim=dim, vocab_size=vocab)
    
    # 1. Simulate an incoming phase-locked (stabilized) biophysical wave from the Syncytium
    psi = torch.randn(batch, dim, dtype=torch.complex64)
    psi = psi / torch.norm(psi, p=2, dim=-1, keepdim=True)
    
    # Artificially inject a massive thermodynamic bias towards an invalid topological structure
    # (Simulating a highly confident AI hallucination requesting Index 4: "MALFORMED_JSON")
    with torch.no_grad():
        crystallizer.projection.bias[4] += 1000.0  
        
    print("\n--- Unconstrained Model Prediction ---")
    wave_features = torch.cat([psi.real, psi.imag], dim=-1)
    raw_logits = crystallizer.projection(wave_features)
    raw_probs = torch.softmax(raw_logits, dim=-1)
    
    print(f"Raw Confidences:  {raw_probs[0].tolist()}")
    print(f"Argmax Selection: Token {torch.argmax(raw_probs, dim=-1).item()} (MALFORMED_JSON)")
    
    # 2. Apply the GBNF Logical Sieve
    # The external Finite State Machine (FSM) asserts only JSON syntax is physically possible.
    # Therefore, Token 0 ("{") is valid. Token 4 ("MALFORMED_JSON") is mathematically blocked.
    fsm_mask = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    
    print("\n--- Masked Structural Crystallization ---")
    constrained_probs, safe_selection = crystallizer(psi, fsm_mask)
    
    print(f"Constrained Probabilities: {constrained_probs[0].tolist()}")
    print(f"Masked Selection: Token {safe_selection.item()}")
    
    # 3. Assert exact mathematical boundaries
    assert constrained_probs[0, 4].item() == 0.0, "Probability leakage detected! Malformed token probability is > 0."
    assert safe_selection.item() == 0, "Sieve failure! The model bypassed the physical FSM boundary constraint."
    
    print("\n[SUCCESS] Continuous wave crystallized. Sieve fully bounded the egress space, destroying malformed action topologies.")
