import torch
import torch.nn.functional as F

class SemanticCleanupMatrix:
    """
    Phase III, Step 3: The Egress Boundary.
    A Modern Hopfield Network that catches continuous, thermally noisy 
    C^4096 phase waves and snaps them into absolute discrete tokens via 
    Log-Sum-Exp energy basins.
    """
    def __init__(self, dim=4096, device=None):
        self.dim = dim
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # The Hopfield Associative Memory Bank (The "Wells")
        self.lexicon_tokens = []
        self.lexicon_waves = None # Will be a tensor of shape [Vocab_Size, 4096]

    def assimilate_lexicon(self, token_dict: dict):
        """
        Loads the foundational Unitary Wave Embeddings (UWE) from the Transducer 
        into the Hopfield memory matrix.
        token_dict: { "token_string": tensor_wave_C4096 }
        """
        self.lexicon_tokens = list(token_dict.keys())
        waves = list(token_dict.values())
        
        if waves:
            # Stack all discrete truth-waves into a single dense matrix
            self.lexicon_waves = torch.stack(waves).to(self.device)
            print(f"[Egress] Hopfield Matrix Assimilated: {len(self.lexicon_tokens)} Discrete Attractors.")

    def calculate_hopfield_energy(self, noisy_wave: torch.Tensor, beta: float = 50.0) -> float:
        """
        Calculates the thermodynamic Free Energy of the wave relative to the known lexicon.
        Energy approaches 0 when the wave perfectly matches a known concept.
        """
        if self.lexicon_waves is None:
            return float('inf')

        # Sagnac Phase Resonance (Real part of complex dot product, normalized)
        # Shape: [Vocab_Size]
        resonances = torch.real(torch.matmul(self.lexicon_waves.conj(), noisy_wave)) / self.dim
        
        # Modern Hopfield Energy Function: E = -(1/beta) * log( sum( exp(beta * resonance) ) )
        energy = - (1.0 / beta) * torch.logsumexp(beta * resonances, dim=0)
        return energy.item()

    def snap_and_decode(self, noisy_wave: torch.Tensor, beta: float = 50.0) -> tuple:
        """
        The Crystallization Event.
        Takes a thermal wave from the SyncytiumCore, applies the Hopfield Softmax retrieval,
        and snaps it to the absolute nearest discrete token.
        
        beta: The inverse temperature of the Hopfield network. 
              Higher beta = deeper wells = absolute rigid snapping.
        """
        if self.lexicon_waves is None:
            raise ValueError("Hopfield matrix is empty. Assimilate lexicon first.")

        # 1. Compute Sagnac Alignment (Phase-matching)
        resonances = torch.real(torch.matmul(self.lexicon_waves.conj(), noisy_wave)) / self.dim
        
        # 2. Apply Deep Thermodynamic Well (Softmax over similarities)
        # This acts as the GBNF Logit Sieve, mathematically annihilating non-resonant frequencies
        snapping_weights = F.softmax(beta * resonances, dim=0)
        
        # 3. Retrieve the pristine, cleaned wave (The continuous output)
        clean_wave = torch.sum(snapping_weights.unsqueeze(1) * self.lexicon_waves, dim=0)
        clean_wave = clean_wave / (torch.abs(clean_wave) + 1e-9) # Unitary projection
        
        # 4. Extract the discrete token (The digital string)
        best_index = torch.argmax(snapping_weights).item()
        crystallized_token = self.lexicon_tokens[best_index]
        confidence = snapping_weights[best_index].item()
        
        return crystallized_token, clean_wave, confidence

    def generative_unbundle(self, noisy_sequence_wave: torch.Tensor, target_role_wave: torch.Tensor, beta: float = 50.0) -> tuple:
        """
        Phase IV: Generative Unbundling.
        Instead of snapping a whole wave to a single token, we unbind a specific 
        topological role from a complex thought-wave, and THEN snap the extracted filler 
        to the known lexicon.
        """
        # 1. Circular Unbind (Extract the filler wave for the given role)
        # Assuming the Transducer uses circular_convolution_bind
        import torch.fft as fft
        extracted_wave = fft.ifft(fft.fft(noisy_sequence_wave) * torch.conj(fft.fft(target_role_wave)))
        extracted_wave = extracted_wave / (torch.abs(extracted_wave) + 1e-9)
        
        # 2. Pass the extracted filler through the Hopfield energy basin
        return self.snap_and_decode(extracted_wave, beta=beta)

# --- Execution Test ---
if __name__ == "__main__":
    from universal_epistemic_transducer import UniversalEpistemicTransducer
    
    # 1. Initialize the environments
    transducer = UniversalEpistemicTransducer(dim=4096)
    egress_matrix = SemanticCleanupMatrix(dim=4096)
    
    # 2. Build a micro-lexicon
    target_words = ["entropy", "sonnet", "def", "return", "tensor"]
    lexicon = {word: transducer.get_concept_wave(word) for word in target_words}
    egress_matrix.assimilate_lexicon(lexicon)
    
    print("\n--- Testing Semantic Cleanup ---")
    
    # 3. Fetch a pristine wave ("sonnet")
    pristine_wave = lexicon["sonnet"]
    
    # 4. Simulate severe Langevin Thermodynamic Noise (The Wave gets blurred during reasoning)
    noise_variance = 1.5  # Massive thermal noise
    thermal_noise = torch.randn_like(pristine_wave.real) + 1j * torch.randn_like(pristine_wave.imag)
    noisy_wave = pristine_wave + (thermal_noise * noise_variance)
    noisy_wave = noisy_wave / (torch.abs(noisy_wave) + 1e-9) # Project back to Stiefel manifold
    
    # Check degradation before cleanup
    degraded_resonance = torch.real(torch.sum(pristine_wave.conj() * noisy_wave)) / 4096.0
    print(f"Degraded Wave Resonance (Before Cleanup): {degraded_resonance:.4f}")
    
    # 5. Snap the wave using the Modern Hopfield Egress
    token, clean_wave, confidence = egress_matrix.snap_and_decode(noisy_wave, beta=100.0)
    
    # 6. Verify absolute recovery
    recovered_resonance = torch.real(torch.sum(pristine_wave.conj() * clean_wave)) / 4096.0
    print(f"Recovered Wave Resonance (After Cleanup): {recovered_resonance:.4f}")
    print(f"Crystallized Token Output: '{token}' (Confidence: {confidence * 100:.2f}%)")

    # 7. Test Generative Unbundling (Sequence Extraction)
    print("\n--- Testing Generative Unbundling ---")
    role_subject = transducer.get_concept_wave("SUBJECT")
    filler_subject = transducer.get_concept_wave("entropy")
    role_action = transducer.get_concept_wave("ACTION")
    filler_action = transducer.get_concept_wave("tensor")
    
    # Create a complex bundled thought (like a sentence)
    bound_subject = transducer.circular_convolution_bind(role_subject, filler_subject)
    bound_action = transducer.circular_convolution_bind(role_action, filler_action)
    complex_thought = transducer.bundle_superposition([bound_subject, bound_action])
    
    # Simulate extraction of just the ACTION from the complex thought
    print(f"Querying complex thought for role: 'ACTION'...")
    extracted_token, _, ext_confidence = egress_matrix.generative_unbundle(complex_thought, role_action, beta=100.0)
    print(f"Unbundled Token: '{extracted_token}' (Confidence: {ext_confidence * 100:.2f}%) - Expected: 'tensor'")
    
    # Simulate extraction of just the SUBJECT from the complex thought
    print(f"Querying complex thought for role: 'SUBJECT'...")
    extracted_token2, _, ext_confidence2 = egress_matrix.generative_unbundle(complex_thought, role_subject, beta=100.0)
    print(f"Unbundled Token: '{extracted_token2}' (Confidence: {ext_confidence2 * 100:.2f}%) - Expected: 'entropy'")