import torch
import torch.nn as nn
import torch.nn.functional as F

class HolographicAssociativeDecoder(nn.Module):
    """
    The Absolute Egress Boundary (Zone B -> Zone A).
    Replaces the 'nn.Linear' Euclidean dot-product head with an exact Holographic Associative Lookup.
    This simulates the physical behavior of 4-bit Comprehension ADCs measuring constructive/destructive 
    interference against the pristine vocabulary.
    """
    def __init__(self, canonical_phase_lexicon: torch.Tensor, dsp_temperature: float = 0.05):
        super().__init__()
        
        # We must ingest the EXACT same immutable memory tensor used by the Vector Lifter.
        # This enforces perfect symmetry; the swarm only has to solve the logic, not the translation.
        if canonical_phase_lexicon.dtype not in [torch.complex64, torch.complex64]:
            raise TypeError("[ALETHEIA FATAL] Lexicon must be a complex phase array to calculate wave resonance.")
            
        self.register_buffer("reference_lexicon", canonical_phase_lexicon)
        self.dsp_temperature = dsp_temperature

    def forward(self, resolved_wave: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the post-Sagnac returning wave against the Platonic vocabulary.
        
        Args:
            resolved_wave: [Batch, SeqLen, Dim] (torch.complex64) - The output from the continuous core.
            
        Returns:
            Logit resonance scores [Batch, SeqLen, VocabSize] representing pure geometric phase alignment.
        """
        # 1. Phase Normalization (Hardware Stability Check)
        # Ensure the wave has not accrued artificial amplitude (energy) from thermal noise injections.
        # We divide the complex vector by its modulus to pin it back flawlessly to the S^4095 surface.
        wave_modulus = resolved_wave.abs() + 1e-9
        unit_wave = resolved_wave / wave_modulus

        # 2. Hermitian Inner Product (Physical Wave Interference)
        # We measure resonance by calculating the Hermitian projection: Re(Wave * Lexicon^H)
        # The conjugate transpose (lexicon.conj().t()) aligns the opposing phase angles.
        # Shape transition: (Batch, SeqLen, Dim) @ (Dim, Vocab) -> (Batch, SeqLen, Vocab)
        lexicon_h = self.reference_lexicon.conj().t()
        
        # The matrix multiplication in complex space effectively runs thousands of virtual Sagnac loops.
        resonance_matrix = torch.matmul(unit_wave, lexicon_h)

        # 3. Extraction of Epiplexity
        # The real component of this complex multiplication represents pure Constructive Interference (Truth).
        # The imaginary component is orthogonal thermodynamic noise, which is discarded mathematically.
        geometric_resonance = resonance_matrix.real

        # 4. Softmax DSP Scaling
        # We scale the physical resonance by the hardware temperature setting to produce sharp, 
        # definitive logit spikes for the UniversalREPL sandbox to parse.
        return geometric_resonance / self.dsp_temperature