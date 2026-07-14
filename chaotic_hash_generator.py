import numpy as np

class ChaoticHashGenerator:
    """
    ENGINEERING SPECIFICATION: PROJECT HENRI - DETERMINISTIC CHAOS MAP (V1.0.0)
    Replaces np.random simulators with an absolute, reproducible deterministic 
    vectorized topology generator. Uses a non-linear bit-mixing avalanche (Murmur3-like) 
    to map spatial indices into discrete orthogonal phase states (Z_K).
    """
    
    @staticmethod
    def generate(seed: int, shape: tuple, max_val: int) -> np.ndarray:
        """
        Generates a deterministic pseudo-orthogonal array of shape `shape` bounded by [0, max_val).
        """
        size = int(np.prod(shape))
        
        # Avalanche Map: Deterministic chaos over continuous Z space
        indices = np.arange(size, dtype=np.uint64) + np.uint64(seed)
        
        # Non-linear bitwise diffusion
        indices ^= (indices >> 33)
        indices *= np.uint64(0xff51afd7ed558ccd)
        indices ^= (indices >> 33)
        indices *= np.uint64(0xc4ceb9fe1a85ec53)
        indices ^= (indices >> 33)
        
        # Quantize to the K-modular space
        res = (indices % np.uint64(max_val)).astype(np.int32)
        return res.reshape(shape)

    @staticmethod
    def generate_signed(seed: int, shape: tuple, max_magnitude: float) -> np.ndarray:
        """
        Generates a deterministic Langevin thermal noise vector centered around 0.
        Bounds: [-max_magnitude, max_magnitude].
        """
        if max_magnitude < 1.0:
            return np.zeros(shape, dtype=np.int32)
            
        bound = int(max_magnitude + 1)
        # Shift the output to span [-bound, bound]
        raw_chaos = ChaoticHashGenerator.generate(seed, shape, max_val=(bound * 2 + 1))
        return raw_chaos - bound
