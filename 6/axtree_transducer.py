import json
import torch
import hashlib
import numpy as np

class AXTreeTransducer:
    """
    AXTree-to-Phasor Transducer:
    Translates WCAG-compliant AXTree JSON layouts into unit-modulus complex vectors
    in the 4096-dimensional phase space S^4095. Natively binds roles, labels, and focus
    states using circular convolution (phasor element-wise multiplication).
    """
    def __init__(self, hrr_dim=4096):
        self.hrr_dim = hrr_dim
        
        # Pre-generate base vectors for common accessibility roles to ensure high contrast
        self.roles = ["button", "input", "table", "cell", "row", "heading", "link", "dialog", "alert", "region"]
        self.role_bases = {}
        for role in self.roles:
            self.role_bases[role] = self._generate_deterministic_wave(role)
            
        # Focus carrier wave
        self.focus_carrier = self._generate_deterministic_wave("focus_carrier_signal_active")

    def _generate_deterministic_wave(self, seed_str: str) -> torch.Tensor:
        """Generates a deterministic unit complex vector based on a string seed."""
        # Use SHA-256 for deterministic hashing
        hash_digest = hashlib.sha256(seed_str.encode('utf-8')).digest()
        # Seed NumPy generator with digest bytes
        seed_int = int.from_bytes(hash_digest[:4], 'big')
        rng = np.random.default_rng(seed_int)
        
        # Generate phase angles in [-pi, pi]
        phases = rng.uniform(low=-np.pi, high=np.pi, size=self.hrr_dim)
        phases_tensor = torch.tensor(phases, dtype=torch.float32)
        
        # Map to complex phasor (unit modulus)
        return torch.polar(torch.ones(self.hrr_dim, dtype=torch.float32), phases_tensor)

    def transduce_node(self, node: dict) -> torch.Tensor:
        """
        Binds a single AXTree node (role, name, value, focus_state) into a complex phasor.
        Uses phasor element-wise multiplication (equivalent to circular convolution).
        """
        role = node.get("role", "generic").lower()
        name = node.get("name", "")
        value = node.get("value", "")
        focus_state = node.get("focus_state", False)
        
        # 1. Role phasor
        v_role = self.role_bases.get(role)
        if v_role is None:
            # Generate ad-hoc base for unknown roles
            v_role = self._generate_deterministic_wave(role)
            
        # 2. Name / Label phasor
        v_name = self._generate_deterministic_wave(name)
        
        # 3. Value phasor
        v_value = self._generate_deterministic_wave(value) if value else torch.polar(torch.ones(self.hrr_dim), torch.zeros(self.hrr_dim))
        
        # Bind using element-wise complex multiplication (circular convolution)
        v_bound = v_role * v_name * v_value
        
        # 4. Modulate with focus carrier if active
        if focus_state:
            v_bound = v_bound * self.focus_carrier
            
        return v_bound

    def transduce_tree(self, axtree_json: str) -> torch.Tensor:
        """
        Transduces an entire AXTree JSON layout by superimposing all node vectors
        and normalizing the sum onto the unit hypersphere S^4095.
        """
        data = json.loads(axtree_json)
        nodes = data.get("nodes", [])
        
        if not nodes:
            # Empty screen fallback
            return torch.polar(torch.ones(self.hrr_dim), torch.zeros(self.hrr_dim))
            
        # Superimpose all node vectors
        v_sum = torch.zeros(self.hrr_dim, dtype=torch.complex64)
        for node in nodes:
            v_node = self.transduce_node(node)
            v_sum += v_node
            
        # Normalize back to the unit hypersphere
        norm = torch.norm(v_sum)
        if norm > 0:
            v_sum = v_sum / norm
            
        return v_sum
