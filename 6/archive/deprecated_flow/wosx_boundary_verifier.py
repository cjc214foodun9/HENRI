import os
import sys
import numpy as np
import torch
import psycopg
import uuid

# Ensure parent and sibling directories are in path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "6"))

from henri_contract import db_to_complex, DIMS

class WosxBoundaryVerifier:
    """
    Loads 4096-D complex continuous boundary conditions (Dirichlet/Neumann) 
    from TimescaleDB and evaluates the physical resonance of candidate hypotheses.
    """
    def __init__(self, db_url=None, threshold=0.35):
        if db_url is None:
            self.db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/henri")
        else:
            self.db_url = db_url
        self.threshold = threshold
        
        self.dirichlet_boundary = None
        self.neumann_boundary = None
        
        self.load_boundary_axioms()
        
    def load_boundary_axioms(self):
        print("[WOSX VERIFIER] Loading continuous boundary axioms from TimescaleDB...")
        try:
            with psycopg.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    # Fetch Dirichlet boundary condition
                    cur.execute(
                        "SELECT hrr_wavefront::text FROM hrr_canonical_lexicon WHERE semantic_label = 'dirichlet_boundary';"
                    )
                    row = cur.fetchone()
                    if row:
                        self.dirichlet_boundary = db_to_complex(row[0], DIMS.hrr_dim)
                    else:
                        print("[WOSX VERIFIER] Warning: 'dirichlet_boundary' not found in database.")
                        
                    # Fetch Neumann boundary condition
                    cur.execute(
                        "SELECT hrr_wavefront::text FROM hrr_canonical_lexicon WHERE semantic_label = 'neumann_boundary';"
                    )
                    row = cur.fetchone()
                    if row:
                        self.neumann_boundary = db_to_complex(row[0], DIMS.hrr_dim)
                    else:
                        print("[WOSX VERIFIER] Warning: 'neumann_boundary' not found in database.")
                        
            if self.dirichlet_boundary is not None and self.neumann_boundary is not None:
                print(f"[WOSX VERIFIER] Successfully loaded boundary axioms. Shapes: Dirichlet={self.dirichlet_boundary.shape}, Neumann={self.neumann_boundary.shape}")
            else:
                print("[WOSX VERIFIER] Fallback: Seeding random unit-modulus boundary vectors due to missing database records...")
                self._seed_fallbacks()
        except Exception as e:
            print(f"[WOSX VERIFIER] Database load error: {e}. Fallback to unit-modulus random boundaries.")
            self._seed_fallbacks()

    def _seed_fallbacks(self):
        rng = np.random.default_rng(101)
        p1 = (rng.random(DIMS.hrr_dim) * 2 * np.pi) - np.pi
        p2 = (rng.random(DIMS.hrr_dim) * 2 * np.pi) - np.pi
        self.dirichlet_boundary = np.exp(1j * p1)
        self.neumann_boundary = np.exp(1j * p2)

    def verify_hypothesis_wave(self, hypothesis_wave: torch.Tensor) -> dict:
        """
        Verifies a 4096-D complex candidate wave against loaded Dirichlet and Neumann boundary axioms.
        """
        if torch.is_tensor(hypothesis_wave):
            hyp_np = hypothesis_wave.detach().cpu().to(torch.complex64).numpy()
        else:
            hyp_np = np.asarray(hypothesis_wave, dtype=np.complex64)
            
        hyp_np = hyp_np.flatten()
        if len(hyp_np) != DIMS.hrr_dim:
            if len(hyp_np) < DIMS.hrr_dim:
                hyp_np = np.pad(hyp_np, (0, DIMS.hrr_dim - len(hyp_np)))
            else:
                hyp_np = hyp_np[:DIMS.hrr_dim]
                
        # Normalize hypothesis wave to unit circle
        hyp_np = hyp_np / (np.abs(hyp_np) + 1e-8)
        
        # Dirichlet alignment: Cosine similarity on complex plane
        d_norm = self.dirichlet_boundary / (np.abs(self.dirichlet_boundary) + 1e-8)
        dirichlet_alignment = np.real(np.mean(hyp_np * np.conj(d_norm)))
        
        # Neumann alignment
        n_norm = self.neumann_boundary / (np.abs(self.neumann_boundary) + 1e-8)
        neumann_alignment = np.real(np.mean(hyp_np * np.conj(n_norm)))
        
        # Combined boundary resonance score
        resonance_score = 0.5 * (dirichlet_alignment + neumann_alignment)
        
        # Verify against threshold boundary
        passed = resonance_score >= self.threshold
        
        return {
            "passed": bool(passed),
            "resonance_score": float(resonance_score),
            "dirichlet_alignment": float(dirichlet_alignment),
            "neumann_alignment": float(neumann_alignment),
            "threshold": self.threshold
        }

    def verify_hypothesis_code(self, candidate_code: str, router) -> dict:
        """
        Tokenizes candidate code, projects it to continuous wave space via the router, and verifies it.
        """
        # Encode string characters as token integers
        if hasattr(router, 'gen_model') and router.gen_model is not None and hasattr(router.gen_model, 'tokenize'):
            tokens = router.gen_model.tokenize(candidate_code.encode("utf-8"))
        else:
            tokens = [ord(c) for c in candidate_code]
            
        token_tensor = torch.tensor(tokens, dtype=torch.long, device='cpu')
        
        # Project using the L3SwarmRouter
        with torch.no_grad():
            wave, _, _ = router(token_tensor.unsqueeze(0))
            wave = wave[0] # [4096] complex
            
        return self.verify_hypothesis_wave(wave)
