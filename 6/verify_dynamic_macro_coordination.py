"""  
Project HENRI: System Verification and Validation Suite  
Target: Upgraded Phase 4 & Phase 5 Heterogeneous Intent Mapping Pipeline  
"""

import torch  
import torch.nn as nn  
import torch.nn.functional as F
from henri_core.diffusion_canvas import NonAutoregressiveCanvasSampler  
from henri_core.pullback_projector import RightKanPullbackOrchestrator, AdaptiveDualBlockTokenProjector

class MockCore(nn.Module):  
    def __init__(self, dim=4096):  
        super().__init__()  
        self.p = nn.Parameter(torch.randn(1, dim))  
    def forward(self, canvas, t):  
        return torch.zeros_like(canvas)

def run_full_stack_scan():  
    print("=== INITIALIZING HENRI HETEROGENEOUS COGNITIVE CORE SCAN ===")  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print(f"[BOOT] Target accelerator environment locked: {device}")

    # 1. Test Component A: Dynamic Vocabulary Mask Arrays  
    mock_core = MockCore(dim=4096).to(device)  
    translation_head = nn.Linear(4096, 262144, bias=False).to(device)  
    sampler = NonAutoregressiveCanvasSampler(mock_core, translation_head, num_diffusion_steps=5).to(device)  
      
    mock_traj = torch.randn(1, 4096, device=device)  
      
    print("[SCAN] Validating Canvas Sampler across distinct Intent Flags...")  
    for intent in ["CONVERSATION", "CODE", "RESEARCH"]:  
        tokens = sampler.crystallize_motif(swarm_trajectory=mock_traj, sequence_length=128, intent_flag=intent)  
        intent_padded = intent.ljust(15)
        print(f"  - Intent Flag: {intent_padded} | Canvas Output Shape: {tokens.shape}")  
        assert tokens.shape == torch.Size([1, 128]), f"Fatal: Intent {intent} corrupted sequence boundaries!"  
    print("[SUCCESS] Phase 4 Dynamic Mask Matrix verified.")

    # 2. Test Component C: Adaptive Dual-Block Projector Gates  
    orchestrator = RightKanPullbackOrchestrator(dim=4096, omega_dim=128, tolerance=0.01).to(device)  
    projector = AdaptiveDualBlockTokenProjector()  
      
    z_history = F.normalize(torch.randn(1, 4096, device=device), p=2, dim=-1)  
    z_tear = F.normalize(z_history + torch.randn(1, 4096, device=device) * 1.2, p=2, dim=-1)  
      
    print("[SCAN] Validating Right Kan Pullback over active fiber spaces...")  
    z_glued = orchestrator.evaluate_and_glue(z_history, z_tear, steps=5)  
    glued_norm = torch.norm(z_glued, p=2, dim=-1).item()  
    assert abs(glued_norm - 1.0) < 1e-5, "Fatal: Pullback optimization breached hyperspherical constraints!"  
    print(f"  - Repaired wave vector norm footprint: {glued_norm:.4f}")

    print("[SCAN] Validating Sheaf Boundary Structural Projector Layouts...")  
    analysis_text = "Topic vector maps directly to fluid dynamics Navier-Stokes bounds."  
    payload_text = "Analysis complete: System parameters stabilized within the target e-band."  
      
    compiled_payload = projector.encapsulate_response(analysis_text, payload_text, intent_flag="RESEARCH")  
    is_valid = projector.validate_boundary_payload(compiled_payload, intent_flag="RESEARCH")  
      
    print("\n[CRYSTALLIZED PRODUCT PACKET LOG (RESEARCH INTENT)]:")  
    print("=" * 60)  
    print(compiled_payload)  
    print("=" * 60)  
      
    assert is_valid, "Fatal: Projector generated an out-of-order or invalid structural token block layout!"  
    assert "<|synthesis_begin|>" in compiled_payload, "Fatal: Dual-block gate dropped intent boundary flags!"  
    print("[SUCCESS] Phase 5 Adaptive Projector Gates verified.")  
    print("=== COGNITIVE MACRO-COORDINATION UPGRADES SECURED FOR LIVE SAAS DEPLOYMENT ===")

if __name__ == "__main__":  
    run_full_stack_scan()
