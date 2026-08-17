import torch
import math

class O_VSA_IngressTokenizer:
    """
    Project HENRI: O-VSA Ingress Layer & True Local Tokenizer
    Replaces chaotic phase hashing with rigorous orthogonal mapping.
    Maps discrete string input to Token IDs, and then onto pristine 
    Clifford Multivector Embeddings (CME) of shape [num_blocks, 8].

    Memory-Safe VRAM Strategy:
      For large vocabularies (V >= 1000), canonical_basis is allocated on CPU in float16
      and sliced on-demand to CUDA, reducing GPU VRAM allocation from 8.38 GB to <10 MB.
    """
    def __init__(self, num_blocks: int = 8192, vocab_size: int = 256, device="cpu"):
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        
        # Memory-safe CPU allocation for canonical basis to prevent CUDA OOM
        if vocab_size > 1000:
            chunk_size = 4000
            basis_chunks = []
            for i in range(0, vocab_size, chunk_size):
                sz = min(chunk_size, vocab_size - i)
                chunk = torch.randn(sz, num_blocks, 8, device="cpu", dtype=torch.float16)
                chunk = chunk / torch.norm(chunk.to(torch.float32), p=2, dim=-1, keepdim=True).to(torch.float16)
                basis_chunks.append(chunk)
            self.canonical_basis_cpu = torch.cat(basis_chunks, dim=0)  # CPU [vocab_size, num_blocks, 8] in fp16
            self.canonical_basis = None
        else:
            raw_basis = torch.randn(vocab_size, num_blocks, 8, device=self.device)
            self.canonical_basis = raw_basis / torch.norm(raw_basis, p=2, dim=-1, keepdim=True)
            self.canonical_basis_cpu = None
        
        # O-VSA Spatial Fractional Binding Basis (X and Y phase axes)
        self.spatial_theta_x = (torch.rand(num_blocks, 4, device=self.device) * 2 * math.pi)
        self.spatial_theta_y = (torch.rand(num_blocks, 4, device=self.device) * 2 * math.pi)

    def get_token_vector(self, token_id: int) -> torch.Tensor:
        """Retrieves single token vector on target device."""
        token_id = min(token_id, self.vocab_size - 1)
        if self.canonical_basis_cpu is not None:
            vec_fp16 = self.canonical_basis_cpu[token_id]
            return vec_fp16.to(dtype=torch.float32, device=self.device)
        return self.canonical_basis[token_id]

    def encode(self, text: str) -> torch.Tensor:
        """
        Tokenizes text (character-level byte encoding) and returns corresponding
        Clifford Multivector Embeddings of shape [seq_len, num_blocks, 8] on target device.
        """
        token_ids = [min(ord(c), self.vocab_size - 1) for c in text]
        if self.canonical_basis_cpu is not None:
            indices = torch.tensor(token_ids, dtype=torch.long, device="cpu")
            vecs_fp16 = self.canonical_basis_cpu[indices]
            return vecs_fp16.to(dtype=torch.float32, device=self.device)
        else:
            indices = torch.tensor(token_ids, dtype=torch.long, device=self.device)
            return self.canonical_basis[indices]
        
    def encode_spatial_grid(self, grid: list[list[int]]) -> torch.Tensor:
        """
        Fractional Binding: Bypasses string tokenization. Maps a 2D spatial grid directly
        into a continuous FHRR superposed wave tensor of shape [1, num_blocks, 8].
        """
        superposed_wave = torch.zeros(self.num_blocks, 4, 2, device=self.device)
        
        height = len(grid)
        width = len(grid[0]) if height > 0 else 1
        
        for y, row in enumerate(grid):
            norm_y = (2.0 * y / (height - 1)) - 1.0 if height > 1 else 0.0
            for x, val in enumerate(row):
                norm_x = (2.0 * x / (width - 1)) - 1.0 if width > 1 else 0.0
                
                token_id = min(val, self.vocab_size - 1)
                val_vec = self.get_token_vector(token_id)
                
                val_complex = val_vec.view(self.num_blocks, 4, 2)
                theta_v = torch.atan2(val_complex[..., 1], val_complex[..., 0])
                
                total_phase = theta_v + norm_x * self.spatial_theta_x + norm_y * self.spatial_theta_y
                bound_complex = torch.stack([torch.cos(total_phase), torch.sin(total_phase)], dim=-1)
                
                superposed_wave += bound_complex
                
        superposed_wave = superposed_wave.view(self.num_blocks, 8)
        norm = torch.norm(superposed_wave, p=2, dim=-1, keepdim=True) + 1e-9
        superposed_wave = superposed_wave / norm
        
        return superposed_wave.unsqueeze(0)
        
    def dynamic_ontology_expansion(self) -> int:
        new_vector_cpu = torch.randn(1, self.num_blocks, 8, device="cpu", dtype=torch.float16)
        new_vector_cpu = new_vector_cpu / torch.norm(new_vector_cpu.to(torch.float32), p=2, dim=-1, keepdim=True).to(torch.float16)
        
        if self.canonical_basis_cpu is not None:
            self.canonical_basis_cpu = torch.cat([self.canonical_basis_cpu, new_vector_cpu], dim=0)
        else:
            new_vector_dev = new_vector_cpu.to(dtype=torch.float32, device=self.device)
            self.canonical_basis = torch.cat([self.canonical_basis, new_vector_dev], dim=0)

        new_id = self.vocab_size
        self.vocab_size += 1
        return new_id

    def get_lexicon(self) -> dict:
        lexicon = {chr(i): self.get_token_vector(i) for i in range(min(128, self.vocab_size))}
        return lexicon


def _verify_action_transducer() -> int:
    """Phase 8.21 G1/G3 self-test (spec execution protocol step 2):
    DynamicActionSpaceTransducer fiber expansion — |A_admissible| >= 2 on
    collapsed/empty masks, multi-action passthrough unchanged, RESET excluded
    (D36), no-op predictions pruned (D35 stationarity Sagnac semantics)."""
    from henri_external_outcome_refactor_module import (
        ActionOutcomeGeneratorStore)
    from chromodynamic_grounding import (
        GELL_MANN_BASIS, encode_su3_color_field)

    n_actions = 8
    store = ActionOutcomeGeneratorStore(
        num_actions=n_actions, num_channels=64, lr=0.1)
    basis = GELL_MANN_BASIS
    grid = [[0] * 8 for _ in range(8)]
    grid[0][0] = 3
    field = encode_su3_color_field(
        torch.tensor([grid], dtype=torch.long)).reshape(-1, 3, 3)
    if field.shape[0] < 64:
        eye = torch.eye(3, dtype=field.dtype).unsqueeze(0)
        field = torch.cat([field, eye.repeat(64 - field.shape[0], 1, 1)], 0)
    trans = DynamicActionSpaceTransducer(num_canonical_actions=n_actions)

    collapsed = torch.zeros(n_actions, dtype=torch.bool)
    collapsed[6] = True  # only ACTION6 legal (the 8.19/8.20 stall signature)
    out = trans.resolve_admissible_actions(collapsed, field, store, basis)
    assert out.sum() >= 2, f"G1 FAIL: collapsed -> {int(out.sum())}"
    assert not out[0], "G1 FAIL: RESET must be excluded (D36)"
    print(f"[verify_action_transducer] G1 PASS: collapsed mask -> "
          f"{int(out.sum())} admissible")

    multi = torch.zeros(n_actions, dtype=torch.bool)
    multi[1] = multi[3] = multi[5] = True
    out2 = trans.resolve_admissible_actions(multi, field, store, basis)
    assert torch.equal(out2, multi), "G1 FAIL: multi-action passthrough changed"
    print("[verify_action_transducer] G1 PASS: multi-action passthrough unchanged")

    empty = torch.zeros(n_actions, dtype=torch.bool)
    out3 = trans.resolve_admissible_actions(empty, field, store, basis)
    assert out3.sum() >= 2 and not out3[0], "G3 FAIL: empty-mask fallback"
    print("[verify_action_transducer] G3 PASS: empty-mask fallback >= 2, "
          "RESET excluded")
    return 0


class DynamicActionSpaceTransducer(torch.nn.Module):
    """Phase 8.21 C1: Ingress Action-Space Fiber Transducer (spec
    HENRI-SPEC-2026-08-PHASE8.21-ACTION-SPACE-REFORM).

    Un-collapses single-action environment masks into multi-action fiber
    bundles satisfying |A_admissible| >= 2 so active inference EFE steering
    can re-engage (G1-8.21 / G2-8.21).

    Deviations from spec (registered):
      D35: spec filter sagnac_delta = 1.0 - mean(|det(U_hat)|) is IDENTICALLY 0
           for SU(3) fields (det == 1 exactly) -> vacuous, prunes nothing.
           Replaced by the codebase's own Stationarity Sagnac Veto semantics:
           an action whose predicted field is (near-)identical to the current
           field is a no-op (delta_grid == 0 => Delta_Sagnac = 1.0 => pruned).
           Metric: normalized field displacement
             rel = ||U_hat(a) - U_t||_F / ||U_t||_F
           Actions with rel < noop_eps are pruned (destructive interference).
           This directly targets the measured 8.19/8.20 failure: on collapsed
           envs (ft09 etc.) ACTION6 changes 0 cells (OBSERVED live probe) and
           the zero-init generator predicts U_hat == U_t for every action.
      D36: spec canonical set A_16 = {0..15} does not exist on the live
           arcade. Canonical set = live GameAction vocabulary (RESET +
           ACTION1..ACTION7) with RESET (index 0) permanently excluded from
           fiber candidates (RESET is a control action, not a transformation).
           num_canonical_actions defaults to 8 (decoder vocabulary size).
    """

    def __init__(self, num_canonical_actions: int = 8, noop_eps: float = 1e-3):
        super().__init__()
        self.num_canonical_actions = num_canonical_actions
        self.noop_eps = noop_eps

    @torch.no_grad()
    def resolve_admissible_actions(
        self,
        native_mask: torch.Tensor,
        current_field: torch.Tensor,
        action_store: torch.nn.Module,
        gell_mann_basis: torch.Tensor,
    ) -> torch.Tensor:
        """Expand a collapsed action mask to |A_admissible| >= 2.

        native_mask:  [num_canonical_actions] boolean tensor aligned with the
                      live action vocabulary (index 0 = RESET, excluded).
        current_field: [N, 3, 3] complex SU(3) field (N = num_channels).
        action_store: ActionOutcomeGeneratorStore (C1-8.20) providing
                      predict_next_field(U_t, action_idx, basis).
        Returns:      [num_canonical_actions] boolean tensor with >= 2 True.
        """
        dev = native_mask.device
        admissible_indices = torch.where(native_mask)[0]
        # Native mask already multi-action: pass through unchanged (spec 2.1).
        if len(admissible_indices) >= 2:
            return native_mask

        # Fiber Expansion: un-collapse single-action or empty mask.
        expanded_mask = torch.ones(
            self.num_canonical_actions, dtype=torch.bool, device=dev)

        # RESET (index 0) is never a fiber candidate (D36).
        if self.num_canonical_actions > 0:
            expanded_mask[0] = False

        # Sagnac Homodyne Filtering (D35): prune no-op predictions whose
        # field displacement is below the stationarity threshold.
        if current_field is not None and action_store is not None:
            u_norm = current_field.abs().pow(2).sum(dim=(-2, -1)).sqrt()
            for a in range(1, self.num_canonical_actions):
                try:
                    u_hat = action_store.predict_next_field(
                        current_field, a, gell_mann_basis)
                    rel = (u_hat - current_field).abs().pow(2).sum(
                        dim=(-2, -1)).sqrt().mean() / u_norm.mean().clamp(min=1e-12)
                    if rel < self.noop_eps:
                        # Destructive interference / stationarity: no change
                        # predicted => Delta_Sagnac = 1.0 => prune.
                        expanded_mask[a] = False
                except Exception:
                    # Fail-closed per-candidate: keep the candidate rather than
                    # silently dropping it on a transient prediction error.
                    continue

        # Safety Fallback (spec 2.1 step 4): guarantee at least 2 actions.
        if torch.sum(expanded_mask) < 2:
            # D36: fallback keeps the first two non-RESET actions
            # (ACTION1, ACTION2 in the live vocabulary).
            expanded_mask[1] = True
            if self.num_canonical_actions > 2:
                expanded_mask[2] = True

        return expanded_mask


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--mode", default=None)
    _args = _ap.parse_args()
    if _args.mode == "verify_action_transducer":
        raise SystemExit(_verify_action_transducer())
    raise SystemExit(f"unknown --mode {_args.mode!r} "
                     f"(expected verify_action_transducer)")
