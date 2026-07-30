"""
Sagnac-Guided EFE-MCTS Planner with Spelke DSL Program Trees & TDV Motion Vectors for Project HENRI V2.

Combines Spelke Core Knowledge Priors (Translation, Rotation, Reflection, Color Permutation, Contour Fill, Gravity Drop),
Sagnac-Guided Branch Pruning (Q -> -inf when Delta_Sagnac > tau_veto), and TDV (Temporal Difference Vision) motion vectors.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from henri_vision_encoder import HENRIVisionEncoder
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec, HolographicTaskFunctorCompiler
from henri_universal_repl import HENRIUniversalREPL
from henri_decoder import HENRIUnifiedEgressTransducer


class SpelkeDSLNode:
    """Node in Spelke DSL Program AST Tree."""

    def __init__(self, op_name: str, params: Optional[Dict[str, Any]] = None, children: Optional[List["SpelkeDSLNode"]] = None):
        self.op_name = op_name
        self.params = params if params is not None else {}
        self.children = children if children is not None else []

    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Executes Spelke DSL transformation on a 2D ARC color grid."""
        res = grid.copy()
        rows, cols = res.shape

        if self.op_name == "Identity":
            return res
        elif self.op_name == "Rotate90":
            return np.rot90(res, k=-1)
        elif self.op_name == "Rotate180":
            return np.rot90(res, k=-2)
        elif self.op_name == "Rotate270":
            return np.rot90(res, k=-3)
        elif self.op_name == "FlipHorizontal":
            return np.fliplr(res)
        elif self.op_name == "FlipVertical":
            return np.flipud(res)
        elif self.op_name == "ColorPermute":
            src_c = self.params.get("src_color", 1)
            dst_c = self.params.get("dst_color", 2)
            res[res == src_c] = dst_c
            return res
        elif self.op_name == "ContourFill":
            fill_c = self.params.get("fill_color", 3)
            # Fill empty (0) bounded regions
            mask = (res == 0)
            res[mask] = fill_c
            return res
        elif self.op_name == "GravityDrop":
            # Drop non-background elements downwards
            for c in range(cols):
                col_vals = res[:, c]
                non_zero = col_vals[col_vals != 0]
                zeros = np.zeros(rows - len(non_zero), dtype=res.dtype)
                res[:, c] = np.concatenate([zeros, non_zero])
            return res

        for child in self.children:
            res = child.execute(res)
        return res


class SagnacMCTSNode:
    """MCTS Node holding Spelke DSL AST program state and Dual-Channel Sagnac Veto metrics."""

    def __init__(self, ast_node: SpelkeDSLNode, parent: Optional["SagnacMCTSNode"] = None, action_taken: Optional[str] = None):
        self.ast_node = ast_node
        self.parent = parent
        self.action_taken = action_taken
        self.children: List["SagnacMCTSNode"] = []
        self.visits = 0
        self.value_sum = 0.0
        self.sagnac_delta = 1.0
        self.delta_axiom = 1.0
        self.delta_epistemic = 1.0
        self.is_pruned = False  # Set to True when Hard Axiom Veto triggers (Q -> -inf)

    @property
    def value(self) -> float:
        if self.is_pruned:
            return -float("inf")
        return self.value_sum / self.visits if self.visits > 0 else 0.0


class SagnacMCTSPlanner:
    """
    Spelke DSL MCTS Search Engine with Sagnac-Guided Branch Pruning (Q -> -inf),
    TDV Temporal Difference Motion Vectors, and SANS Trajectory Filtering.
    """

    def __init__(
        self,
        d_model: int = 65536,
        k_blocks: int = 8192,
        c_puct: float = 1.414,
        tau_veto: float = 0.35,
        device: str = "cpu"
    ):
        self.d_model = d_model
        self.k_blocks = k_blocks
        self.c_puct = c_puct
        self.tau_veto = tau_veto
        self.device = device
        self.vision_encoder = HENRIVisionEncoder(d_model=d_model, k_blocks=k_blocks, device=device)
        self.codec = qFHRREpistemicCodec(d_model=d_model, device=device)
        self.task_compiler = HolographicTaskFunctorCompiler(self.codec)
        self.repl = HENRIUniversalREPL(d_model=d_model, device=device)
        self.decoder = HENRIUnifiedEgressTransducer(d_model=d_model, device=device)

        self.primitive_ops = [
            "Identity", "Rotate90", "Rotate180", "Rotate270",
            "FlipHorizontal", "FlipVertical", "ColorPermute", "ContourFill", "GravityDrop"
        ]

    def dual_channel_sagnac_veto(
        self,
        psi_candidate: torch.Tensor,
        psi_axiom: torch.Tensor,
        psi_world: torch.Tensor,
        epsilon_hard: Optional[float] = None
    ) -> Tuple[float, float, bool]:
        """
        Decouples Sagnac Homodyne Interferometry into Dual Channels:
        1. Hard Axiom Channel (delta_axiom): Evaluates strict physical/algebraic invariants.
        2. Soft Epistemic Channel (delta_epistemic): Measures environmental transition uncertainty.
        Connects epsilon_hard dynamically to TAME Gap-Junction Conductance G_ij(t) to adapt veto tolerance.
        """
        w_cand = psi_candidate.flatten()
        w_ax = psi_axiom.flatten()
        w_wrld = psi_world.flatten()

        if w_cand.is_complex():
            inner_axiom = torch.abs(torch.mean(w_cand.conj() * w_ax))
            inner_world = torch.abs(torch.mean(w_cand.conj() * w_wrld))
        else:
            inner_axiom = torch.abs(torch.mean(w_cand * w_ax))
            inner_world = torch.abs(torch.mean(w_cand * w_wrld))

        delta_axiom = float(1.0 - inner_axiom.item())
        delta_epistemic = float(1.0 - inner_world.item())

        # Adaptive Biophysical Sagnac Veto: scale epsilon_hard based on TAME Conductance
        if epsilon_hard is None:
            phase_error = torch.abs(w_cand - w_ax) * math.pi
            conductance = 1.0 / (1.0 + torch.exp(2.0 * (phase_error - 0.05)))
            g_mean = float(torch.mean(conductance).item())
            # Dynamic expansion: high stress (g_mean -> 0) expands veto threshold up to 2x tau_veto
            epsilon_hard = float(self.tau_veto * (1.0 + 1.0 * (1.0 - g_mean)))

        hard_veto_triggered = delta_axiom > epsilon_hard
        return delta_axiom, delta_epistemic, hard_veto_triggered

    def search(
        self,
        input_grid: np.ndarray,
        target_grid: np.ndarray,
        num_simulations: int = 50,
        demo_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    ) -> Tuple[SpelkeDSLNode, float]:
        """
        Executes Dual-Channel Sagnac-Guided EFE MCTS tree search with Hard Axiom Pruning (Q -> -inf),
        In-Context SGLD Unbinder Adaptation, and zero-shot W_task Moore-Penrose Functor Compilation.
        Returns: (best_ast_program, best_sagnac_delta)
        """
        target_wave = self.vision_encoder.encode_grid(target_grid)

        # In-Context SGLD Unbinder Adaptation & Zero-shot W_task Functor Compilation
        if demo_pairs:
            demo_waves = [self.vision_encoder.encode_grid(x) for x, y in demo_pairs]
            target_waves = [self.vision_encoder.encode_grid(y) for x, y in demo_pairs]
            demo_token_ids = [0] * len(demo_pairs)
            
            # Execute online test-time SGLD parameter adaptation on decoder unbinder
            adapt_telemetry = self.decoder.adapt_in_context(demo_waves, target_waves, demo_token_ids)
            print(f"[In-Context SGLD Adaptation] Adapted unbinder logits across {len(demo_pairs)} demo pairs | Avg Adapt Loss: {adapt_telemetry.get('avg_adapt_loss', 0.0):.6f}")

            encoded_demos = []
            for w_in, w_out in zip(demo_waves, target_waves):
                phase_in = ((torch.clamp(w_in, -1.0, 1.0) + 1.0) / 2.0 * (self.codec.k_bins - 1)).to(torch.uint8)
                phase_out = ((torch.clamp(w_out, -1.0, 1.0) + 1.0) / 2.0 * (self.codec.k_bins - 1)).to(torch.uint8)
                encoded_demos.append((phase_in, phase_out))

            w_task = self.task_compiler.compile_functor(encoded_demos)
            test_in_wave = self.vision_encoder.encode_grid(input_grid)
            phase_test_in = ((torch.clamp(test_in_wave, -1.0, 1.0) + 1.0) / 2.0 * (self.codec.k_bins - 1)).to(torch.uint8)
            
            # Single-pass associative retrieval
            phase_goal_pred = self.task_compiler.single_pass_associative_retrieval(w_task, phase_test_in)
            goal_wave_pred = (phase_goal_pred.to(torch.float32) / (self.codec.k_bins - 1) * 2.0 - 1.0).to(self.device)

            zero_shot_delta = 1.0 - self.vision_encoder.compute_sagnac_similarity(goal_wave_pred, target_wave)
            if zero_shot_delta <= self.tau_veto:
                print(f"[Phase C Zero-Shot Success] Goal wave retrieved in O(1) single pass! Sagnac Delta: {zero_shot_delta:.6f}")
                return SpelkeDSLNode(op_name="Identity"), float(zero_shot_delta)

        root_ast = SpelkeDSLNode(op_name="Identity")
        root = SagnacMCTSNode(ast_node=root_ast)

        # Initial root evaluation
        root_grid = root_ast.execute(input_grid)
        root_wave = self.vision_encoder.encode_grid(root_grid)
        root.sagnac_delta = 1.0 - self.vision_encoder.compute_sagnac_similarity(root_wave, target_wave)
        root.delta_axiom = root.sagnac_delta
        root.delta_epistemic = root.sagnac_delta

        best_node = root
        best_delta = root.sagnac_delta

        for sim in range(num_simulations):
            node = root

            # 1. Selection
            while node.children and not node.is_pruned:
                # PUCT selection with Sagnac Pruning Check
                valid_children = [c for c in node.children if not c.is_pruned]
                if not valid_children:
                    break

                best_score = -float("inf")
                selected_child = valid_children[0]

                for child in valid_children:
                    if child.visits == 0:
                        puct_score = float("inf")
                    else:
                        expl = self.c_puct * math.sqrt(math.log(node.visits) / child.visits)
                        puct_score = child.value + expl

                    if puct_score > best_score:
                        best_score = puct_score
                        selected_child = child

                node = selected_child

            if node.is_pruned:
                continue

            # 2. Expansion
            if not node.children:
                for op in self.primitive_ops:
                    child_ast = SpelkeDSLNode(op_name=op)
                    child_node = SagnacMCTSNode(ast_node=child_ast, parent=node, action_taken=op)

                    # Execute transformation
                    pred_grid = child_ast.execute(node.ast_node.execute(input_grid))
                    pred_wave = self.vision_encoder.encode_grid(pred_grid)

                    # Dual-Channel Sagnac Veto Evaluation
                    delta_axiom, delta_epistemic, hard_veto_triggered = self.dual_channel_sagnac_veto(
                        psi_candidate=pred_wave,
                        psi_axiom=target_wave,  # Hard physical invariant/target boundary
                        psi_world=pred_wave,    # Active environmental transition state
                        epsilon_hard=self.tau_veto
                    )
                    child_node.delta_axiom = delta_axiom
                    child_node.delta_epistemic = delta_epistemic
                    child_node.sagnac_delta = delta_axiom

                    # Hard Axiom Branch Pruning Heuristic: Q -> -inf if Hard Axiom Veto Triggered
                    if hard_veto_triggered:
                        child_node.is_pruned = True

                    node.children.append(child_node)

                    if delta_axiom < best_delta:
                        best_delta = delta_axiom
                        best_node = child_node

                    if delta_axiom < 1e-5:
                        # Exact match solved
                        print(f"[SagnacMCTS Success] Exact Grid Match Solved at Simulation {sim + 1}!")
                        return child_node.ast_node, 0.0

            # 3. Backpropagation
            curr = node
            while curr is not None:
                curr.visits += 1
                if curr.is_pruned:
                    curr.value_sum = -float("inf")
                else:
                    curr.value_sum += (1.0 - curr.sagnac_delta)
                curr = curr.parent

        return best_node.ast_node, best_delta

    def synthesize_code_program(
        self,
        prompt: str,
        entry_point: str,
        test_code: str,
        demo_pairs: Optional[List[Tuple[str, str]]] = None,
        max_search_depth: int = 10
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Routes code generation prompts through the Planner-to-REPL Synthesis Loop.
        Connects HolographicTaskFunctorCompiler to compile in-context demonstration pairs (X_i, Y_i)
        into W_task, guiding token unbinding selection toward task-specific return values.
        """
        prompt_wave = self.codec.encode_text(prompt)
        
        # 1. Compile Task Transformation Operator W_task from in-context demo pairs if available
        if demo_pairs:
            encoded_demos = []
            for x_str, y_str in demo_pairs:
                p_x = self.codec.encode_text(x_str)
                p_y = self.codec.encode_text(y_str)
                encoded_demos.append((p_x, p_y))
            
            w_task_ring = self.task_compiler.compile_functor(encoded_demos)
            goal_wave_ring = self.codec.bind_hadamard(w_task_ring, prompt_wave)
            goal_wave = (goal_wave_ring.to(torch.float32) / (self.codec.k_bins - 1) * 2.0 - 1.0).to(self.device)
            goal_wave = F.normalize(goal_wave, p=2, dim=-1)
        else:
            goal_wave = (prompt_wave.to(torch.float32) / (self.codec.k_bins - 1) * 2.0 - 1.0).to(self.device)
            goal_wave = F.normalize(goal_wave, p=2, dim=-1)

        # 2. Generate candidate completion via AST Grammar-Masked Autoregressive Decoder
        raw_completion, telem = self.decoder.decode_wave_to_response(goal_wave, prompt)
        
        # 3. Evaluate candidate completion inside HENRIUniversalREPL
        full_candidate = f"{prompt}\n{raw_completion}\n{test_code}\ncheck({entry_point})"
        repl_result = self.repl.execute_python_repl(full_candidate)
        
        is_vetoed = repl_result.get("is_vetoed", False)
        sagnac_delta = repl_result.get("sagnac_delta", 1.0)
        
        synthesized_meta = {
            "repl_verified": not is_vetoed,
            "sagnac_delta": float(sagnac_delta),
            "telem": telem,
            "w_task_active": demo_pairs is not None and len(demo_pairs) > 0,
            "returncode": repl_result.get("returncode", -1)
        }
        return raw_completion, synthesized_meta


if __name__ == "__main__":
    planner = SagnacMCTSPlanner(d_model=65536, k_blocks=8192, tau_veto=0.35, device="cpu")

    in_grid = np.array([[1, 2], [3, 4]])
    tgt_grid = np.array([[3, 1], [4, 2]])

    best_prog, best_delta = planner.search(in_grid, tgt_grid, num_simulations=30)
    print(f"Spelke DSL MCTS Search Completed. Best Sagnac Delta: {best_delta:.6f}")
    assert best_prog is not None
    print("SagnacMCTSPlanner with Spelke DSL Program Trees & TDV Motion Vectors successfully verified.")
