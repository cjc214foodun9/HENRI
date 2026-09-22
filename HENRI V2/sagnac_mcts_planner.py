"""
Sagnac-Guided EFE-MCTS Planner with Spelke DSL Program Trees & TDV Motion Vectors for Project HENRI V2.

Combines Spelke Core Knowledge Priors (Translation, Rotation, Reflection, Color Permutation, Contour Fill, Gravity Drop),
Sagnac-Guided Branch Pruning (Q -> -inf when Delta_Sagnac > tau_veto), and TDV (Temporal Difference Vision) motion vectors.
"""

import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from henri_vision_encoder import HENRIVisionEncoder
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec, HolographicTaskFunctorCompiler
from henri_universal_repl import HENRIUniversalREPL
from henri_decoder import HENRIUnifiedEgressTransducer
from efe_planner import INTACTIsomorphicConjugacyHead


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
        self.decoder = HENRIUnifiedEgressTransducer(
            d_model=d_model,
            device=device,
            checkpoint_policy="disabled" if d_model != 65536 else "required",
        )
        self.intact_head = INTACTIsomorphicConjugacyHead(d_model=d_model, device=device)

        self.primitive_ops = [
            "Identity", "Rotate90", "Rotate180", "Rotate270",
            "FlipHorizontal", "FlipVertical", "ColorPermute", "ContourFill", "GravityDrop"
        ]

    # ------------------------------------------------------------------ scale
    # SAGNAC SCALE DEFECT -- FIXED 2026-10-12.
    #
    # This method used `torch.mean(w_cand.conj() * w_ref)`, which is only a valid
    # inner product when the waves are qFHRR-style UNIT-MODULUS (each |w_n| = 1, so
    # ||w||_2 = sqrt(D)). For L2-NORMALIZED waves (||w||_2 = 1, each |w_n| = 1/sqrt(D))
    # the mean is the inner product DIVIDED BY D:
    #
    #     delta = 1 - |<a,b>| / D          ->  1 - O(1/D) for EVERY input
    #
    # MEASURED at D = 1024 (experiments/verification/sagnac_scale_defect.py):
    #     identical pair   delta 0.999023   (should be 0.0)
    #     orthogonal pair  delta 0.999982
    #     random pair      delta 0.999954
    #     range across those cases 9.59e-04   <- flat, carries no information
    #
    # CONSEQUENCE, measured: `hard_veto_triggered = delta_axiom > epsilon_hard` with
    # epsilon_hard = 0.35 fired on 9/9 root children (prune rate 1.000), so the MCTS
    # could not expand. Meanwhile the ROOT is scored by
    # HENRIVisionEncoder.compute_sagnac_similarity, which uses the CORRECT real
    # convention, so the root looked healthy (0.497) while every child looked
    # catastrophic (~0.999). Net effect: search() always returned the Identity
    # program. Two incompatible conventions in one file was the bug.
    #
    # This defect is ALREADY DOCUMENTED in this repository and is NOT a new
    # discovery: arc_sagnac_veto.py records it as "FALSIFIED ... (OBSERVED
    # 2026-08-12, worktree fdb7fd3)" and henri_dual_speed_harness.py re-confirmed it
    # 2026-08-18. The canonical norm-consistent similarity already exists there as
    # `_sagnac_similarity`. This patch applies that canonical form here, with
    # explicit normalization so BOTH wave conventions are handled by one formula.
    #
    # To reproduce the legacy scale for A/B, set HENRI_SAGNAC_LEGACY_SCALE=1. The
    # legacy branch is kept deliberately so the defect stays reproducible as
    # evidence, exactly as the random-axiom measure is kept in henri_wave_readout.
    def _norm_consistent_similarity(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Sagnac homodyne similarity S in [0, 1], scale-consistent. Identical -> 1.

        Real waves:      S = 0.5 * (1 + <a,b> / (||a|| ||b||)), so S = 1 at identity.
        Complex waves:   S = |<a,b>| / (||a|| ||b||), so S = 1 at identity.

        The explicit norm division is the fix: it makes the formula correct for
        L2-normalized waves as well as for unit-modulus qFHRR waves, instead of
        silently depending on which convention the caller used.
        """
        na = float(a.norm().item())
        nb = float(b.norm().item())
        if na < 1e-12 or nb < 1e-12:
            return 0.0                      # fail closed on a zero-energy wave
        if a.is_complex() or b.is_complex():
            val = torch.abs((a.conj() * b).sum()).item() / (na * nb)
            return float(min(max(val, 0.0), 1.0))
        val = (a * b).sum().item() / (na * nb)
        return float(min(max(0.5 * (1.0 + val), 0.0), 1.0))

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

        Both channels now use the scale-consistent similarity (see
        _norm_consistent_similarity). With HENRI_SAGNAC_LEGACY_SCALE=1 the original
        `1 - |mean(...)|` scale is restored for A/B comparison and for reproducing
        the recorded defect.
        """
        w_cand = psi_candidate.flatten()
        w_ax = psi_axiom.flatten()
        w_wrld = psi_world.flatten()

        if os.environ.get("HENRI_SAGNAC_LEGACY_SCALE", "0") == "1":
            # LEGACY (DEFECTIVE for L2-normalized waves; kept as evidence).
            if w_cand.is_complex():
                inner_axiom = torch.abs(torch.mean(w_cand.conj() * w_ax))
                inner_world = torch.abs(torch.mean(w_cand.conj() * w_wrld))
            else:
                inner_axiom = torch.abs(torch.mean(w_cand * w_ax))
                inner_world = torch.abs(torch.mean(w_cand * w_wrld))
        else:
            inner_axiom = torch.tensor(
                self._norm_consistent_similarity(w_cand, w_ax))
            inner_world = torch.tensor(
                self._norm_consistent_similarity(w_cand, w_wrld))

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
        num_simulations: int = 50,
        demo_pairs: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        *,
        goal_wave: Optional[torch.Tensor] = None,
    ) -> Tuple[SpelkeDSLNode, float]:
        """
        Executes Dual-Channel Sagnac-Guided EFE MCTS tree search with Hard Axiom Pruning (Q -> -inf),
        In-Context SGLD Unbinder Adaptation, and zero-shot W_task Moore-Penrose Functor Compilation.

        ANSWER-COUPLING REMOVED (2026-10-12, Milestone 1).
            This method previously took `target_grid` -- the held-out output the
            search claims to predict -- and built its scoring reference from it:

                target_wave = self.vision_encoder.encode_grid(target_grid)
                ...
                if zero_shot_delta <= self.tau_veto:
                    return SpelkeDSLNode(op_name="Identity"), float(zero_shot_delta)

            Both the early return AND every expansion score in the tree compared
            a candidate against `target_wave`. The search therefore could not run
            without the answer, and its "success" label was measured to fire on a
            row-shuffled UNRELATED target with byte-identical SGLD trajectories
            (experiments/verification/demo_path_sgld_attribution.json, verdict
            BANNER_IS_ANSWER_COUPLED). `target_grid` is no longer a parameter, so
            answer coupling is now structurally impossible rather than merely
            policed.

        REPLACEMENT CRITERION (pre-prediction information only).
            The scoring reference is the INDUCED GOAL, compiled from the
            demonstration pairs (X_i, Y_i) alone:

                W_task      = compile_functor({(encode(X_i), encode(Y_i))})
                goal_wave   = normalize(decode(W_task @ encode(X_test)))

            A candidate program P is scored by how well it reproduces that
            induced goal:

                delta_axiom = 1 - Sagnac(encode(P(X_test)), goal_wave)

            Every term is available before any prediction about the held-out
            output is made. When no demonstrations are supplied, the caller must
            pass `goal_wave` explicitly (a stated objective), otherwise the call
            raises -- the search never invents an objective and never accepts a
            target grid.

        Returns: (best_ast_program, best_sagnac_delta)
        """
        # --- Build the scoring reference from demonstrations ONLY -------------
        reference_wave: Optional[torch.Tensor] = None

        if demo_pairs:
            demo_waves = [self.vision_encoder.encode_grid(x) for x, y in demo_pairs]
            demo_target_waves = [self.vision_encoder.encode_grid(y) for x, y in demo_pairs]

            # Online test-time SGLD parameter adaptation (C2 corrected protocol:
            # full softmax target distributions + Sagnac phase alignment +
            # scheduled temperature + unit-normalized Langevin noise). The
            # all-zero argmax-label CE-only variant was measured INERT (MBPP
            # run4); this variant demonstrated real internal learning (MBPP
            # run6: isolation 0.888, loss descent, sagnac distance halved).
            # NOTE: the targets here are DEMONSTRATION outputs Y_i, which are
            # given by the task; this is not answer coupling.
            adapt_telemetry = self.decoder.unbinder.adapt_in_context_sgld_wave(
                active_waves=torch.stack(demo_waves),
                target_waves=torch.stack(demo_target_waves),
                steps=500,
                seed=0,
            )
            print(
                f"[In-Context SGLD Adaptation] soft-target protocol across {len(demo_pairs)} demo pairs | "
                f"loss {adapt_telemetry.get('loss_first', 0.0):.6f} -> {adapt_telemetry.get('loss_last', 0.0):.6f} | "
                f"sagnac_dist_final {adapt_telemetry.get('sagnac_dist_final', 0.0):.6f}"
            )

            encoded_demos = []
            for w_in, w_out in zip(demo_waves, demo_target_waves):
                phase_in = ((torch.clamp(w_in, -1.0, 1.0) + 1.0) / 2.0 * (self.codec.k_bins - 1)).to(torch.uint8)
                phase_out = ((torch.clamp(w_out, -1.0, 1.0) + 1.0) / 2.0 * (self.codec.k_bins - 1)).to(torch.uint8)
                encoded_demos.append((phase_in, phase_out))

            w_task = self.task_compiler.compile_functor(encoded_demos)
            test_in_wave = self.vision_encoder.encode_grid(input_grid)
            phase_test_in = ((torch.clamp(test_in_wave, -1.0, 1.0) + 1.0) / 2.0 * (self.codec.k_bins - 1)).to(torch.uint8)

            # Single-pass associative retrieval of the INDUCED goal. The target
            # grid is never consulted.
            phase_goal_pred = self.task_compiler.single_pass_associative_retrieval(
                w_task, phase_test_in)
            goal_wave_pred = (phase_goal_pred.to(torch.float32) / (self.codec.k_bins - 1) * 2.0 - 1.0).to(self.device)
            reference_wave = F.normalize(goal_wave_pred.reshape(-1), p=2, dim=0)

        if reference_wave is None and goal_wave is not None:
            reference_wave = F.normalize(goal_wave.reshape(-1).to(self.device),
                                         p=2, dim=0)

        if reference_wave is None:
            raise ValueError(
                "search() needs a scoring reference that is available BEFORE any "
                "prediction: supply demo_pairs (to compile W_task from the "
                "demonstrations) or an explicit goal_wave. A held-out target grid "
                "is deliberately not accepted -- accepting one is what made the "
                "former target_grid parameter an answer-coupling leak."
            )

        # In-context adaptation reported its own convergence; the tree now runs
        # unconditionally. There is no demonstration-match shortcut: the search
        # must expand autonomously even when the demonstrations contain a solved
        # case, which is the property Milestone 1 exists to guarantee.

        root_ast = SpelkeDSLNode(op_name="Identity")
        root = SagnacMCTSNode(ast_node=root_ast)

        # Initial root evaluation
        root_grid = root_ast.execute(input_grid)
        root_wave = self.vision_encoder.encode_grid(root_grid)
        root.sagnac_delta = 1.0 - self.vision_encoder.compute_sagnac_similarity(
            root_wave, reference_wave)
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

                    # Dual-Channel Sagnac Veto Evaluation. The hard-axiom
                    # reference is the INDUCED GOAL, not the held-out target.
                    delta_axiom, delta_epistemic, hard_veto_triggered = self.dual_channel_sagnac_veto(
                        psi_candidate=pred_wave,
                        psi_axiom=reference_wave,  # induced goal (pre-prediction)
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
                        # Candidate reproduces the induced goal. This is a
                        # convergence test against a DEMONSTRATION-DERIVED
                        # reference, not an answer match.
                        print(f"[SagnacMCTS Converged] Induced goal reached at Simulation {sim + 1}!")
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

    def score(
        self,
        program: SpelkeDSLNode,
        input_grid: np.ndarray,
        target_grid: np.ndarray,
    ) -> float:
        """OFFLINE scorer. Returns the Sagnac delta of a program against a target.

        This is the ONLY method that accepts a target grid. It is deliberately
        separate from `search()` so that the held-out output can never influence
        planning: a caller can score a finished program, but it cannot search
        against the answer. Keep this separation -- merging the two re-creates
        the answer-coupling leak that Milestone 1 removed.
        """
        pred_grid = program.execute(input_grid)
        pred_wave = self.vision_encoder.encode_grid(pred_grid)
        target_wave = self.vision_encoder.encode_grid(target_grid)
        return float(1.0 - self.vision_encoder.compute_sagnac_similarity(
            pred_wave, target_wave))



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
            w_task_vector = (w_task_ring.to(torch.float32) / (self.codec.k_bins - 1) * 2.0 - 1.0).to(self.device)
            goal_wave_ring = self.codec.bind_hadamard(w_task_ring, prompt_wave)
            goal_wave = (goal_wave_ring.to(torch.float32) / (self.codec.k_bins - 1) * 2.0 - 1.0).to(self.device)
            goal_wave = F.normalize(goal_wave, p=2, dim=-1)
        else:
            w_task_vector = None
            goal_wave = (prompt_wave.to(torch.float32) / (self.codec.k_bins - 1) * 2.0 - 1.0).to(self.device)
            goal_wave = F.normalize(goal_wave, p=2, dim=-1)

        # 2. Generate candidate completion via AST Grammar-Masked Autoregressive Decoder with W_task modulation
        raw_completion, telem = self.decoder.decode_wave_to_response(goal_wave, prompt, w_task=w_task_vector)
        
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


def compute_ryu_takayanagi_entropy(
    wave_state: torch.Tensor, partition_size: int | None = None,
) -> torch.Tensor:
    """Phase 8.22 C2: Ryu-Takayanagi holographic entanglement entropy.

    Spec: HENRI-SPEC-2026-08-PHASE8.21-8.22-WIRING (§2.2)
    S_RT = -Tr(rho_A ln rho_A), rho_A = Psi Psi^dag / Tr(Psi Psi^dag),
    where the [D] hypervector is reshaped into (num_blocks, partition_size)
    and rho_A is the normalized Gram matrix (bipartite cut).

    wave_state: [D] complex or real unit hypervector (D = 65,536).
    partition_size: cut width. Default None -> auto = power-of-two
        ~sqrt(D/2) so the reduced matrix is NOT rank-saturated (saturated
        cuts cannot discriminate successors; D37).
    Returns: scalar von Neumann entropy of the reduced density matrix.
    """
    D = wave_state.shape[0]
    if partition_size is None:
        ps = int(2 ** round(math.log2(max(1.0, (D / 2.0) ** 0.5))))
        partition_size = max(32, min(ps, D))
    num_blocks = D // partition_size
    psi_matrix = wave_state.reshape(num_blocks, partition_size)
    rho_A = torch.matmul(psi_matrix, psi_matrix.conj().T)
    tr = torch.trace(rho_A).real.clamp(min=1e-12)
    rho_A = rho_A / tr
    eigvals = torch.linalg.eigvalsh(rho_A)
    eigvals = torch.clamp(eigvals, min=1e-12)
    s_rt = -torch.sum(eigvals * torch.log(eigvals))
    return s_rt


def _reduced_rho(
    wave: torch.Tensor, partition_size: int,
) -> torch.Tensor:
    """Normalized reduced density matrix rho_A = psi psi^dag / Tr."""
    D = wave.shape[0]
    nb = D // partition_size
    m = wave.reshape(nb, partition_size)
    rho = torch.matmul(m, m.conj().T)
    tr = torch.trace(rho).real.clamp(min=1e-12)
    return rho / tr


def compute_rt_information_gain(
    psi_t: torch.Tensor, psi_hat: torch.Tensor,
    partition_size: int | None = None,
) -> torch.Tensor:
    """Delta I_RT(a): holographic structural information gain of a successor.

    D37 (deviation, OBSERVED 2026-08-17): the spec's literal formula
        S_t + S_hat - S_joint   (joint = concat of the two [D] states)
    is mathematically incapable of vanishing on a no-op successor: the
    concat joint of two identical states has the SAME normalized spectrum
    as one state, so no-op gain = S(psi psi^dag) (measured 3.3991, must be
    0). The joint construction double-counts the spectrum.

    Replaced with the Jensen-Shannon divergence of the reduced density
    matrices (the concavity gap of von Neumann entropy):
        Delta I_RT(a) = S((rho_t + rho_hat)/2)
                        - (S(rho_t) + S(rho_hat)) / 2
    This is exactly the entanglement-structure divergence the RT cut
    intends: it is >= 0, vanishes iff rho_t == rho_hat (no-op successor),
    and equals ln 2 for orthogonal successors. Gate G2-8.22 (gain > 0.1)
    is satisfiable with an unsaturated cut (auto partition ~sqrt(D/2)).
    """
    if partition_size is None:
        D = psi_t.shape[0]
        ps = int(2 ** round(math.log2(max(1.0, (D / 2.0) ** 0.5))))
        partition_size = max(32, min(ps, D))
    rho_t = _reduced_rho(psi_t, partition_size)
    rho_hat = _reduced_rho(psi_hat, partition_size)
    rho_mix = (rho_t + rho_hat) / 2.0
    ev_mix = torch.linalg.eigvalsh(rho_mix).clamp(min=1e-12)
    ev_t = torch.linalg.eigvalsh(rho_t).clamp(min=1e-12)
    ev_h = torch.linalg.eigvalsh(rho_hat).clamp(min=1e-12)
    s_mix = -torch.sum(ev_mix * torch.log(ev_mix))
    s_t = -torch.sum(ev_t * torch.log(ev_t))
    s_h = -torch.sum(ev_h * torch.log(ev_h))
    return s_mix - (s_t + s_h) / 2.0


def _verify_rt_entropy() -> int:
    """G2-8.22 self-test: meaningful RT information gain between a state and
    a random successor vs a no-op (identical) successor."""
    torch.manual_seed(0)
    D = 8192
    psi_t = torch.randn(D, dtype=torch.complex64)
    psi_t = psi_t / psi_t.norm()
    psi_noop = psi_t.clone()
    psi_rand = torch.randn(D, dtype=torch.complex64)
    psi_rand = psi_rand / psi_rand.norm()

    gain_noop = float(compute_rt_information_gain(psi_t, psi_noop))
    gain_rand = float(compute_rt_information_gain(psi_t, psi_rand))
    s_rt = float(compute_ryu_takayanagi_entropy(psi_t))

    assert gain_rand > 0.1000, f"G2-8.22 FAIL: random gain {gain_rand:.4f}"
    assert abs(gain_noop) < 0.0100, f"G2-8.22 FAIL: no-op gain {gain_noop:.4f}"
    print(f"[verify_rt_entropy] G2-8.22 PASS: S_RT={s_rt:.4f} "
          f"gain_noop={gain_noop:.4f} gain_rand={gain_rand:.4f}")
    return 0


if __name__ == "__main__":
    import argparse
    import inspect

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--mode", default=None)
    _args = _ap.parse_args()
    if _args.mode == "verify_rt_entropy":
        raise SystemExit(_verify_rt_entropy())

    # ---- Milestone 1 self-test: the planner must run WITHOUT the answer -----
    sig = inspect.signature(SagnacMCTSPlanner.search)
    assert "target_grid" not in sig.parameters, (
        f"search() must not accept a target grid; signature is {sig}")
    print(f"[milestone1] search() signature verified: {list(sig.parameters)}")

    planner = SagnacMCTSPlanner(d_model=8192, k_blocks=1024, tau_veto=0.35, device="cpu")

    in_grid = np.array([[1, 2], [3, 4]])
    demos = [(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),
             (np.array([[5, 6], [7, 8]]), np.array([[5, 6], [7, 8]]))]

    best_prog, best_delta = planner.search(in_grid, num_simulations=4,
                                            demo_pairs=demos)
    print(f"Spelke DSL MCTS Search Completed. Best Sagnac Delta: {best_delta:.6f}")
    assert best_prog is not None

    # Anti-coupling proof: the plan must be identical no matter what the
    # held-out answer is, because the answer is never an input.
    tgt_a = np.array([[3, 1], [4, 2]])
    tgt_b = np.rot90(tgt_a)
    prog_a, delta_a = planner.search(in_grid, num_simulations=4, demo_pairs=demos)
    delta_score_a = planner.score(prog_a, in_grid, tgt_a)
    delta_score_b = planner.score(prog_a, in_grid, tgt_b)
    print(f"[milestone1] plan is target-independent; offline score vs target A "
          f"= {delta_score_a:.6f}, vs target B = {delta_score_b:.6f}")
    assert delta_score_a != delta_score_b, (
        "the offline scorer is not sensitive to the target, so this control "
        "cannot distinguish coupling from its absence")

    # Fails closed when no pre-prediction reference is available.
    try:
        planner.search(in_grid, num_simulations=2)
    except ValueError as exc:
        print(f"[milestone1] fail-closed without a reference: {exc}")
    else:
        raise AssertionError("search() must refuse to run with no reference")

    print("SagnacMCTSPlanner Milestone-1 verification: answer coupling removed.")
