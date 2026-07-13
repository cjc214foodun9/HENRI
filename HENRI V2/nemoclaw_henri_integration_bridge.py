import torch
import json
import time
import asyncio
import numpy as np
from constraint_based_swarm_thermostat import WaveMechanics, ConstraintMatrix, ConstraintBasedThermostat
from semantic_cleanup_matrix import SemanticCleanupMatrix
from openshell_causal_loop_sandbox import NemoClawCausalSandbox
from zone_c_live_epistemic_ignition import ZoneCEpistemicIgnition




class NemoClawEpistemicBridge:
    """
    The Sensory-Motor Boundary.
    Translates discrete LangChain JSON into continuous waves, runs the thermodynamic 
    constraint solver, and unbundles the output into secure NemoClaw execution syntax.
    """
    def __init__(self, db_url: str, dim=4096, swarm_size=32):
        self.dim = dim
        self.device = torch.device("cpu")
        
        # Production Organelles
        self.zone_c = ZoneCEpistemicIgnition(db_url)
        self.sandbox = NemoClawCausalSandbox(strict_mode=True)
        self.thermostat = ConstraintBasedThermostat(swarm_size=swarm_size, dim=dim)
        
        # Instantiate the Semantic Cleanup Matrix (Hopfield Lexicon)
        self.cleanup_matrix = SemanticCleanupMatrix(dim=dim, device=self.device)
        # Production Lexicon for Sandbox Execution
        # In a fully deployed system, this is populated dynamically from Zone C.
        sandbox_lexicon = {
            "RUN_PYTHON_REPL": WaveMechanics.generate_uwe("FILLER_RUN_PYTHON_REPL", dim, self.device),
            "RUN_BASH_COMMAND": WaveMechanics.generate_uwe("FILLER_RUN_BASH_COMMAND", dim, self.device),
            "print": WaveMechanics.generate_uwe("FILLER_PRINT", dim, self.device),
            "(": WaveMechanics.generate_uwe("FILLER_LPAREN", dim, self.device),
            "'Somatic boundaries confirmed.'": WaveMechanics.generate_uwe("FILLER_STRING_1", dim, self.device),
            ")": WaveMechanics.generate_uwe("FILLER_RPAREN", dim, self.device),
            "\\n": WaveMechanics.generate_uwe("FILLER_NEWLINE", dim, self.device),
            "result": WaveMechanics.generate_uwe("FILLER_RESULT", dim, self.device),
            "=": WaveMechanics.generate_uwe("FILLER_EQUALS", dim, self.device),
            "4096": WaveMechanics.generate_uwe("FILLER_4096", dim, self.device),
            "*": WaveMechanics.generate_uwe("FILLER_MULTIPLY", dim, self.device),
            "2": WaveMechanics.generate_uwe("FILLER_2", dim, self.device),
            "ls": WaveMechanics.generate_uwe("FILLER_LS", dim, self.device),
            "-la": WaveMechanics.generate_uwe("FILLER_LA", dim, self.device),
            "/nemoclaw_sandbox/tmp/": WaveMechanics.generate_uwe("FILLER_TMP_DIR", dim, self.device)
        }
        self.cleanup_matrix.assimilate_lexicon(sandbox_lexicon)

    def _json_to_wave(self, payload: dict) -> torch.Tensor:
        """The Universal Epistemic Transducer (Ingress)"""
        print(f"[Ingress] Transducing JSON payload to C^{self.dim} wave...")
        superposition = torch.zeros(self.dim, dtype=torch.complex64, device=self.device)
        
        for key, value in payload.items():
            role_wave = WaveMechanics.generate_uwe(f"ROLE_{key.upper()}", self.dim, self.device)
            filler_wave = WaveMechanics.generate_uwe(f"FILLER_{str(value).upper()}", self.dim, self.device)
            bound = WaveMechanics.circular_bind(role_wave, filler_wave)
            superposition += bound
            
        return superposition / (torch.abs(superposition) + 1e-9)

    def _wave_to_json(self, wave: torch.Tensor) -> dict:
        """Generative Unbundling (Egress) for NemoClaw via Atomic Sequential Extraction"""
        print("[Egress] Unbundling continuous Policy Wave into discrete AST tokens...")
        output_payload = {}
        
        # Extract Action Type
        role_action = WaveMechanics.generate_uwe("ROLE_ACTION_TYPE", self.dim, self.device)
        extracted_action = WaveMechanics.circular_unbind(wave, role_action)
        action_token, _, confidence = self.cleanup_matrix.snap_and_decode(extracted_action, beta=100.0)
        output_payload["action_type"] = action_token
        print(f"   -> Extracted Action Type '{action_token}' with Confidence: {confidence*100:.2f}%")
        
        # Extract Execution Block via Sequential AST Roles
        execution_block = ""
        max_ast_length = 20
        for i in range(1, max_ast_length + 1):
            role_ast = WaveMechanics.generate_uwe(f"ROLE_AST_{i}", self.dim, self.device)
            extracted_ast = WaveMechanics.circular_unbind(wave, role_ast)
            token, _, ast_confidence = self.cleanup_matrix.snap_and_decode(extracted_ast, beta=100.0)
            
            if ast_confidence < 0.15:
                break # Coherence dropped, sequence ended
                
            print(f"   -> Extracted AST Token [{i}]: '{token}' (Confidence: {ast_confidence*100:.2f}%)")
            if token in ["(", ")", "\\n"]:
                execution_block += token.replace("\\n", "\n")
            else:
                execution_block += token + " "
                
        output_payload["execution_block"] = execution_block.strip()
        return output_payload

    async def execute_agentic_loop(self, langchain_payload: dict):
        """
        The Master Integration Loop.
        1. Ingest JSON from LangChain
        2. Fetch Laws from TimescaleDB via Live Ignition
        3. Evolve Swarm until it respects all laws
        4. Excrete secure JSON for NemoClaw
        5. Sandbox Execution & Sagnac Error Feedback
        """
        print("\n" + "="*60)
        print("🔗 INITIATING NEMOCLAW <-> HENRI THERMODYNAMIC BRIDGE")
        print("="*60)
        
        # Connect to DB
        await self.zone_c.connect_and_provision()
        
        # 1. Transduce the environment
        task_wave = self._json_to_wave(langchain_payload)
        
        # 2. Fetch the Epistemic Anchors (Constraints) Live
        # Convert tensor to numpy for DB lookup
        query_np = task_wave.cpu().numpy()
        # Ensure modulus is 1.0 (sometimes it's complex, we only search on real part/magnitudes for DB approximation, 
        # but the actual system would index complex fields. For now we use the real projection.)
        query_real = np.real(query_np)
        query_real = query_real / (np.linalg.norm(query_real) + 1e-9)
        
        fetched_laws = await self.zone_c.fetch_epistemic_adjacency(query_real, limit=3)
        print(f"[Zone C] Holographic Adjacency Lookup Complete. Fetched {len(fetched_laws)} Constraints.")
        
        active_constraints = []
        for law in fetched_laws:
            # Reconstruct complex tensor from DB
            wave_np = law['wavefront']
            # Simplification: assuming real embeddings in DB for test, mapping back to complex tensor
            wave_tensor = torch.tensor(wave_np, dtype=torch.complex64, device=self.device)
            active_constraints.append(wave_tensor)
            
        constraint_matrix = ConstraintMatrix(active_constraints)
        
        # The Sagnac Causal Loop
        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            # 3. Darwinian Thermodynamic Phase (The "Thinking")
            print(f"\n[Thermostat] Igniting {self.thermostat.swarm_size}-Expert Swarm against Constraint Matrix (Attempt {attempt})...")
            max_generations = 300
            start_time = time.time()
            
            for gen in range(1, max_generations + 1):
                # Active Inference Transition: Pass Task Wave into Thermostat
                alpha_expert = self.thermostat.evolutionary_epoch(task_wave, constraint_matrix)
                
                if gen % 25 == 0 or gen == 1:
                    print(f"   [Gen {gen:03d}] Min Constraint Satisfaction: {alpha_expert.fitness:.4f}")
                    
                # Convergence: The FUTURE STATE violates zero known physical/syntax laws.
                if alpha_expert.fitness > 0.995:
                    print(f"\n✅ [Thermostat] CRYSTALLIZATION ACHIEVED AT GENERATION {gen}.")
                    print(f"   Time elapsed: {time.time() - start_time:.2f} seconds.")
                    break

            # 4. Extract for NemoClaw Sandbox Execution
            # The Hopfield Matrix unbundles the raw wave directly into the structural payload
            nemoclaw_json = self._wave_to_json(alpha_expert.wave)
            
            payload_str = json.dumps(nemoclaw_json, indent=2)
            print("\n[NemoClaw] Execution Payload Ready:")
            print(payload_str)
            
            # 5. The Physical Causal Hook
            print("\n[OpenShell] Firing crystallized wave into isolated execution muscle...")
            sagnac_feedback = self.sandbox.execute_crystallized_wave(payload_str)
            
            if sagnac_feedback["is_isothermal_lock"]:
                print(f"✅ [CAUSAL LOOP CLOSED] Execution Succeeded:\n{sagnac_feedback['stdout']}")
                return nemoclaw_json
            else:
                print(f"❌ [CAUSAL LOOP FAILED] Epistemic Surprise Detected:\n{sagnac_feedback['stderr']}")
                print(f"   Injecting {sagnac_feedback['requested_langevin_heat']}T of targeted heat back into Thermostat.")
                
                # Apply the Sagnac Error directly to the constraint matrix to force the swarm out of the hallucination
                error_wave = torch.tensor(sagnac_feedback["error_wavefront_delta"], dtype=torch.complex64, device=self.device)
                constraint_matrix.laws.append(error_wave) 
                # The swarm will now evolve to avoid this specific error state
                
        print("\n[FATAL] Thermodynamic collapse. Failed to resolve constraints within maximum causal attempts.")
        return None

# --- Execution Test ---
async def main():
    import os
    db_url = os.getenv("POSTGRES_DSN", "postgresql://postgres:password@127.0.0.1:5432/henri")
    
    # Simulate an incoming payload from LangChain
    incoming_environment_state = {
        "task": "evaluate_sandbox_execution",
        "permissions": "read_write_isolated"
    }
    
    bridge = NemoClawEpistemicBridge(db_url=db_url, swarm_size=16)
    await bridge.execute_agentic_loop(incoming_environment_state)

if __name__ == "__main__":
    asyncio.run(main())