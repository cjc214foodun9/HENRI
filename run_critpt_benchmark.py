import os
import sys
import json
import torch
import numpy as np
import inspect
import argparse
from pathlib import Path

# Add paths to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "6"))

# Add CritPt src to path
CRITPT_SRC = r"c:\Users\chan\Desktop\henri gemma\CritPt-main\CritPt-main\src"
if os.path.exists(CRITPT_SRC):
    sys.path.insert(0, CRITPT_SRC)
else:
    print(f"[WARNING] CritPt src directory not found at: {CRITPT_SRC}")

from cognitive_swarm import HenriCognitiveSwarmOrchestrator, AletheiaAgent
try:
    from critpt.submission import Submission
    HAS_CRITPT = True
except ImportError:
    HAS_CRITPT = False
    print("[WARNING] Could not import Submission from critpt. Outputting raw JSONs instead.")

def verify_critpt_candidate(agent, candidate, code_template, target_label):
    """
    Verifier for CritPt solutions.
    Executes the candidate python block in UniversalREPL, validates that def answer()
    is defined, calls it with appropriate symbols, and checks boundary invariants.
    """
    if "<|python_begin" not in candidate or "<|python_end|>" not in candidate:
        return False, "Error: The solution does not contain a Python code block enclosed in <|python_begin|> and <|python_end|> tags.", None

    idx_begin = candidate.find("<|python_begin")
    idx_end = candidate.find("<|python_end|>")
    idx_close_bracket = candidate.find("|>", idx_begin)
    if idx_close_bracket == -1 or idx_close_bracket >= idx_end:
        return False, "Error: Invalid python block tags.", None

    code_block = candidate[idx_close_bracket + 2 : idx_end].strip()

    # Clear REPL locals to ensure clean execution and reload physics stack
    agent.orchestrator.repl.console.locals.clear()
    agent.orchestrator.repl.pre_load_physics_stack()

    # Run the code block statefully
    res = agent.orchestrator.repl.execute_block(code_block)
    if not res["success"]:
        err = res["error_message"].strip() or res["stderr"].strip()
        return False, f"REPL Execution Error: Code block compilation or execution failed with error: {err}", None

    locals_dict = agent.orchestrator.repl.console.locals
    if "answer" not in locals_dict or not callable(locals_dict["answer"]):
        return False, "Error: The code block did not define the function 'answer()'. Ensure def answer(...) is present.", None

    # Dynamically call answer() using SymPy symbols in locals
    func = locals_dict["answer"]
    try:
        sig = inspect.signature(func)
        args = []
        for param in sig.parameters.values():
            val = locals_dict.get(param.name)
            if val is not None:
                args.append(val)
            else:
                import sympy as sp
                args.append(sp.symbols(param.name))
        
        func_res = func(*args)
        print(f"[VERIFIER] Executed answer() successfully -> {str(func_res)[:120]}...")
    except Exception as e:
        import traceback
        err_trace = traceback.format_exc()
        return False, f"REPL Execution Error: Calling answer() raised an exception:\n{e}\nTrace:\n{err_trace}", None

    # Project to 4096-D complex wave using L3 Router
    emb_res = agent.orchestrator.base_model.create_embedding(candidate)
    h_7b_raw = torch.tensor(emb_res["data"][0]["embedding"], dtype=torch.float32)
    h_7b_lora = agent.orchestrator.lora_managers[0].apply_lora(h_7b_raw)
    
    psi_candidate = agent.orchestrator.l3_router.activation_to_wave(h_7b_lora)
    if len(psi_candidate.shape) == 2:
        psi_candidate = torch.mean(psi_candidate, dim=0)

    # Validate holographic boundaries
    target_vector = agent.orchestrator.hopfield.vocabulary.get(target_label)
    if target_vector is None:
        target_vector = agent.orchestrator.get_stream_address(0)
        
    target_np = target_vector.detach().numpy().astype(np.complex64)
    psi_cand_np = psi_candidate.detach().numpy().astype(np.complex64)

    truth_np, delta_np, alignment = agent.orchestrator.optical_core.forward(
        hr_wavefront=psi_cand_np,
        target_manifold=target_np,
        langevin_heat=0.0
    )

    truth_tensor = torch.tensor(truth_np, dtype=torch.complex64)
    is_valid, veto_reason, error_energy, h_cft = agent.orchestrator.boundary_validator.validate_boundary(truth_tensor)

    if not is_valid:
        feedback = f"Sagnac Veto: The candidate logic violated Dirichlet boundary axioms. Reason: {veto_reason} | Error Energy: {error_energy:.4f}"
        return False, feedback, delta_np

    return True, "Code execution verified and boundary axioms satisfied.", delta_np

class CritPtAletheiaAgent(AletheiaAgent):
    def __init__(self, orchestrator, code_template):
        super().__init__(orchestrator)
        self.code_template = code_template

    def verify(self, candidate, target_label="SCADA_Pressure_Control") -> tuple:
        return verify_critpt_candidate(self, candidate, self.code_template, target_label)

def run_benchmark(mode="mock", challenge_ids=[1], revisions=2):
    print("=====================================================================")
    print(f"             HENRI CRITPT BENCHMARK RUNNER (MODE: {mode.upper()})    ")
    print("=====================================================================")

    # Initialize the core orchestrator
    if mode == "real":
        model_path = "gemma-4-E4B-it-Q4_0.gguf"
        gemma_dim = 2560
    else:
        model_path = "mock_only.gguf"
        gemma_dim = 2560 # Match the dimensions of loaded LoRAs

    print(f"[SYSTEM] Initializing swarm orchestrator with model_path={model_path}...")
    orchestrator = HenriCognitiveSwarmOrchestrator(
        model_path=model_path,
        num_streams=16,
        gemma_dim=gemma_dim
    )

    # Locate the challenge folder
    challenge_dir = Path(r"c:\Users\chan\Desktop\henri gemma\CritPt-main\CritPt-main\data\public_test_challenges\json")
    if not challenge_dir.exists():
        print(f"[ERROR] Challenge directory does not exist: {challenge_dir}")
        sys.exit(1)

    output_dir_base = Path("results/generations")
    print(f"[SYSTEM] Saving submissions under prefix: {output_dir_base}")

    for cid in challenge_ids:
        c_name = f"Challenge_{cid}"
        c_file = challenge_dir / f"{c_name}.json"
        if not c_file.exists():
            print(f"[WARNING] Challenge file not found: {c_file}. Skipping.")
            continue

        print(f"\n--- Loading {c_name} ---")
        with open(c_file, 'r', encoding='utf-8') as f:
            c_data = json.load(f)

        for problem in c_data.get("problems", []):
            problem_id = problem["problem_id"]
            p_desc = problem["problem_description"]
            code_template = problem["code_template"]

            print(f"\n[Aletheia Harness] Solving problem: {problem_id}")
            
            # 1. Instantiate the specialized Aletheia agent with the current code template
            agent = CritPtAletheiaAgent(orchestrator, code_template)

            # 2. Build instructions and prompt
            prompt = (
                f"Solve the following physics research problem and write the final solution in python.\n\n"
                f"Problem Description:\n{p_desc}\n\n"
                f"You must complete the code template below. Fill in the missing logic indicated by `...`.\n"
                f"Ensure you define the function answer(...) exactly as requested.\n\n"
                f"Code Template:\n{code_template}\n\n"
                f"Write your complete python script inside these block tags:\n"
                f"<|python_begin: heat=0.0|>\n"
                f"# Your complete Python script with def answer(...)\n"
                f"<|python_end|>\n"
            )

            # 3. Execute the closed-loop reasoning process
            # Target label corresponds to physics/attractor concepts in Hopfield Network
            target_label = "Thermodynamic_Conservation"
            candidate, used_revisions, status = agent.execute_reasoning_loop(
                prompt, 
                target_label=target_label, 
                max_revisions=revisions
            )

            print(f"[Aletheia Harness] Solved. Status: {status} | Revisions used: {used_revisions}")

            # 4. Extract the final Python code block
            final_code = ""
            if "<|python_begin" in candidate and "<|python_end|>" in candidate:
                idx_begin = candidate.find("<|python_begin")
                idx_end = candidate.find("<|python_end|>")
                idx_close_bracket = candidate.find("|>", idx_begin)
                if idx_close_bracket != -1 and idx_close_bracket < idx_end:
                    final_code = candidate[idx_close_bracket + 2 : idx_end].strip()
            
            if not final_code:
                # Fallback to candidate itself
                final_code = candidate

            # 5. Format standard Submission JSON
            config_meta = {
                "use_golden_for_prev_steps": False,
                "parsing": False,
                "multiturn_with_answer": False,
                "use_python": True,
                "use_web_search": False
            }

            model_name = "henri_gemma" if mode == "real" else "henri_gemma_mock"
            
            # Recreate path structure expected by evaluate_all_results.py:
            # results/generations/<dataset_name>/use_golden_for_prev_steps_False/parsing_False/multiturn_with_answer_False/<model_name>/0/<problem_id>.json
            output_dir = (
                output_dir_base / 
                c_name / 
                "use_golden_for_prev_steps_False" / 
                "parsing_False" / 
                "multiturn_with_answer_False" / 
                model_name / 
                "0"
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{problem_id}.json"

            if HAS_CRITPT:
                # Use Submission class for standardized formatting
                submission = Submission(
                    problem_id=problem_id,
                    generated_code=final_code,
                    model=model_name,
                    timestamp=torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else "2026-06-06T00:00:00",
                    generation_config=config_meta,
                    messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": candidate}]
                )
                # Save henri telemetry alongside the submission for diagnostic logs
                submission.henri_telemetry = {
                    "mode": mode,
                    "revisions": used_revisions,
                    "status": status,
                    "target_label": target_label
                }
                submission.to_json(output_file)
            else:
                raw_submission = {
                    "problem_id": problem_id,
                    "generated_code": final_code,
                    "model": model_name,
                    "timestamp": "2026-06-06T00:00:00",
                    "generation_config": config_meta,
                    "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": candidate}]
                }
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    json.dump(raw_submission, out_f, indent=2, ensure_ascii=False)

            print(f"[Aletheia Harness] Submission JSON successfully written to: {output_file}")

    print("\n=======================================================")
    print("          BENCHMARK RUN COMPLETED SUCCESSFULLY         ")
    print("=======================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HENRI CritPt Benchmark integration runner.")
    parser.add_argument("--mode", type=str, default="mock", choices=["mock", "real"], help="Execution mode (mock or real)")
    parser.add_argument("--challenges", type=str, default="1,2", help="Comma-separated challenge numbers to evaluate (e.g. 1,2)")
    parser.add_argument("--revisions", type=str, default="2", help="Maximum revisions in Aletheia verifier loop")
    
    args = parser.parse_args()
    
    c_ids = [int(x.strip()) for x in args.challenges.split(",") if x.strip()]
    max_revs = int(args.revisions)
    
    run_benchmark(mode=args.mode, challenge_ids=c_ids, revisions=max_revs)
