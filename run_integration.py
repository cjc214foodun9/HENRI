import os
import sys
import time
import torch
import numpy as np

# Ensure root paths are in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "6"))

from cognitive_swarm import HenriCognitiveSwarmOrchestrator

def main():
    print("=====================================================================")
    print("             HENRI ASYNCHRONOUS COGNITIVE SWARM SYSTEM ENGINE        ")
    print("=====================================================================")

    # Initialize the core orchestrator
    # We use a small mock model path or let it fallback to mock mode if not present
    orchestrator = HenriCognitiveSwarmOrchestrator(
        model_path="gemma-4-E4B-it-Q4_0.gguf",
        num_streams=16
    )

    # Define initial prompts for the 16 streams
    # One stream will explicitly write some python code to test the REPL/scratchpad
    initial_prompts = {}
    for i in range(16):
        if i == 3:
            # Code-generating stream to test the stateful scratchpad and REPL libraries
            initial_prompts[i] = (
                "Solve the SCADA thermodynamic pressure loop. Code block:\n"
                "<|python_begin: heat=0.3|>\n"
                "import sympy as sp\n"
                "t = sp.Symbol('t')\n"
                "expr = sp.sin(t)**2 + sp.cos(t)**2\n"
                "print(sp.simplify(expr))\n"
                "<|python_end|>"
            )
        else:
            initial_prompts[i] = f"Refine reasoning stream {i} for SCADA pressure control boundary conditions."

    # Start the continuous wave constructionTimed loop (tick every 0.1s for simulation speed)
    orchestrator.start_swarm_loop(initial_prompts, interval=0.1)

    print("\n[SYSTEM] Swarm timed loop running on background thread.")
    print("[SYSTEM] Synchronized series processing of constructed wavefronts starting...")
    
    # Run the processing loop for 10 steps
    max_steps = 10
    success = False
    
    try:
        for step in range(1, max_steps + 1):
            time.sleep(0.15) # Wait for the background loop to construct a wave
            
            # Retrieve and process the next bulk wave constructed by the swarm
            # We target the "SCADA_Pressure_Control" axiom
            result = orchestrator.process_next_wave(target_label="SCADA_Pressure_Control")
            
            status = result["status"]
            if status == "VETOED":
                print(f" -> Tick Result: [VETOED] | Reason: {result['reason']} | Error Energy: {result['error']:.4f}")
                # Simulate the model adjusting its prompt based on rehypothecation directives
                # We inject the lateral variance prompt into the swarm
                for i in range(16):
                    orchestrator.stream_prompts[i] += "\n[Rehypothecation Nudge]: Discard previous syntactic failures. Focus on the conservation invariants."
                    
            elif status == "CONVERGED":
                print(f" -> Tick Result: [CONVERGED] | Concept: '{result['concept']}' | Similarity Confidence: {result['confidence']*100:.2f}%")
                success = True
                break
            elif status == "TIMEOUT":
                print(f" -> Tick Result: [TIMEOUT] | Msg: {result['msg']}")
                
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user.")
    finally:
        # Gracefully stop the background timed loop thread
        orchestrator.stop_swarm_loop()
        
    print("\n=====================================================================")
    if success:
        print("          INTEGRATION VERIFICATION: SUCCESSFUL CONVERGENCE           ")
    else:
        print("          INTEGRATION VERIFICATION: COMPLETED WITH TIMEOUT           ")
    print("=====================================================================")

if __name__ == "__main__":
    main()
