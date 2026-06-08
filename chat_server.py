import os
import sys
import threading
import torch
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import re

# Ensure paths are set
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cognitive_swarm import HenriCognitiveSwarmOrchestrator
from active_inference_engine import ActiveInferenceSwarmAgent

app = FastAPI(title="HENRI Cognitive Swarm Chatbot API")

def classify_prompt(prompt: str) -> dict:
    prompt_lower = prompt.lower()
    
    # Check if prompt requires strict math / physics / wave verification
    physics_keywords = [
        "scada", "thermodynamic", "pressure", "wave", "metamaterial", "bandgap", 
        "birefringence", "quantum", "wigner", "metric", "sonic", "horizon", 
        "langevin", "ness", "attractor", "dihedral", "d4", "fourier", "translation", 
        "affine", "symmetry", "dirichlet", "neumann", "lipschitz", "veto", "repl", "proof", "derive"
    ]
    
    requires_verification = any(kw in prompt_lower for kw in physics_keywords)
    
    if not requires_verification:
        return {
            "path": "GENERAL",
            "domain": "General_Reasoning",
            "max_revisions": 0,
            "reason": "Intent classified as General Chat (no physical boundary constraints detected)."
        }
        
    # Map prompt keywords to one of our 9 seeded boundary axioms
    domain_mappings = {
        "Acoustic_Metamaterial_Bandgap_Limits": ["bandgap", "metamaterial", "acoustic lattice"],
        "Acoustic_Metric_Sonic_Horizons": ["sonic", "horizon", "acoustic metric"],
        "Birefringence_Kinematic_Scaling": ["birefringence", "kinematic", "scaling", "dilation"],
        "Rigid_Translation_Fourier_Ramp": ["fourier", "translation", "ramp", "shift theorem"],
        "Langevin_Dynamics_NESS_Attractors": ["langevin", "ness", "surprisal", "steady state"],
        "Quantum_Holographic_Wigner_Functions": ["wigner", "quasi-probability", "phase-space"],
        "D4_Symmetry_SU_N_Rotations": ["d4", "dihedral", "rotation", "reflection"],
        "Bounding_Box_Bandpass_Cutoff": ["bounding box", "cutoff", "dirichlet"],
        "SCADA_Pressure_Control": ["scada", "pressure", "valve", "clamping"]
    }
    
    selected_domain = "General_Reasoning"
    for domain, kws in domain_mappings.items():
        if any(kw in prompt_lower for kw in kws):
            selected_domain = domain
            break
            
    # Default to SCADA_Pressure_Control if generic physics/wave was matched but no specific domain
    if selected_domain == "General_Reasoning" and requires_verification:
        selected_domain = "SCADA_Pressure_Control"
        
    max_revisions = 3
    if "scada" in prompt_lower:
        max_revisions = 3
    elif "proof" in prompt_lower or "derive" in prompt_lower:
        max_revisions = 2
        
    return {
        "path": "ALETHEIA_SWARM",
        "domain": selected_domain,
        "max_revisions": max_revisions,
        "reason": f"Intent classified as Symbolic Physics/Math. Auto-matched domain: {selected_domain}."
    }

# Global state
orchestrator = None
agent = None
status_lock = threading.Lock()
startup_status = "Initializing..."
error_message = None
is_ready = False

def run_init():
    global orchestrator, agent, startup_status, error_message, is_ready
    try:
        model_name = "Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf"
        if not os.path.exists(model_name):
            model_name = "gemma-4-26B-A4B-it-uncensored-Q8_0.gguf"
        if not os.path.exists(model_name):
            model_name = "gemma-4-E4B-it-Q4_0.gguf"
        if not os.path.exists(model_name):
            print(f"[CHAT SERVER] Model weights '{model_name}' not found. Falling back to high-fidelity mock model.")
            model_name = "mock_only.gguf"
        
        print("[CHAT SERVER] Loading cognitive swarm components...")
        orchestrator = HenriCognitiveSwarmOrchestrator(
            model_path=model_name,
            num_streams=16
        )
        agent = ActiveInferenceSwarmAgent(orchestrator)
        with status_lock:
            startup_status = "Ready"
            is_ready = True
        print("[CHAT SERVER] Initialized successfully. Ready to handle chat requests.")
    except Exception as e:
        with status_lock:
            error_message = str(e)
            startup_status = f"Initialization Error: {error_message}"
        print(f"[CHAT SERVER] Critical Error during startup: {e}")

# Start initialization in thread
threading.Thread(target=run_init, daemon=True).start()

class ChatRequest(BaseModel):
    prompt: str

@app.get("/api/status")
def get_status():
    with status_lock:
        return {
            "status": startup_status,
            "ready": is_ready,
            "error": error_message,
            "model_mode": "Mock Engine" if (orchestrator is not None and orchestrator.is_mock) else "Gemma E4B RAM Swarm"
        }

@app.post("/api/chat")
def post_chat(req: ChatRequest):
    if not is_ready:
        raise HTTPException(status_code=503, detail="The cognitive swarm engine is still loading. Please wait.")
    
    try:
        # 1. Dynamically route and classify user intent
        routing = classify_prompt(req.prompt)
        
        # 2. Run the active inference loop
        response = agent.run_active_inference_loop(
            prompt=req.prompt,
            target_label=routing["domain"],
            max_revisions=routing["max_revisions"],
            path=routing["path"]
        )
        
        # Insert routing decision as the first log entry
        steps = [{
            "stage": "DYNAMIC ROUTING",
            "message": routing["reason"],
            "status": "success" if routing["path"] == "GENERAL" else "warning"
        }] + list(agent.step_logs)
        
        # Determine loop completion status
        outcome = "GENERAL_CONVERGED" if routing["path"] == "GENERAL" else "CONVERGED"
        
        # Hard Manifold Reset between chat requests to stabilize RAM and KV attention cache
        try:
            orchestrator.flush_cognitive_manifold()
        except Exception as flush_err:
            print(f"[CHAT SERVER] Warning: Failed to flush cognitive manifold: {flush_err}")
            
        return {
            "response": response,
            "revisions": routing["max_revisions"],
            "outcome": outcome,
            "steps": steps
        }
    except Exception as e:
        print(f"[CHAT ERROR] Exception during reasoning loop: {e}")
        raise HTTPException(status_code=500, detail=f"Error executing reasoning loop: {str(e)}")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
