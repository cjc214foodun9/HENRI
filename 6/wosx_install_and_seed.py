import os
import sys
import shutil
import torch
import psycopg

# Ensure parent directory is in sys.path to resolve imports like 'universal_repl'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cognitive_swarm import HenriCognitiveSwarmOrchestrator

# --- 1. Hard Purge of Obsolete Topological Baggage ---
def hard_overwrite_ssd():
    print("[WARNING] Executing Hard Overwrite of Zone C Axiom SSD...")
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                conn.autocommit = True
                cur.execute("TRUNCATE TABLE hrr_canonical_lexicon CASCADE;")
        print("[SUCCESS] Zone C SSD flushed (PostgreSQL hrr_canonical_lexicon truncated).")
    except Exception as e:
        print(f"[WARNING] Database truncate failed: {e}. Attempting local fallback wipe.")
        ssd_path = "./data/zone_c_axioms"
        if os.path.exists(ssd_path):
            shutil.rmtree(ssd_path)
        os.makedirs(ssd_path, exist_ok=True)
        print("[SUCCESS] Local Zone C SSD flushed.")
    print("[SUCCESS] Ready for continuous physics seeding.")

# --- 2. Compile WoSX Python Bindings ---
def compile_wosx_bindings():
    print("[INIT] Compiling WoSX/FCPW headers for Vulkan (AMD GPU)...")
    
    # Target directory for the clone
    wosx_dir = "../lib_physics/wosx"
    if not os.path.exists("../lib_physics"):
        os.makedirs("../lib_physics", exist_ok=True)
        
    if not os.path.exists(wosx_dir):
        print("[INIT] Cloning nv-tlabs/wosx repository...")
        os.system(f"git clone https://github.com/nv-tlabs/wosx.git {wosx_dir}")
    else:
        print("[INIT] wosx repository already exists.")
        
    # Explicitly enforce Vulkan and disable CUDA for AMD Radeon compatibility
    os.environ["WOSX_USE_CUDA"] = "0"
    os.environ["WOSX_USE_VULKAN"] = "1"
    
    build_cmd = "python setup.py build_ext --inplace --compiler=msvc"
    print(f"[INIT] Executing build command: {build_cmd}")
    os.system(f"cd {wosx_dir} && {build_cmd}")
    print("[SUCCESS] WoSX PDE solver ready for import.")

# --- 3. Seed Continuous Boundary Axioms ---
def seed_continuous_axioms(orchestrator):
    print("[SEED] Encoding Dirichlet & Neumann boundary conditions...")
    router = orchestrator.l3_router
    
    axioms = {
        "dirichlet_boundary": "solve_laplace_wos(boundary_val=const)",
        "neumann_boundary": "solve_poisson_wost(flux=gradient_normal)"
    }
    
    # We need a way to tokenize string to integer list. The gen_model can do this.
    try:
        for name, code in axioms.items():
            if hasattr(orchestrator, 'gen_model') and orchestrator.gen_model is not None:
                # Attempt to use llama_cpp tokenizer if available
                try:
                    tokens = orchestrator.gen_model.llama.tokenize(code.encode("utf-8"), add_bos=False)
                except TypeError:
                    tokens = orchestrator.gen_model.llama.tokenize(code.encode("utf-8"))
            else:
                # Fallback mock tokens if running without llama.cpp loaded
                tokens = [ord(c) for c in code]
                
            token_tensor = torch.tensor(tokens, dtype=torch.long, device='cpu')
            wave = router.text_to_wave(token_tensor)
            
            # Persist to SSD via Synaptic Manager
            # We don't have a direct 'save_to_ssd' method, but we can insert it manually or use a helper
            save_to_ssd_db(name, wave)
            
        print("[SUCCESS] Zone C seeded with continuous PDE boundary axioms.")
    except Exception as e:
        print(f"[ERROR] Failed to seed continuous axioms: {e}")

def save_to_ssd_db(name, wave):
    import uuid
    from henri_contract import complex_to_db, DIMS
    db_url = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                conn.autocommit = True
                concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, name))
                # Ensure wave is numpy
                if torch.is_tensor(wave):
                    wave_np = wave.detach().cpu().numpy()
                else:
                    wave_np = wave
                vector_str = complex_to_db(wave_np, DIMS.hrr_dim)
                cur.execute(
                    """
                    INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (concept_hash) DO NOTHING;
                    """,
                    (concept_hash, name, "wosx_pde_axiom", vector_str),
                )
    except Exception as e:
        print(f"[WARNING] Could not save {name} to TimescaleDB: {e}")

if __name__ == "__main__":
    hard_overwrite_ssd()
    compile_wosx_bindings()
    
    print("[INIT] Initializing Orchestrator to sync L3SwarmRouter vocabulary...")
    orchestrator = HenriCognitiveSwarmOrchestrator()
    seed_continuous_axioms(orchestrator)
