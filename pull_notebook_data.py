import subprocess
import json
import time
import os
import sys
import uuid
import glob
import re
import numpy as np

# Database imports
try:
    import psycopg
    HAS_PSYCOPG = True
except ImportError:
    HAS_PSYCOPG = False

# Default configurations
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432/henri")
HRR_DIM = 4096
NOTEBOOK_ID = "henri-mathematical-foundation-"

# RAG static data compiled from successful manual queries (for offline seeder fallback)
STATIC_RAG_DATA = {
    "spatial_logic": """Hyperdimensional Computing (HDC) & Holographic Reduced Representations (HRR) in the HENRI 7B Core:
1. Algebraic Binding (Circular Convolution)
Continuous variables are bound compositionally on the S^4095 hypersphere using circular convolution:
z = x ⊛ y
Using the Convolution Theorem, this is evaluated in the frequency domain as:
X_freq = FFT(x), Y_freq = FFT(y)
Z_freq = X_freq ⊙ Y_freq
z = IFFT(Z_freq)
z = z / ||z||_2 (Thermodynamic normalization to unit norm)

2. Algebraic Retrieval (Vector Involution)
Retrieving concepts from a bound state using the involution key:
involution_key = [key_0, key_-1, key_-2, ..., key_1]
x_retrieved = z ⊛ involution_key ≈ x

3. Olivera et al. (2026) Statistical Structural Regularizer R(X)
To prevent amplitude collapse across deep layers, we enforce:
R(X; τ_n, τ_v) = (sum_{i=1}^n ||x_i||^2 - τ_n)^2 + (mean(X))^2 + (Var(X) - τ_v)^2
- L_latent: Enforced on the latent wave transpose z^T (n=1, τ_n=m, τ_v=m/d) to match m bundled HRR pairs.
- L_value: Enforced on the denoised retrievals V_bar (n=m, τ_n=1, τ_v=1/d) to restore expected unit norm.

4. Information-Theoretic Capacity Bound
The mutual information I between input x and the quantized, m-slot symbolic latent z_hat is bounded by:
I(x; z_hat) <= min(m * (d / 2) * log(1 + 1/m), m * log(k))

5. Generalized Holographic Reduced Representations (GHRR)
GHRR extends traditional Fourier HRR to block-diagonal unitary matrices in the U(m) group:
H = [A^(1), ..., A^(D)]^T in C^(D x m x m), where A^(j) in U(m)
Non-Commutative Binding: H_1 * H_2 = [A^(j) * B^(j)]_{j=1}^D
Representational Similarity δ: δ(H_1, H_2) = (1 / mD) * Re(tr(sum_{j=1}^D A^(j) (B^(j))^†))
Holographic Attention: [σ(Q * K^†) * V]_j = σ(W_qj Λ_qj Λ_kj^† W_kj^†) W_vj * Λ_vj where σ(·) := softmax(Re[·]).""",

    "thermodynamic_architecture": """Basal Cognition & Thermodynamic Optimization:
1. Continuum Free Energy Loss F (Natural Induction)
The continuous loss is defined by the sum of internal propagation gradients and a Dirichlet boundary condition:
F(Ψ, W) = (1/2) * \int_{Ω} ||∇Ψ||^2 dV + (λ / 2) * \oint_{∂Ω} ||Ψ - A_ZoneC||^2 dS
In the tensor graph, this is discretized as:
- Internal Propagation Stress: internal_stress = (1 / 2BD) * sum_{b=1}^B sum_{d=1}^D ||Ψ_{b,d} - Ψ_{b,d-1}||_2^2
- Boundary Resonance Penalty: resonance = sum(Ψ_final · A_ZoneC), boundary_penalty = λ * mean(1.0 - resonance)
- Langevin Entropic Allowance: entropic_allowance = T * mean(||Ψ||^2)
Total Loss L_topo = internal_stress + boundary_penalty - entropic_allowance

2. Langevin State Update (The Divergent Master)
To bypass non-convex local energy basins:
dΨ / dt = -∇_{Ψ} F(Ψ, W) + sqrt(2T) * η(t)
where T is the active thermodynamic temperature and η(t) is standard Gaussian noise.

3. Second-Order Viscoelastic Creep Weight Update
The orthogonal weight matrices (W) evolve according to viscoelastic material creep:
∂W / ∂t = -μ * ∇_{W} F(Ψ, W)
where μ represents the plasticity coefficient of the fluid experts.

4. Continuous Phase-Space Routing (Fluid MoE)
Geometric resonance (Cosine Similarity R_i) against the learned phase attractors a_i:
R_i = <x, a_i> / (||x|| * ||a_i||)
Superposition collapsed according to thermodynamic weight:
w_i = exp(β * R_i) / sum_j exp(β * R_j)""",

    "logic_puzzles": """Wave-Geometric Duality & Boundary Conformal Field Theory (BCFT):
1. Free Boson Action on a Strip
S = (1 / 4π) * \int dσ dτ [(∂_σ X)^2 + (∂_τ X)^2]
Boundary conditions:
- Neumann: ∂_σ X = 0 at σ = 0, π
- Dirichlet: δX = 0 at σ = 0, π
Laurent modes gluing conditions:
- Neumann: j_n - \bar{j}_{-n} = 0
- Dirichlet: j_n + \bar{j}_{-n} = 0
Closed sector gluing:
- Neumann: (j_n + \bar{j}_{-n}) |B_N> = 0, solution |B_N> = (1 / \sqrt{2}) * exp(-sum_{k=1}^∞ (1/k) j_-k \bar{j}_-k) |0>
- Dirichlet: (j_n - \bar{j}_{-n}) |B_D> = 0, solution |B_D> = exp(sum_{k=1}^∞ (1/k) j_-k \bar{j}_-k) |0>

2. Conformal Ward Identities and Cardy Doubling Trick
Boundary condition: T(z) = \bar{T}(\bar{z}) for z = \bar{z}.
Doubling Trick: Maps the boundary problem on H^+ to a chiral problem on the full plane C:
<T(ζ) ϕ_1(z_1)...ϕ_m(z_m) \bar{ϕ}_1(z_1^*)...\bar{ϕ}_m(z_m^*)>_C

3. Ishibashi States, Cardy States, and Cardy Conditions
Conformal gluing: (L_n - \bar{L}_{-n}) |B> = 0. Ishibashi State: |h>> = sum_{N=0}^∞ sum_{j=1}^{d_h(N)} |h,N;j> ⊗ |h,N;j>.
Cardy State: |a> = sum_h (S_0h / \sqrt{S_ah}) |h>> where S is the modular S-matrix.
Cardy Conditions: n_{ab}^h = sum_{h'} (S_h^{h'} S_a^{h'} S_b^{h'} / S_0^{h'}) in Z_0^+.

4. Loop-Channel vs. Tree-Channel Equivalence
Open Sector: Z_{ab}^C(t) = Tr_{H_B}(q^{L_0 - c/24}) with q = e^{-2πt}.
Closed Sector: \tilde{Z}_{ab}^C(l) = <Θa| e^{-2πl(L_0 + \bar{L}_0 - 2c/24)} |b>.
Equivalence establishes t = 1 / 2l and normalizes the Neumann state to N_N = 2^(1/4).

5. Holographic AdS/BCFT Duality
Israel junction equation for ETW brane x(z): sqrt(1 + x'(z)^2) / x'(z) = R * σ.
Static spatial trajectory: x(z) = x_b - (sqrt(1 - R^2 σ^2) / Rσ) * z.
One-point function: <O(t,w,x_1)> = (2 |x_1 - x_b|)^-Δ * A^Δ.
Boundary entropy: g = (c/12)*log(2) + (c/24)*log((1 - Rσ) / (1 + Rσ)).

6. Disordered Nishimori Boundary Condition
Boundary spin-orientation tensor: B_σ^θ = cos(π/4 - θ/2) δ_{σ,+1} + sin(π/4 - θ/2) δ_{σ,-1}.
Renormalization group flow drives from unstable free boundary (θ=0) to stable Dirichlet boundary (θ=±π/2).

7. Field-Theoretic Derivation of the Constructal Law (Miguel, 2026)
Dimensionless Sagnac-stressed objective tensor rate:
K_*^o = Π_1 J_* ⊗ J_* - K_* + Π_2 ∇_*^2 K_* + Π_3 ξ_*
where Π_1 is Morphogenic Number, Π_2 is Structural Diffusion Number.
Lyapunov functional: F[K] = \int_{Ω} [-α * Tr(J ⊗ J · K) + (β/2) * Tr(K^2) + (γ/2) * |∇K|^2] d^n x."""
}

def complex_to_db(psi, hrr_dim: int = 4096) -> str:
    """Encode a complex wave of length hrr_dim into a pgvector literal of length 2*hrr_dim."""
    arr = np.asarray(psi)
    if arr.dtype == object:
        arr = arr.astype(np.complex128)
    if not np.iscomplexobj(arr):
        arr = arr.astype(np.complex128)
    arr = arr.reshape(-1)

    db_vec = np.empty(2 * hrr_dim, dtype=np.float32)
    db_vec[:hrr_dim] = arr.real.astype(np.float32)
    db_vec[hrr_dim:] = arr.imag.astype(np.float32)
    return "[" + ",".join(map(str, db_vec.tolist())) + "]"

def find_mcp_script():
    """Find the path to the index.js file of notebooklm-mcp on Windows."""
    appdata = os.environ.get("APPDATA")
    localappdata = os.environ.get("LOCALAPPDATA")
    
    candidates = []
    if localappdata:
        search_path = os.path.join(localappdata, "npm-cache", "_npx", "**", "notebooklm-mcp", "dist", "index.js")
        candidates.extend(glob.glob(search_path, recursive=True))
    
    if appdata:
        search_path = os.path.join(appdata, "npm", "node_modules", "mcp-server-notebooklm", "dist", "index.js")
        candidates.extend(glob.glob(search_path, recursive=True))
        
    for c in candidates:
        if os.path.exists(c):
            return c
            
    direct_fallback = r"C:\Users\chan\AppData\Local\npm-cache\_npx\0d29dd9f4e472da9\node_modules\notebooklm-mcp\dist\index.js"
    if os.path.exists(direct_fallback):
        return direct_fallback
        
    return None

def send_rpc(proc, method, params=None, msg_id=None):
    """Send a JSON-RPC message to the MCP server."""
    payload = {
        "jsonrpc": "2.0",
        "method": method
    }
    if params is not None:
        payload["params"] = params
    if msg_id is not None:
        payload["id"] = msg_id
    
    msg = json.dumps(payload) + "\n"
    proc.stdin.write(msg)
    proc.stdin.flush()

def read_rpc(proc, req_id=None):
    """Read JSON-RPC messages from the MCP server, filtering out log notifications."""
    while True:
        line = proc.stdout.readline()
        if not line:
            return None
        try:
            msg = json.loads(line)
            if "id" not in msg:
                if msg.get("method") == "notifications/logMessage":
                    params = msg.get("params", {})
                    print(f"[MCP LOG] ({params.get('level', 'info')}) {params.get('message')}", file=sys.stderr)
                continue
            
            if req_id is None or msg.get("id") == req_id:
                return msg
        except Exception as e:
            print(f"[PARSE ERROR] Failed parsing: {line} - {e}", file=sys.stderr)

def query_notebook(proc, question, msg_id):
    """Execute a query against the NotebookLM server using the ask_question tool."""
    print(f"\n[QUERYING] {question[:80]}...")
    send_rpc(proc, "tools/call", {
        "name": "ask_question",
        "arguments": {
            "notebook_id": NOTEBOOK_ID,
            "question": question
        }
    }, msg_id=msg_id)
    
    resp = read_rpc(proc, msg_id)
    if resp and "result" in resp:
        content = resp["result"].get("content", [])
        for c in content:
            if c.get("type") == "text":
                return c.get("text")
    return None

def parse_local_heuristics(filepath):
    """Parse Structural Pairings (heuristics) from the local markdown paper."""
    heuristics = {}
    if not os.path.exists(filepath):
        print(f"[-] Local heuristics file not found: {filepath}")
        return heuristics
        
    print(f"[*] Reading local heuristics from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    parts = re.split(r'\*\*Structural Pairing (\d+):\s*([^\*]+)\*\*', content)
    for i in range(1, len(parts), 3):
        if i+2 >= len(parts):
            break
        num = parts[i]
        name = parts[i+1].strip()
        body = parts[i+2]
        
        body_clean = re.split(r'\n-{3,}', body)[0]
        body_clean = re.split(r'\n\*\*Chapter', body_clean)[0]
        body_clean = body_clean.strip()
        
        heuristic_match = re.search(r'\*\s*\*\*THE HEURISTIC\*\*:\s*([^\n]+)', body_clean)
        tensor_match = re.search(r'\*\s*\*\*THE SPATIAL TENSOR PROFILE\*\*:\s*(.*)', body_clean, re.DOTALL)
        
        heuristic_text = heuristic_match.group(1).strip() if heuristic_match else ""
        tensor_text = tensor_match.group(1).strip() if tensor_match else ""
        
        full_text = f"Heuristic: {heuristic_text}\nSpatial Tensor Profile: {tensor_text}" if heuristic_text else body_clean
        
        concept_name = name.replace(" ", "_").replace("'", "").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_")
        concept_name = re.sub(r'_+', '_', concept_name).strip("_")
        
        heuristics[concept_name] = {
            "domain_tag": "human_heuristic",
            "raw_text": f"Structural Pairing {num} ({name}):\n{full_text}"
        }
        print(f"[+] Parsed local heuristic: {concept_name} (Pairing {num})")
        
    return heuristics

def parse_curriculum_map(filepath):
    """Parse concepts and syllabus sections from the Training Token Curriculum Map."""
    concepts = {}
    if not os.path.exists(filepath):
        print(f"[-] Curriculum map file not found: {filepath}")
        return concepts
        
    print(f"[*] Reading curriculum map from {filepath}...")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split content by sub-headers like ##### or ####
    parts = re.split(r'\n(#####?|####?)\s*([^\n]+)', content)
    
    current_module = "General"
    for i in range(1, len(parts), 3):
        if i+2 >= len(parts):
            break
        h_type = parts[i]
        h_title = parts[i+1].strip()
        body = parts[i+2]
        
        # If it's a Module header, track it
        if "Module" in h_title or "Stage" in h_title:
            current_module = h_title
            continue
            
        body_clean = re.split(r'\n(####|#####)', body)[0].strip()
        
        # Format concept name for database semantic_label
        concept_name = h_title.replace(" ", "_").replace("'", "").replace("(", "").replace(")", "").replace("/", "_").replace("-", "_").replace(".", "_")
        concept_name = re.sub(r'_+', '_', concept_name).strip("_")
        
        if len(concept_name) > 3:
            concepts[concept_name] = {
                "domain_tag": "curriculum_map",
                "raw_text": f"Curriculum Module ({current_module}) - {h_title}:\n{body_clean}"
            }
            print(f"[+] Parsed curriculum concept: {concept_name}")
            
    return concepts

def run_live_extraction():
    """Launch the MCP server and run queries to NotebookLM live."""
    mcp_script = find_mcp_script()
    if mcp_script:
        print(f"[+] Found local notebooklm-mcp script: {mcp_script}")
        cmd = ["node", mcp_script]
    else:
        print("[!] Local script not found, falling back to global 'npx notebooklm-mcp'")
        cmd = ["npx", "notebooklm-mcp"]
        
    print(f"[*] Starting MCP server process: {cmd}")
    env = os.environ.copy()
    env["NOTEBOOK_PROFILE_STRATEGY"] = "isolated"
    env["NOTEBOOK_CLONE_PROFILE"] = "true"
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            bufsize=1,
            env=env
        )
    except Exception as e:
        print(f"[-] Failed to start MCP process: {e}")
        return None
        
    print("[*] Executing Model Context Protocol initialization...")
    send_rpc(proc, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "HenriDatasetPuller", "version": "1.0"}
    }, msg_id=1)
    
    init_resp = read_rpc(proc, 1)
    if not init_resp:
        print("[-] Handshake failed. Exiting.")
        proc.terminate()
        return None
    print("[+] Handshake initialization completed.")
    
    send_rpc(proc, "notifications/initialized")
    
    queries = {
        "spatial_logic": (
            "What are the specific mathematical formulations, equations, and continuous-space "
            "translation/rotation matrices for mapping discrete topological rules (symmetry, bounding boxes, "
            "translations, affine constraints) to continuous-space wave mechanics (SU(N) rotations, bandpass "
            "filters, Fourier shift theorem, phase birefringence bounds)? Output all equations and details fully."
        ),
        "thermodynamic_architecture": (
            "What are the specific coding training principles, structural coding playbooks, and human heuristics "
            "for Thermodynamic Software Architecture? Focus on Abstract Syntax Trees (ASTs), schema invariants, "
            "strict interfaces (Rust traits, TypeScript interfaces, Python Protocols), and the BMAD Method. "
            "Explain all rules and provide examples."
        ),
        "logic_puzzles": (
            "Please list all specific spatial logic puzzles, ARC-AGI examples, and concept matrices that "
            "should be used as training data for the 7 billion parameter model. Provide their inputs, outputs, "
            "and corresponding wave vector transformations."
        )
    }
    
    dataset = {}
    msg_id = 2
    for key, question in queries.items():
        ans = query_notebook(proc, question, msg_id)
        if ans:
            if "launchPersistentContext" in ans or '"success": false' in ans:
                print(f"[-] Received browser launch error from RAG for {key}. Skipping response.")
            else:
                dataset[key] = {
                    "question": question,
                    "answer": ans
                }
                print(f"[+] Successfully extracted {key} ({len(ans)} chars)")
        else:
            print(f"[-] Failed to query {key}")
        msg_id += 1
        
    print("\n[*] Terminating MCP process...")
    proc.terminate()
    return dataset

def run_extraction():
    print("="*60)
    print("      HENRI 7B Swarm Dataset Generator & Seeder Script      ")
    print("="*60)
    
    # 1. Parse local files (heuristics and curriculum map)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    heuristics_file = os.path.join(script_dir, "archive", "The Quantitative Matrix of Aesthetic Experience_ Tensor Formulations of Architectural, Industrial, and Cognitive Space.md")
    curriculum_file = os.path.join(script_dir, "archive", "Training Token Curriculum Map for HENRI's 7B Swarm Agent.md")
    
    local_heuristics = parse_local_heuristics(heuristics_file)
    local_curriculum = parse_curriculum_map(curriculum_file)
    
    # 2. Get RAG results (check if --live flag is passed)
    if "--live" in sys.argv:
        print("[*] Running live extraction from NotebookLM via MCP...")
        rag_dataset = run_live_extraction()
        if not rag_dataset:
            print("[!] Live extraction failed or timed out. Falling back to static verified RAG data.")
            rag_dataset = {}
            for key, val in STATIC_RAG_DATA.items():
                rag_dataset[key] = {"question": "Static query", "answer": val}
    else:
        print("[*] Using pre-compiled verified RAG data (run with --live for active crawling)...")
        rag_dataset = {}
        for key, val in STATIC_RAG_DATA.items():
            rag_dataset[key] = {"question": "Static query", "answer": val}
            
    # 3. Combine everything and serialize to local JSON archive
    archive_dir = os.path.join(script_dir, "archive")
    os.makedirs(archive_dir, exist_ok=True)
    archive_path = os.path.join(archive_dir, "pulled_notebook_data.json")
    
    combined_data = {
        "remote_notebook_rag": rag_dataset,
        "local_human_heuristics": local_heuristics,
        "local_curriculum_map": local_curriculum
    }
    
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, indent=2)
    print(f"[+] Saved complete training dataset to {archive_path}")
    
    # 4. Ingest into database
    concepts_to_seed = {}
    
    # Add RAG findings
    for key, val in rag_dataset.items():
        ans = val.get("answer", "")
        if ans:
            concepts_to_seed[f"RAG_{key.upper()}"] = ("notebook_rag", ans)
            
    # Add individual canonical RAG concepts
    spatial_ans = rag_dataset.get("spatial_logic", {}).get("answer", "")
    thermo_ans = rag_dataset.get("thermodynamic_architecture", {}).get("answer", "")
    puzzles_ans = rag_dataset.get("logic_puzzles", {}).get("answer", "")
    
    if spatial_ans:
        concepts_to_seed["Fourier_Shift_Theorem"] = ("spatial_logic", "Fourier Shift Theorem mapping discrete spatial translations to linear phase ramps.")
        concepts_to_seed["SU_N_Rotations"] = ("spatial_logic", "Unitary symmetry rotations mapping to circular phase shifts under HRR.")
        concepts_to_seed["Bounding_Box_Invariants"] = ("spatial_logic", "Bounding boxes mapped to frequency cutoffs in spatial bandpass filters.")
        concepts_to_seed["Wavelength_Dilation"] = ("spatial_logic", "Scaling of shape topologies mapped to dilation of continuous wavelengths.")
        concepts_to_seed["Sagnac_Interference"] = ("spatial_logic", "Destructive interference vetoing non-isometric transformations.")
        
    if thermo_ans:
        concepts_to_seed["Thermodynamic_Software_Architecture"] = ("thermodynamic_architecture", "System state conditional entropy minimized to approach zero to guarantee architectural predictability.")
        concepts_to_seed["Schema_Invariants"] = ("thermodynamic_architecture", "Interface boundaries restricting state expansion to approach zero entropy.")
        concepts_to_seed["AST_Driven_Engineering"] = ("thermodynamic_architecture", "AST node validation rules forcing structural compliance.")
        concepts_to_seed["BMAD_Method"] = ("thermodynamic_architecture", "Phase-decoupled Agile Development workflow isolating planning from implementation.")
        
    if puzzles_ans:
        concepts_to_seed["ARC_Spatial_Puzzles"] = ("logic_puzzles", puzzles_ans)
        
    # Add local heuristics
    for name, data in local_heuristics.items():
        concepts_to_seed[name] = (data["domain_tag"], data["raw_text"])
        
    # Add curriculum map concepts
    for name, data in local_curriculum.items():
        concepts_to_seed[name] = (data["domain_tag"], data["raw_text"])
        
    print(f"\n[*] Seeding hrr_canonical_lexicon in TimescaleDB (Total concepts to seed: {len(concepts_to_seed)})...")
    db_connected = False
    if HAS_PSYCOPG:
        try:
            with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
                db_connected = True
        except Exception as e:
            print(f"[!] Warning: TimescaleDB unreachable at {DATABASE_URL} ({e}). Entering offline mode.")
    else:
        print("[!] psycopg package not installed. Entering offline mode.")
        
    if db_connected:
        try:
            seeded_count = 0
            with psycopg.connect(DATABASE_URL) as conn:
                with conn.cursor() as cur:
                    conn.autocommit = True
                    
                    # Clean up old error entries if any
                    cur.execute("""
                        DELETE FROM hrr_canonical_lexicon 
                        WHERE raw_text LIKE '%launchPersistentContext%' 
                           OR raw_text LIKE '%"success": false%';
                    """)
                    deleted = cur.rowcount
                    if deleted > 0:
                        print(f"[*] Pruned {deleted} invalid error entries from hrr_canonical_lexicon.")
                    
                    for label, (domain_tag, raw_text) in concepts_to_seed.items():
                        # Generate random unit-magnitude complex vector representing concept
                        phases = (np.random.rand(HRR_DIM) * 2 * np.pi) - np.pi
                        vec = np.exp(1j * phases)
                        vector_str = complex_to_db(vec, HRR_DIM)
                        
                        # Generate deterministic UUID using concept label
                        concept_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, label))
                        
                        cur.execute("""
                            INSERT INTO hrr_canonical_lexicon (concept_hash, semantic_label, domain_tag, hrr_wavefront, raw_text)
                            VALUES (%s, %s, %s, %s::vector, %s)
                            ON CONFLICT (concept_hash) DO UPDATE 
                            SET raw_text = EXCLUDED.raw_text, hrr_wavefront = EXCLUDED.hrr_wavefront, last_verified = NOW();
                        """, (concept_hash, label[:128], domain_tag, vector_str, raw_text))
                        seeded_count += cur.rowcount
            print(f"[SUCCESS] Seeded {seeded_count} canonical concepts into hrr_canonical_lexicon.")
        except Exception as db_err:
            print(f"[-] Database error during seeding: {db_err}")
    else:
        print("[Offline Mode] Skipping TimescaleDB insertion. Generated vectors locally.")

if __name__ == "__main__":
    run_extraction()
