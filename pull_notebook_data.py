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
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5433/henri")
HRR_DIM = 4096
NOTEBOOK_ID = "discrete-to-continuous-spatial"

# RAG static data compiled from successful manual queries (for offline seeder fallback)
STATIC_RAG_DATA = {
    "spatial_logic": """Discrete topological and structural invariants are holographically bound to continuous wave physics by mathematically defining attractors (constructive resonant states) and repellers (destructive ghost states) in the continuous domain.

1. Spatial Translation via the Fourier Shift Theorem
Moving a discrete object rigidly across a grid without altering its shape (rigid translation) is bound to continuous wave physics using the Fourier Shift Theorem.
- Continuous-Space Formulation: In the frequency domain, a rigid spatial translation (x0) maps strictly to a linear phase ramp:
  Ψ(x - x0) -> F{Ψ_hat(k)} e^(-i k x0)
  If the wave tries to translate outside the permitted grid domain, the phase ramp mathematically exceeds the boundary condition, the entanglement thread is severed, and the movement is vetoed.
- Discrete 2D Formulation: Translating a 2D signal ℓ[n,m] by a displacement of (n0,m0) pixels yields a circular shift property in the Fourier domain:
  F{ℓ[n - n0, m - m0]} = L[u,v] exp(-2π j (u n0 / N + v m0 / M))
  Where N×M is the image size.
- Translation Verification Metric: To mathematically estimate the exact displacement [n0, m0] on an N=M=128 grid, the real part of the ratio of the Discrete Fourier Transforms (DFTs) of the two translated images is evaluated as:
  cos(-2π j (u n0 / N + v m0 / M))

2. Symmetries via SU(N) Rotations
ARC-AGI grid transformations heavily rely on D4 (dihedral) symmetries—the rotations and reflections of shapes.
- Unitary Symmetry Attractor: A valid continuous spatial rotation maps to a specific unitary operator in the special unitary group SU(N). Under a Holographic Reduced Representation (HRR), this corresponds to a deterministic circular phase shift.
- Destructive Repeller: If a non-isometric transformation occurs (such as a rotation that distorts the shape’s internal proportions), the phase evolution of the wave falls out of sync with the orthogonal generator seeds. This phase mismatch triggers destructive interference at the polarizing Sagnac loop, driving the transmission coefficient to zero: T -> 0.

3. Bounding Box Invariants as Spatial Bandpass Filters
Strict bounding boxes that define an object's discrete footprint are translated into continuous wave physics as spatial bandpass filters.
- The Boundary Tensor Constraint: The boundary tensor (hcft) acting on the solid-state device (SSD) is configured to enforce a strict frequency cutoff representing the dimensions of the invariant bounding box.
- The Propagation Limit: If a wave vector attempts to "smear" the object outside its bounded topology, it generates high-frequency spatial components that exceed this cutoff. The physical material—such as a Barium Titanate (BaTiO3) crystal—simply refuses to propagate these invalid high frequencies, naturally annihilating the error.

4. Affine Constraints (Scaling) via Wavelength Dilation
Scaling a discrete shape to a different size is physically translated as dilating the continuous wave's length.
- Total Phase Birefringence Bounds: Scaling limits are mathematically etched into the Dirichlet Boundary of the waveguide as total phase birefringence bounds:
  B = BG + BS
- The Repelling Mechanism: If the spatial logic attempts an invalid scale change, the dilated wavelength shifts completely out of the physical material's transmission window (such as the glass constitution), causing the continuous wave to be entirely repelled and vetoed.

5. Supporting Mathematical Foundations (DFT & IDFT)
The discrete-to-continuous translation architecture is underpinned by the exact mathematical definitions of 2D wave equations and Fourier transformations:
- 2D Discrete Sine and Cosine Waves: Spatial frequencies u and v define how fast the waves vary along the spatial dimensions n and m:
  su,v[n,m] = A sin(2π (u n / N + v m / M))
  cu,v[n,m] = A cos(2π (u n / N + v m / M))
- 2D Discrete Fourier Transform (DFT): Converts a finite image ℓ[n,m] of size N×M into its spatial frequency representation L[u,v]:
  L[u,v] = sum_{n=0}^{N-1} sum_{m=0}^{M-1} ℓ[n,m] exp(-2π j (u n / N + v m / M))
- 2D Inverse Discrete Fourier Transform (IDFT): Reconstructs the original image by adding together the corresponding amplitudes of complex conjugate exponentials:
  ℓ[n,m] = 1/(NM) sum_{u=0}^{N-1} sum_{v=0}^{M-1} L[u,v] exp(+2π j (u n / N + v m / M))""",

    "thermodynamic_architecture": """Thermodynamic Software Architecture defines software engineering as a physical and mathematical struggle against computational entropy, where loosely typed systems naturally decay toward a state of maximum disorder (H_max). Unconstrained codebases suffer from spaghetti code, which is a physical manifestation of information-theoretic entropy.
To enforce architectural predictability, the conditional entropy of a system state space S governed by an interface boundary specification I—written as H(S|I)—must approach zero:
H(S|I) = - sum_i P(s_i | I) log_2 P(s_i | I) ≈ 0

I. The Abstract Syntax Tree (AST) Playbook
Rather than treating source code as unstructured strings, high-performance systems analyze and generate code by manipulating its Abstract Syntax Tree (AST).
1. The ESTree Architectural Philosophy:
- Backwards Compatible: Non-additive modifications to existing constructs are rejected to maintain ecosystem stability.
- Contextless: AST nodes must not retain any information about their parent (e.g., an expression node should not know if it is nested inside a loop).
- Unique: Information must never be duplicated in the tree.
- Extensible: New node types must be designed globally to allow future specification additions.
2. Performance-Critical AST Manipulation (The ts-morph Playbook):
- Work with Structures, Not Strings: Structures are simplified, lightweight representations of ASTs. Code-generating agents should build complete structures in memory before writing them to the compiler API, avoiding intermediate parsing cycles.
- Batch Operations: Group multiple modifications together. Bundling them into a single array write reduces compilation time by 80%.
- Analyze Then Manipulate: Always complete semantic analysis (using the compiler's type checker or symbol table) before performing any structural mutations.
3. Zero-Dependency, Zero-Allocation AST Parsers:
- Custom zero-dependency JSON and AST parsers designed specifically to process stream payloads from LLMs without memory leaks or execution drops.

II. Strict Interface Design Patterns
Unstructured data objects must never pass across internal system boundaries. Thermodynamic software architectures use strict, compiler-enforced interfaces to restrict the available degrees of freedom.
1. TypeScript Discriminated Unions:
- TypeScript AST interfaces prevent dynamic, sloppy states by using strict, immutable readonly properties and discriminated union types.
2. Python Structural Typing via typing.Protocol:
- Enforces compile-time duck-typing without runtime overhead using runtime checkable Protocols.
3. Rust Compile-Time Traits:
- Organizing code loops around generic traits rather than concrete types. Marker traits (like Send, Sync, and Future) eliminate runtime thread safety bugs at compile time.

III. Schema Invariant Playbooks
A schema is a mathematical boundary that defines the permissible states of a resource.
1. AWS CloudFormation Resource Provider Schemas (RPDS):
- No-Bypass Rule: Imperative programming constructs like if, then, else, and not are disallowed to ensure configuration determinism.
- Strict-Containment Rule: Every schema must define additionalProperties: false.
- Immutability Isolation Rule: Properties that can only be written during creation must be declared in createOnlyProperties.
2. Stripe's OpenAPI Dynamic Resource Expansions:
- x-expandableFields: Defines which object keys can be expanded at runtime.
- x-expansionResources: Points to the schema type that will replace the ID string upon expansion.
3. GraphQL Schema Introspection:
- __typename: Acts as a runtime type discriminator.
- __type and __schema: Introspection meta-fields reserved for structural discovery.

IV. The BMAD Method Playbook (Context Engineering)
Decoupling development into clean, progressive phases to eliminate planning inconsistency and context loss.
1. Strict Context Sharding: Requirements in prd.md (Phase 2), mapped to architectural constraints in architecture.md (Phase 3), and sharded into story-[slug].md tasks in Phase 4.
2. Two-Spine UX Contract (DESIGN.md and EXPERIENCE.md): DESIGN.md holds visual tokens; EXPERIENCE.md holds interactive behaviors. Downstream code resolves values via path-based token notation (e.g. {path.to.token}).

V. Human Heuristics & Swarm Vulnerabilities
1. Target Audience vs. Supervision Paradox: The human-in-the-loop must act as a strict system architect, writing specifications and type boundaries, while delegating only low-level, self-contained method implementations to the agents.
2. The Safeguard Gap in Developer Loops (The "Green-Build" Fallacy): Development agents must be programmatically forced to re-read files, analyze context, and run targeted tests, rather than working from high-level titles alone.
3. Parallel Workspace Git Clashes: Use distinct git worktrees for every concurrent agent session to prevent filesystem conflicts.
4. The Ablohian 3% Sparse Matrix Heurbation Heuristic: Limit code changes strictly to a highly localized 3% perturbation matrix (ΔP) over a 97% familiar codebase (P0):
   P_remix = P0 + ΔP subject to 0.02 <= ||ΔP||_F / ||P0||_F <= 0.04""",

    "logic_puzzles": """ARC-AGI grid transformations require extracting abstract, discrete spatial laws from minimal examples. In continuous optoelectronic systems, these discrete rules map directly onto continuous physical wave states through exact group and Fourier operator mappings:

1. Dihedral (D4) Symmetry Patterns (Rotations & Reflections)
- Input-Output Mapping: Shape rotated by multiples of 90 degrees or reflected.
- Wave Vector Transformation: Attractor maps to a unitary operator in SU(N), corresponding to circular phase shift in HRRs. Invalid states trigger phase mismatches, causing destructive interference (T -> 0) at the Sagnac loop.

2. Bounding Box Invariants
- Input-Output Mapping: Shape translated/manipulated while preserving its outer boundary.
- Wave Vector Transformation: Boundary tensor (hcft) enforces a frequency cutoff as a spatial bandpass filter. Out-of-bounds leakage generates high frequencies which are destroyed by the physical medium.

3. Spatial Translation
- Input-Output Mapping: Shape position shifts.
- Wave Vector Transformation: Linear phase ramp governed by the Fourier Shift Theorem. Boundary-violating translations trigger phase ramps that violate Dirichlet boundary conditions, vetoing the token.

4. Affine Constraints (Scaling)
- Input-Output Mapping: Shape resized proportionally.
- Wave Vector Transformation: Wavelength dilation. Invalid scaling causes the wavelength to drift outside the medium's transmission window, repelling the wave.

5. WRIGHT Constraint-Based Spatial Layouts
- Input: Layout canvas, design units.
- Output: Non-overlapping, aligned layout.
- Mathematical Constraints: Overlap prevention using Big-M formulation:
  x_j + w_j <= x_k + M(1 - z_jk)
  x_k + w_k <= x_j + M z_jk
  Adjacency: |x_j - (x_k + w_k)| <= ϵ subject to z_jk = 1

6. Conformal Welding & Shape Geodesics
- Welding: Möbius transformations and power scaling z_3 = z^α mapping unit circle pre-vertices to boundary curves:
  Ψ(θ) = 2 arctan(±|tan(θ/2)|^(1/2))
- Geodesics: Path of diffeomorphisms in Diff(S1)/PSL2(R) shape space cosets. Minimizes Weil-Petersson energy E_{0,1}:
  E_{0,1}(Ψ) = sum_{i,j,s} w_{i-j} f_{i,s}(Ψ) f_{j,s}(Ψ)

7. Continuous Scalar-Transport Gravitational Analogs
- Ray-like Bending: Wave packet traversal through a graded transport tensor medium. Evolves under:
  ∂_tt Φ - ∇·(R(x)∇Φ) + Λ(x)∂_t Φ = 0
- Anisotropy-Driven Drift: Transverse lateral drift via rotated transport tensor R(x) = Q^T diag(r1, r2) Q with R_xy != 0. Drift sign matches sign(R_xy)."""
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
