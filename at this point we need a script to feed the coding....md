To unblock the system and bridge the gap between your hardware layer and the multi-agent execution loop, you can feed this comprehensive master integration blueprint directly into your coding assistant.  
This script explicitly corrects the silent fallback to mock mode, implements the AST block isolation gate, protects the complex wave phase coordinates, and enforces the non-accumulative Stanford ACE playbook routing loop.

### **📋 AI Coding Assistant Prompt & Source Patch Blueprint**

Plaintext  
Purpose: Unify the fragmented HENRI architecture modules.  
Target Files to Patch: cognitive\_swarm.py, neurosymbolic\_program\_induction.py, active\_experimentation\_engine.py  
Hardware Targets: AMD Radeon RX 9070 XT via Vulkan (llama\_cpp) and DirectML (PyTorch)  
Base Model: Huihui-gemma-4-12B-it-abliterated.Q8\_0.gguf (Latent Dimension: 3840\)

#### **Module 1: Core Model Loader & Swarm Fabric (cognitive\_swarm.py)**

**Assistant Instructions**: Patch the model initialization phase. Disable the silent try/except block that rolls back to a 2560-D mock model upon a C++ loading crash. Force the system to fail loudly if the binary compilation rejects Gemma 4's non-uniform layer structures. Implement in-memory logits steering and the ACE playbook-to-wave compiler.

Python  
\# \=====================================================================  
\# FILE: cognitive\_swarm.py  
\# \=====================================================================  
import os  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from llama\_cpp import Llama

\# Enforce hardware acceleration environment variables  
os.environ\["GGML\_DISABLE\_VULKAN"\] \= "0"  
os.environ\["GGML\_VULKAN\_DISABLE"\] \= "0"

class HenriCognitiveSwarmOrchestrator:  
    def \_\_init\_\_(self, model\_path, num\_experts=16):  
        print("\[SYSTEM\] Booting high-rigor Vulkan-accelerated Swarm Orchestrator...")  
          
        self.gemma\_dim \= 3840      \# Hard invariant for Gemma 4 12B  
        self.hopfield\_dim \= 4096   \# Continuous wave-space manifold dimension  
        self.num\_experts \= num\_experts  
          
        \# CRITICAL FIX: Load the true GGUF model. DO NOT allow a silent fallback to a mock model  
        try:  
            self.base\_model \= Llama(  
                model\_path=model\_path,  
                n\_ctx=16384,        \# Double sequence ceiling for parallel streams  
                n\_batch=512,  
                n\_gpu\_layers=35,    \# Offload \~75% of layers to Radeon VRAM  
                verbose=True  
            )  
            print(f"\[SUCCESS\] Base model weights mapped directly to Vulkan VRAM.")  
        except Exception as e:  
            \# Force thread termination so weights on disk aren't overwritten by shape mismatches  
            print(f"\[FATAL FAILURE\] llama.cpp failed to allocate tensors: {e}")  
            print("\[HELP\] Upgrade llama-cpp-python to a version supporting Gemma 4's non-uniform GQA arrays.")  
            raise SystemExit(1)

        \# Initialize the lightweight CPU instance for zero-VRAM Curation/Reflection via mmap  
        self.reflector\_model \= Llama(  
            model\_path=model\_path,  
            n\_ctx=8192,  
            n\_gpu\_layers=0,        \# Pinned entirely to CPU to protect GPU compute lanes  
            use\_mmap=True,         \# Shared page cache with base\_model to save RAM  
            n\_threads=8            \# Prevent thread starvation across background loops  
        )

        \# Initialize the rigid L3 Router Space  
        from l3\_router\_model import L3SwarmRouter  
        self.l3\_router \= L3SwarmRouter(  
            num\_experts=self.num\_experts,   
            hopfield\_dim=self.hopfield\_dim,   
            gemma\_dim=self.gemma\_dim  
        ).to(torch.device('cpu')) \# Keep routing overhead coupled to system RAM

    @torch.no\_grad()  
    def compile\_playbook\_to\_wave(self, playbook\_sections: dict) \-\> torch.Tensor:  
        """  
        ACE Neurosymbolic Compilation: Compiles textual rules into a 4096-D wave.  
        """  
        rule\_waves \= \[\]  
        w\_down\_matrix \= self.l3\_router.w\_down.weight \# Shape: \[3840, 4096\]  
          
        for section, guideline in playbook\_sections.items():  
            rule\_text \= f"{section}: {guideline}"  
              
            \# Embed rule via CPU-mmap instance  
            emb\_response \= self.reflector\_model.create\_embedding(rule\_text)  
            g\_rule \= torch.tensor(emb\_response\["data"\]\[0\]\["embedding"\], dtype=torch.float32, device='cpu')  
              
            \# Inverse orthogonal projection: Spin 3840-D back up to 4096-D continuous wave topology  
            psi\_rule \= torch.matmul(g\_rule, w\_down\_matrix)  
            rule\_waves.append(psi\_rule)  
              
        if not rule\_waves:  
            return torch.zeros(self.hopfield\_dim, device='cpu')  
              
        \# Superposition: Merge all active guidelines into a single interference pattern  
        compiled\_wave \= torch.sum(torch.stack(rule\_waves), dim=0)  
        return F.normalize(compiled\_wave, p=2, dim=-1)

#### **Module 2: Sanity Isolation & Program Induction (neurosymbolic\_program\_induction.py)**

**Assistant Instructions**: Re-engineer the symbolic validation gate to resolve the invalid decimal literal crash. Use regex to strip away conversational introductory prose and extract raw, parseable blocks before passing variables to the Python AST compiler.

Python  
\# \=====================================================================  
\# FILE: neurosymbolic\_program\_induction.py  
\# \=====================================================================  
import ast  
import re

class ProgramInductor:  
    def \_\_init\_\_(self):  
        print("\[SYSTEM\] Initializing Neurosymbolic AST Program Inductor...")

    def verify\_syntax(self, raw\_hypothesis: str):  
        """  
        Isolates executable syntax from natural language text to prevent compiler errors.  
        """  
        \# Search for explicit code generation block delimiters  
        code\_match \= re.search(r"\<\\|python\_begin\\|\>(.\*?)\<\\|python\_end\\|\>", raw\_hypothesis, re.DOTALL)  
        if not code\_match:  
            code\_match \= re.search(r"\`\`\`python(.\*?)\`\`\`", raw\_hypothesis, re.DOTALL)  
              
        if not code\_match:  
            \# If no tags are found, attempt to clean up raw text block  
            lines \= raw\_hypothesis.split('\\n')  
            sanitized\_lines \= \[line for line in lines if not re.match(r"^\\s\*(\\d+\\.|\\-|Wait|Let's|\\\*)", line)\]  
            pure\_code \= "\\n".join(sanitized\_lines).strip()  
        else:  
            pure\_code \= code\_match.group(1).strip()

        if not pure\_code or "def transform" not in pure\_code:  
            return False, "VETO: Generation failed to yield an isolated functional 'transform(input\_grid)' block."

        \# Execute Abstract Syntax Tree validation  
        try:  
            ast.parse(pure\_code)  
            return True, pure\_code  
        except SyntaxError as e:  
            \# Feed exact line numbers and syntax diagnostics back into the ACE Reflector loop  
            error\_feedback \= f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset} code: {e.text}"  
            return False, error\_feedback

    def assert\_generalization(self, pure\_code: str):  
        """  
        Inductive Generalization Guard: Scans the AST tree to block look-up tables and forcing overfitting paths.  
        """  
        tree \= ast.parse(pure\_code)  
        for node in ast.walk(tree):  
            \# Block hardcoded comparisons of literal multidimensional arrays  
            if isinstance(node, ast.Compare):  
                for comparator in node.comparators:  
                    if isinstance(comparator, ast.List):  
                        return False, "VETO: Explicit list/grid value matching detected. Look-up tables are forbidden."  
        return True, "Passed Inductive Generalization Verification."

#### **Module 3: Active Experimentation Engine & Test Harness (active\_experimentation\_engine.py)**

**Assistant Instructions**: Implement the main evolutionary MCTS trajectory loop. Pin the static ARC grid arrays to the top of the prompt space to safeguard the Vulkan KV-cache, and place the volatile, moving playbook at the bottom. Ensure complex calculations maintain phase dimensions.

Python  
\# \=====================================================================  
\# FILE: active\_experimentation\_engine.py  
\# \=====================================================================  
import time  
import torch  
from neurosymbolic\_program\_induction import ProgramInductor  
from zone\_b\_emulator import ZoneBPhysicalEmulator

class ActiveExperimentationEngine:  
    def \_\_init\_\_(self, orchestrator, time\_limit=1200):  
        self.orchestrator \= orchestrator  
        self.time\_limit \= time\_limit  
        self.inductor \= ProgramInductor()  
        self.emulator \= ZoneBPhysicalEmulator()  
        self.stagnation\_counter \= 0  
        self.last\_error \= None

    def execute\_task\_manifold(self, task\_dict: dict):  
        """  
        Drives evolutionary search over the parallel token manifolds.  
        """  
        start\_time \= time.time()  
          
        \# Initialize an empty Stanford ACE playbook dictionary structure  
        playbook\_dict \= {  
            "Syntax\_Rules": "Wrap code blocks within explicit markdown fences.",  
            "Topological\_Constraints": "Compute spatial coordinate differentials before moving values."  
        }  
          
        while time.time() \- start\_time \< self.time\_limit:  
            \# 1\. Map text playbook keys into continuous 4096-D intent patterns  
            playbook\_wave \= self.orchestrator.compile\_playbook\_to\_wave(playbook\_dict)  
              
            \# 2\. Extract weights and compute MoE routing weights via the frozen orthogonal layer  
            lora\_weights \= self.orchestrator.l3\_router.compute\_routing\_weights(  
                playbook\_wave.unsqueeze(0),   
                temperature=self.determine\_langevin\_heat()  
            )  
              
            \# 3\. Prompt Construction: Keep static task grids at top of context to save KV-Cache hits\!  
            task\_grid\_context \= f"ARC Task Data Structure:\\n{task\_dict\['train'\]}\\n"  
            serialized\_playbook \= f"Active Heuristics Playbook:\\n{self.serialize\_playbook(playbook\_dict)}"  
            clean\_prompt \= f"{task\_grid\_context}\\n{serialized\_playbook}\\nGenerate the transform function:\\n"  
              
            \# 4\. Generate parallel candidate streams across the in-memory LoRA layers  
            candidates \= self.orchestrator.generate\_swarm\_hypotheses(  
                prompt=clean\_prompt,   
                dynamic\_lora\_weights=lora\_weights  
            )  
              
            scored\_trajectories \= \[\]  
            for raw\_code in candidates:  
                \# Gate 1: Parse syntax and extract pure code blocks  
                syntax\_pass, code\_payload \= self.inductor.verify\_syntax(raw\_code)  
                if not syntax\_pass:  
                    scored\_trajectories.append((raw\_code, 0.0, code\_payload)) \# feedback is stored in code\_payload  
                    continue  
                  
                \# Gate 2: Stop lookup table cheating  
                gen\_pass, guard\_feedback \= self.inductor.assert\_generalization(code\_payload)  
                if not gen\_pass:  
                    scored\_trajectories.append((code\_payload, 0.05, guard\_feedback))  
                    continue  
                  
                \# Gate 3: REPL Sandbox verification against the training grids  
                score, sandbox\_feedback \= self.run\_repl\_sandbox(code\_payload, task\_dict\["train"\])  
                if score \== 1.0:  
                    \# Absolute inductive alignment reached. Resolve unseen test inputs.  
                    test\_output \= self.run\_repl\_sandbox(code\_payload, task\_dict\["test"\])  
                    self.orchestrator.l3\_router.update\_expert\_centroids(playbook\_wave) \# Anchor centroids  
                    return test\_output, "SUCCESS"  
                      
                scored\_trajectories.append((code\_payload, score, sandbox\_feedback))  
              
            \# 5\. Evolution: Sort candidates based on training set correctness  
            scored\_trajectories.sort(key=lambda x: x\[1\], reverse=True)  
            best\_code, best\_score, best\_feedback \= scored\_trajectories\[0\]  
              
            \# 6\. Apply Thermodynamic Shaking if stuck in a local logic lock  
            self.evaluate\_stagnation(best\_feedback)  
              
            \# 7\. Asynchronous Reflection and Playbook Curation via CPU-mmap node  
            reflection\_insight \= self.orchestrator.reflect\_on\_failure(clean\_prompt, best\_code, best\_feedback)  
            playbook\_dict \= self.orchestrator.curate\_playbook(playbook\_dict, reflection\_insight)  
              
        return None, "TIMEOUT\_FAILURE"

    def determine\_langevin\_heat(self):  
        if self.stagnation\_counter \>= 2:  
            print("\[THERMODYNAMICS\] Stagnation detected inside logic basin. Spiking temperature to 1.35...")  
            return 1.35  
        return 0.70

    def evaluate\_stagnation(self, current\_error):  
        if current\_error \== self.last\_error:  
            self.stagnation\_counter \+= 1  
        else:  
            self.stagnation\_counter \= 0  
            self.last\_error \= current\_error

    def serialize\_playbook(self, playbook):  
        return "\\n".join(\[f"- \[{k}\]: {v}" for k, v in playbook.items()\])

    def run\_repl\_sandbox(self, code, cases):  
        \# Implementation executes the code string against inputs in an isolated context  
        \# Returns (accuracy\_float, diagnostic\_string)  
        pass

#### **Module 4: High-Precision Vector Bending (dynamic\_lora.py Patch)**

**Assistant Instructions**: Update the weights update loop. Modify dynamic\_lora.py to preserve the imaginary phase coordinates when updating the weights with error energy, preventing the phase-erasure warning.

Python  
\# \[PATCH LINE 62 FOR dynamic\_lora.py\]  
\# CRITICAL FIX: Split complex vectors into independent real/imag features to keep phase geometry  
real\_channel \= delta\_projected.real  
imag\_channel \= delta\_projected.imag

\# Concatenate channels along a new feature axis rather than casting down to real-only float32  
delta\_tensor \= torch.cat(\[real\_channel, imag\_channel\], dim=-1).to(torch.float32)

### **🚀 Execution & Hot-Deployment Strategy**

To deploy this cleanly without creating temporary file locks on your Windows environment, execute the following workflow:

1. **Kill Active Workers**: Kill any lingering background run tasks using PowerShell:  
   PowerShell  
   Stop-Process \-Name python \-Force

2. **Inject Scripts**: Feed the snippets above to your assistant to rewrite the code.  
3. **Execute Benchmark Harness**: Boot the evaluation process using unbuffered logging output to track the ROUTING GATEWAY transitions in real time:  
   PowerShell  
   C:\\Python312\\python.exe \-u run\_arc\_benchmark.py  
