import code
import sys
import io
import traceback
import contextlib
import hashlib
import socket
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
# Also catch the specific SymPy warning
try:
    from sympy.utilities.exceptions import SymPyDeprecationWarning
    warnings.filterwarnings("ignore", category=SymPyDeprecationWarning)
except ImportError:
    pass

# Orig connection function reference to allow loopback connections while blocking external calls
_orig_socket_connect = socket.socket.connect

def _restricted_socket_connect(self, address):
    """
    Monkeypatch connector that enforces the Epistemic Air-Gap boundary conditions.
    Permits local loopback connections (localhost, 127.0.0.1, ::1) for Neo4j database,
    but blocks all external outbound IP traffic process-wide.
    """
    host = address[0]
    
    # 1. Allow loopback/localhost directly
    if host in ("127.0.0.1", "localhost", "::1"):
        return _orig_socket_connect(self, address)
    
    # 2. Check if host resolves to loopback
    is_loopback = False
    try:
        resolved_ip = socket.gethostbyname(host)
        if resolved_ip in ("127.0.0.1", "::1"):
            is_loopback = True
    except Exception:
        pass
        
    if is_loopback:
        return _orig_socket_connect(self, address)
        
    # 3. Block all other connections
    raise PermissionError(f"[AIR-GAP] Outgoing network connection blocked to external host: {host}")

class UniversalREPL:
    """
    Air-Gapped Universal Physics Sandbox (UniversalREPL).
    Orchestrates process-wide outbound network isolation, pre-loads the scientific
    computation physics stack, truncates massive symbolic/LaTeX tensor outputs,
    and detects SymPy equality contradictions to feed Sagnac Veto loops.
    """
    def __init__(self):
        # Enforce socket connection restrictions process-wide
        self.enforce_epistemic_air_gap()

        # Create persistent InteractiveConsole
        self.console = code.InteractiveConsole()
        
        # Pre-import physics stack
        self.pre_load_physics_stack()

    @property
    def sandbox_globals(self):
        """Getter for the REPL sandbox globals dictionary."""
        return self.console.locals

    @sandbox_globals.setter
    def sandbox_globals(self, value):
        """Setter for the REPL sandbox globals dictionary. Preloads scientific packages if empty."""
        self.console.locals = value
        if not value:
            self.pre_load_physics_stack()

    def enforce_epistemic_air_gap(self):
        """Monkeypatches socket connector to establish outbound network isolation."""
        socket.socket.connect = _restricted_socket_connect

    def prepare_physics_sandbox(self, problem_text: str, rigor_level: int = 1):
        """
        Injects advanced theoretical physics libraries into the REPL globals
        based on the semantic domain of the problem.
        """
        if rigor_level == 0:
            return

        sandbox_globals = self.console.locals
        problem_text_lower = problem_text.lower()
        hints = []
        
        # 0. The Ansatz & Lambdify Symbolic-to-Numeric Bridge Hint
        hints.append(
            "# [PhysLib Syntax Hint: Symbolic to Numeric Bridge]\n"
            "# 1. Define your metric and derive the Equations of Motion (EoM) using sympy.\n"
            "# 2. CRITICAL: You cannot pass sympy expressions directly to solve_ivp.\n"
            "# 3. Use `sp.lambdify` to convert your symbolic EoM into a numerical function.\n"
            "# Example:\n"
            "# H_dot_sym = ... # Your symbolic expression for dH/dt\n"
            "# H_dot_num = sp.lambdify((t, H, theta), H_dot_sym, 'numpy')\n"
            "# def ode_system(t, y):\n"
            "#     return [H_dot_num(t, y[0], y[1]), ...]\n"
            "# sol = scipy.integrate.solve_ivp(ode_system, ...)"
        )
        
        # 1. Quantum Information, Science & Technology (QIST)
        if any(kw in problem_text_lower for kw in ["qubit", "decoherence", "lindblad", "entanglement fidelity", "density matrix"]):
            try:
                sandbox_globals['qutip'] = __import__('qutip')
            except ImportError:
                pass
            try:
                sandbox_globals['pennylane'] = __import__('pennylane')
            except ImportError:
                pass
            hints.append("# [Hint: Use qutip.mesolve for open quantum systems]")
            hints.append("# [Hint: Use pennylane for quantum differentiable programming]")

        # 2. Atomic, Molecular & Optical (AMO)
        if any(kw in problem_text_lower for kw in ["rydberg", "hartree-fock", "dipole transition", "optical cavity", "rabi frequency"]):
            try:
                sandbox_globals['pyscf'] = __import__('pyscf')
            except ImportError:
                pass
            try:
                sandbox_globals['arc'] = __import__('arc')
            except ImportError:
                pass
            hints.append("# [Hint: Use pyscf for Hartree-Fock and molecular quantum chemistry simulations]")
            hints.append("# [Hint: Use arc for Alkali Rydberg interactions and calculators]")

        # 3. High Energy Physics (HEP)
        if any(kw in problem_text_lower for kw in ["feynman diagram", "loop integral", "scattering amplitude", "renormalization group"]):
            try:
                sandbox_globals['pySecDec'] = __import__('pySecDec')
            except ImportError:
                pass
            hints.append("# [Hint: Use pySecDec for numerical loop integration of divergent amplitudes]")

        # 4. Mathematical Physics
        if any(kw in problem_text_lower for kw in ["zeta", "hypergeometric", "asymptotic expansion", "arbitrary precision"]):
            try:
                sandbox_globals['mp'] = __import__('mpmath').mp
            except ImportError:
                pass
            hints.append("# [Hint: Use mpmath for arbitrary precision functions (zeta, hypergeometric series)]")

        # 5. Gravitation & Cosmology
        cosmo_kws = ["metric", "e-folds", "geodesic", "cosmology", "frw", "hubble", "expansion", "kerr", "redshift", "friedmann", "manifold", "weyl", "holographic", "trace", "ricci", "christoffel", "riemann", "anomaly"]
        if any(kw in problem_text_lower for kw in cosmo_kws):
            try:
                sandbox_globals['einsteinpy'] = __import__('einsteinpy')
            except ImportError:
                pass
            try:
                sandbox_globals['astropy_cosmo'] = __import__('astropy.cosmology')
            except ImportError:
                pass
            try:
                sandbox_globals['SYMPY_TENSOR_HINT'] = (
                    "# [PhysLib Syntax Hint: General Relativity & Topology]\n"
                    "# 1. `sympy.diffgeom` is DEPRECATED. Do not use it.\n"
                    "# 2. Use standard SymPy matrices or `sympy.tensor` to define metrics.\n"
                    "# Example: g = sp.Matrix([[-1, 0, 0, 0], [0, a(t)**2, 0, 0], ...])\n"
                )
            except Exception:
                pass
            try:
                import cadabra2 as cd
                sandbox_globals['cd'] = cd
                sandbox_globals['CADABRA_HINT'] = (
                    "# [PhysLib Syntax Hint: Cadabra2]\n"
                    "# Use cadabra2 for abstract tensor algebra.\n"
                    "# Example: \n"
                    "# {\\mu, \\nu, \\rho, \\sigma}::Indices(position=free).\n"
                    "# R_{\\mu \\nu \\rho \\sigma}::RiemannTensor.\n"
                    "# obj = Ex('R_{\\mu \\nu \\rho \\sigma} R^{\\mu \\nu \\rho \\sigma}')\n"
                )
            except ImportError:
                pass
            hints.append("# [Hint: Use einsteinpy for Schwarzschild/Kerr geodesics]")
            hints.append("# [Hint: Use astropy.cosmology for Friedmann-Robertson-Walker (FRW) integrations]")
            hints.append("# [Hint: Use standard SymPy matrices or sympy.tensor to define metrics]")
            hints.append("# [Hint: Use Cadabra2 for abstract tensor polynomial simplification]")
            if 'SYMPY_TENSOR_HINT' in sandbox_globals:
                hints.append(sandbox_globals['SYMPY_TENSOR_HINT'])
            hints.append(
                "# [PhysLib Syntax Hint: General Relativity & Cosmology]\n"
                "# 1. If calculating cosmic expansion, DO NOT use flat Minkowski space.\n"
                "# 2. ANSATZ ALLOWANCE: You must define a scale factor as an unknown SymPy Function:\n"
                "#    a = sp.Function('a')(t)\n"
                "#    g = dt**2 - a**2 * (dx**2 + dy**2 + dz**2)\n"
                "# 3. Use sp.lambdify to convert the resulting Friedmann equations for scipy.integrate."
            )

        # 6. Statistical Physics & Thermodynamics
        if any(kw in problem_text_lower for kw in ["partition function", "ising", "microcanonical", "entropy", "fugacity"]):
            try:
                sandbox_globals['numba'] = __import__('numba')
            except ImportError:
                pass
            hints.append("# [Hint: Use numba @jit for fast Monte Carlo and Ising partition simulations]")

        # 7. Nuclear Physics
        if any(kw in problem_text_lower for kw in ["isotope", "cross-section", "binding energy", "decay rate", "q-value"]):
            try:
                sandbox_globals['mendeleev'] = __import__('mendeleev')
            except ImportError:
                pass
            hints.append("# [Hint: Use mendeleev / nucdata to load isotope and decay bindings]")

        # 8. Nonlinear Dynamics & Chaos
        if any(kw in problem_text_lower for kw in ["lyapunov", "bifurcation", "chaos", "strange attractor", "poincare"]):
            try:
                sandbox_globals['jitcode'] = __import__('jitcode')
            except ImportError:
                pass
            hints.append("# [Hint: Use jitcode for JIT compiled Lyapunov ODE systems]")

        # 9. Fluid Dynamics & Hydrodynamic Instabilities
        fluid_kws = ["navier-stokes", "vorticity", "fluid", "rayleigh", "darcy", "convection", "porous"]
        if any(kw in problem_text_lower for kw in fluid_kws):
            try:
                sandbox_globals['dedalus'] = __import__('dedalus')
                hints.append("# [PhysLib Syntax Hint: Fluid Dynamics & PDEs]")
                hints.append("# 1. Use `dedalus` for spectral PDE solving and eigenvalue problems (like Rayleigh-Darcy).")
                hints.append("# 2. Define your domain, variables, and boundary conditions explicitly.")
            except ImportError:
                print("[WARNING] Dedalus not found. PDE eigenvalue problems will yield.")

        # 10. Biophysics
        if any(kw in problem_text_lower for kw in ["polymer", "reaction-diffusion", "turing pattern", "membrane", "protein"]):
            try:
                sandbox_globals['MDAnalysis'] = __import__('MDAnalysis')
            except ImportError:
                pass
            hints.append("# [Hint: Use MDAnalysis for biophysical polymer trajectory evaluations]")
                
        # Consolidate hints into the environment variable PHYSICS_HINT
        if hints:
            sandbox_globals['PHYSICS_HINT'] = "\n".join(hints)
            
        # Ensure PHYSICS_HINT is explicitly set to the exact requested topology hint if SYMPY_TENSOR_HINT is active
        if 'SYMPY_TENSOR_HINT' in sandbox_globals:
            sandbox_globals['PHYSICS_HINT'] = (
                "# [PhysLib Syntax Hint: General Relativity & Topology]\n"
                "# 1. `sympy.diffgeom` is DEPRECATED. Do not import or use it.\n"
                "# 2. Use standard SymPy matrices (`sp.Matrix`) or `sympy.tensor` to define metrics.\n"
                "# Example: g = sp.Matrix([[-1, 0, 0, 0], [0, a(t)**2, 0, 0], [0, 0, a(t)**2, 0], [0, 0, 0, a(t)**2]])\n"
            )

    def pre_load_physics_stack(self):
        """Pre-populates the persistent namespace with standard and advanced scientific packages."""
        import builtins
        for k in dir(builtins):
            if not k.startswith("__"):
                self.console.locals[k] = getattr(builtins, k)
        self.console.locals["__builtins__"] = builtins.__dict__
        self.console.locals["print"] = print
        
        pre_imports = [
            ("import numpy as np", "numpy"),
            ("import jax.numpy as jnp", "jax"),
            ("import scipy", "scipy"),
            ("import scipy.integrate as integrate", "scipy.integrate"),
            ("from scipy.integrate import solve_ivp", "scipy.integrate.solve_ivp"),
            ("from scipy.optimize import minimize", "scipy.optimize.minimize"),
            ("import sympy as sp", "sympy"),
            ("import mpmath as mp", "mpmath"),
            ("import networkx as nx", "networkx"),
            ("import torch", "torch"),
            ("import pandas as pd", "pandas"),
            ("import matplotlib.pyplot as plt", "matplotlib"),
            ("import qutip as qt", "qutip"),
            ("import cadabra2 as cd", "cadabra2"),
            ("from neo4j import GraphDatabase", "neo4j"),
            ("import astropy as ap", "astropy"),
            ("import astropy.units as u", "astropy"),
            ("import astropy.constants as const", "astropy"),
            ("import cvxpy as cp", "cvxpy"),
            ("import einops", "einops"),
            ("import numexpr as ne", "numexpr"),
            ("import CoolProp", "CoolProp"),
            ("import CoolProp.CoolProp as CP", "CoolProp"),
            ("import pylatexenc", "pylatexenc"),
            ("import pint", "pint"),
            ("import sympy.tensor as sp_tensor", "sympy"),
            ("import sympy.diffgeom as sp_geom", "sympy"),
            ("import pysr", "pysr")
        ]
        
        for statement, package_name in pre_imports:
            # Execute block without printing warnings to stdout
            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                try:
                    compiled = compile(statement, "<universal_repl>", "exec")
                    exec(compiled, self.console.locals)
                except Exception:
                    # Ignore missing optional package errors quietly during bootstrap
                    pass

    def truncate_output(self, text: str) -> str:
        """
        Intercepts stdout/stderr and truncates it if it exceeds the 16384 character threshold.
        Appends a SHA-256 hash summary along with leading and trailing terms.
        """
        if len(text) <= 16384:
            return text
            
        total_len = len(text)
        hash_val = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        
        leading = text[:600]
        trailing = text[-600:]
        
        truncated_msg = (
            f"\n[OUTPUT TRUNCATED: {total_len} characters. SHA-256 Hash: {hash_val}]\n"
            f"--- LEADING TERMS ---\n{leading}\n"
            f"...\n"
            f"--- TRAILING TERMS ---\n{trailing}\n"
        )
        return truncated_msg

    def execute_block(self, code_str: str) -> dict:
        """
        Executes a block of Python code statefully within the air-gapped persistent environment.
        Captures and truncates outputs, and detects SymPy relational equality failures.
        """
        # Clean up raw escape newlines if the code block is completely flattened to a single line
        if "\n" not in code_str and "\\n" in code_str:
            code_str = code_str.replace("\\n", "\n")


        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        success = True
        error_msg = ""
        
        # Stash current locals to check for newly created or modified dense variables
        before_locals = dict(self.console.locals)
        
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                # Compile and execute code block directly in locals dictionary
                compiled = compile(code_str, "<universal_repl>", "exec")
                exec(compiled, self.console.locals)
                
                # Check for SymPy Equality violations (relational contradictions)
                try:
                    import sympy
                    from sympy.logic.boolalg import BooleanFalse
                    for k, v in list(self.console.locals.items()):
                       if isinstance(v, (BooleanFalse, sympy.core.relational.Equality)) and v == False:
                            raise ValueError(f"Mathematical Contradiction: SymPy relation '{k}' evaluated to False!")
                except ImportError:
                    pass
                    
                # Verify mathematical work on the yielded target variables (Anti-Mimicry check)
                after_locals = self.console.locals
                
                import sympy as sympy_lib
                import numpy as numpy_lib
                import torch as torch_lib
                
                # Find the yielded final output target
                yielded_val = None
                if "answer" in after_locals and callable(after_locals["answer"]):
                    try:
                        yielded_val = after_locals["answer"]()
                    except Exception:
                        pass
                elif "coeffs" in after_locals:
                    yielded_val = after_locals["coeffs"]
                elif "ans" in after_locals:
                    yielded_val = after_locals["ans"]
                elif "result" in after_locals:
                    yielded_val = after_locals["result"]
                elif "sol" in after_locals:
                    yielded_val = after_locals["sol"]
                
                # If no explicit final variable is found, check any newly created/modified/assigned variables
                target_variables = []
                use_strict_yield = False
                if yielded_val is not None:
                    target_variables = [yielded_val]
                    use_strict_yield = True
                else:
                    # Extract variables explicitly assigned/defined in the current code block
                    import ast
                    defined_names = set()
                    try:
                        tree = ast.parse(code_str)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                                defined_names.add(node.id)
                            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                                defined_names.add(node.name)
                            elif isinstance(node, ast.arg):
                                defined_names.add(node.arg)
                    except Exception:
                        pass

                    for k, v in after_locals.items():
                        if k.startswith("_") or k == "compiled":
                            continue
                        
                        # If the name is explicitly defined in the current block, it is a target
                        if k in defined_names:
                            target_variables.append(v)
                            continue
                            
                        is_modified = False
                        if k not in before_locals or id(before_locals[k]) != id(v):
                            is_modified = True
                        else:
                            try:
                                diff = (before_locals[k] != v)
                                if hasattr(diff, "any"):
                                    is_modified = bool(diff.any())
                                else:
                                    is_modified = bool(diff)
                            except Exception:
                                is_modified = True
                                
                        if is_modified:
                            target_variables.append(v)
                            
                has_dense_work = False
                
                def is_dense_element(x, allow_raw_numeric=False):
                    type_str = str(type(x)).lower()
                    if isinstance(x, sympy_lib.Basic):
                        return True
                    if isinstance(x, numpy_lib.ndarray):
                        return True
                    if isinstance(x, torch_lib.Tensor):
                        return True
                    if any(cls in type_str for cls in ["matrix", "array", "tensor", "sol", "result", "opt", "ode", "symbol", "expr", "function", "type"]):
                        return True
                    if isinstance(x, numpy_lib.number):
                        return True
                    if allow_raw_numeric and isinstance(x, (float, int, complex)) and not isinstance(x, bool):
                        if x != 0:
                            return True
                    return False
                    
                def check_dense_recursive(val, allow_raw_numeric=False):
                    if is_dense_element(val, allow_raw_numeric=allow_raw_numeric):
                        return True
                    if isinstance(val, (list, tuple)):
                        return any(check_dense_recursive(item, allow_raw_numeric=allow_raw_numeric) for item in val)
                    if isinstance(val, dict):
                        return any(check_dense_recursive(item, allow_raw_numeric=allow_raw_numeric) for item in val.values())
                    return False
                    
                for var_val in target_variables:
                    # Strict yield checks require actual dense mathematical types (tensors, sympy, numpy number) rather than raw float lists
                    if check_dense_recursive(var_val, allow_raw_numeric=(not use_strict_yield)):
                        has_dense_work = True
                        break
                        
                # If dynamic rigor level is 0, completely bypass the dense work verification
                if getattr(self, "rigor_level", None) == 0:
                    has_dense_work = True
                    
                if not has_dense_work:
                    raise ValueError(
                        "Aletheia Verification Veto: The yielded variable, answer() output, or updated namespace "
                        "does not contain any mathematically dense components (tensors, sympy expressions, or non-zero numeric values). "
                        "Epistemic Fraud / Cargo Cult detected."
                    )
                    
                    
            except (Exception, SystemExit) as e:
                success = False
                error_msg = str(e)
                # Print the traceback cleanly to capture it in both stdout and stderr
                tb_str = traceback.format_exc()
                sys.stdout.write(tb_str)
                sys.stderr.write(tb_str)
                
        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()
        
        # Apply output truncation if length exceeds threshold
        stdout_val = self.truncate_output(stdout_val)
        stderr_val = self.truncate_output(stderr_val)
        
        return {
            "success": success,
            "stdout": stdout_val,
            "stderr": stderr_val,
            "error_message": error_msg
        }
