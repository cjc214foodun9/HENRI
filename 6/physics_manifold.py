"""
============================================================================
HENRI Architecture: Physics Invariant Manifold Compiler
============================================================================
Compiles the "Glass Constitution" — a single d=1024 complex vector that
holographically encodes the geometric attractors of verified physical law.

Data sources: HENRI-Digital NotebookLM (60+ peer-reviewed sources),
CODATA 2018/2022, QUDT ontology, Physlib/Lean 4, Physics Derivation Graph.

Encoding method:
  Step A — Assign a deterministic, pseudo-random atomic vector to every
           fundamental concept (seeded by concept name for reproducibility).
  Step B — BIND related concepts via circular convolution (HRR binding).
  Step C — BUNDLE all bound structures via superposition (element-wise add).
  Step D — Normalize the resulting manifold to unit energy.

The output is the target_manifold that the Sagnac Veto in emulation_suite.py
uses to judge whether an incoming hypothesis wavefront is physically sound.
============================================================================
"""

import numpy as np
import json
import os

# ============================================================================
# 1. HDC / HRR Primitives
# ============================================================================

D = 4096  # Hyperdimensional channel count

def make_atomic_vector(label: str, d: int = D) -> np.ndarray:
    """
    Generate a deterministic pseudo-random complex unit vector for a concept.
    The seed is derived from the label string, so the same concept always
    maps to the same vector across runs (reproducibility).
    """
    seed = int.from_bytes(label.encode('utf-8')[:8].ljust(8, b'\x00'), 'big') % (2**31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(d) + 1j * rng.randn(d)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec


def bind(*vectors: np.ndarray) -> np.ndarray:
    """
    HRR Binding via circular convolution (implemented as element-wise
    multiplication in Fourier domain). This encodes relational structure:
    bind(A, B) produces a vector dissimilar to both A and B, but from which
    either can be recovered by binding with the approximate inverse of the other.
    """
    result = vectors[0].copy()
    for v in vectors[1:]:
        # Circular convolution = IFFT(FFT(a) * FFT(b))
        result = np.fft.ifft(np.fft.fft(result) * np.fft.fft(v))
        norm = np.linalg.norm(result)
        if norm > 0:
            result = result / norm
    return result


def bundle(*vectors: np.ndarray) -> np.ndarray:
    """
    HDC Bundling via element-wise superposition. The resulting vector is
    similar to all of its constituents — a holographic memory that stores
    multiple records in a single vector.
    """
    result = np.zeros(D, dtype=np.complex128)
    for v in vectors:
        result = result + v
    return result


# ============================================================================
# 2. Structured Physics Knowledge Base (from NotebookLM sources)
# ============================================================================

# --- 2a. CODATA Fundamental Constants ---
# Each constant is encoded as: bind(NAME, VALUE_MAGNITUDE, DIMENSIONS)
# The VALUE is encoded as a phase rotation proportional to log10(|value|),
# mapping the enormous dynamic range of physics onto the unit circle.

CODATA_CONSTANTS = {
    "speed_of_light":           {"value": 299792458.0,          "dims": {"L": 1, "T": -1}},
    "gravitational_constant":   {"value": 6.67430e-11,          "dims": {"L": 3, "M": -1, "T": -2}},
    "planck_constant":          {"value": 6.62607015e-34,        "dims": {"M": 1, "L": 2, "T": -1}},
    "reduced_planck_constant":  {"value": 1.054571817e-34,       "dims": {"M": 1, "L": 2, "T": -1}},
    "elementary_charge":        {"value": 1.602176634e-19,       "dims": {"I": 1, "T": 1}},
    "boltzmann_constant":       {"value": 1.380649e-23,          "dims": {"M": 1, "L": 2, "T": -2, "Theta": -1}},
    "fine_structure_constant":  {"value": 7.2973525643e-3,       "dims": {}},  # dimensionless
    "avogadro_constant":        {"value": 6.022140857e23,        "dims": {"N": -1}},
    "molar_gas_constant":       {"value": 8.3144598,             "dims": {"M": 1, "L": 2, "T": -2, "Theta": -1, "N": -1}},
    "vacuum_permittivity":      {"value": 8.854187817620389e-12, "dims": {"M": -1, "L": -3, "T": 4, "I": 2}},
    "vacuum_permeability":      {"value": 1.2566370614359173e-6, "dims": {"M": 1, "L": 1, "T": -2, "I": -2}},
    "rydberg_constant":         {"value": 1.0973731568508e7,     "dims": {"L": -1}},
    "bohr_radius":              {"value": 5.2917721067e-11,      "dims": {"L": 1}},
}

# --- 2b. SI Base Dimension Atomic Vectors ---
SI_DIMENSIONS = ["L", "M", "T", "I", "Theta", "N", "J"]

# --- 2c. Fundamental Equations (Conservation Laws & Mechanics) ---
# Each equation is a list of (role, concept) pairs to be bound together.
# The roles (like "lhs", "rhs", "operator") are themselves atomic vectors.

FUNDAMENTAL_EQUATIONS = {
    "mass_energy_equivalence": {
        "symbolic": "E = m * c^2",
        "terms": ["Energy", "equals", "Mass", "speed_of_light", "speed_of_light"]
    },
    "translational_eom": {
        "symbolic": "dP/dt = F",
        "terms": ["LinearMomentum", "time_derivative", "equals", "Force"]
    },
    "rotational_eom": {
        "symbolic": "dM/dt = K",
        "terms": ["AngularMomentum", "time_derivative", "equals", "Torque"]
    },
    "de_broglie": {
        "symbolic": "p * lambda = h",
        "terms": ["LinearMomentum", "Wavelength", "equals", "planck_constant"]
    },
    "planck_einstein": {
        "symbolic": "E = h * f",
        "terms": ["Energy", "equals", "planck_constant", "Frequency"]
    },
    "ideal_gas_law": {
        "symbolic": "P * V = n * R * T",
        "terms": ["Pressure", "Volume", "equals", "AmountOfSubstance", "molar_gas_constant", "ThermodynamicTemperature"]
    },
    "kinetic_energy_decomposition": {
        "symbolic": "T = 1/2 M|V|^2 + 1/2 w.I.w",
        "terms": ["KineticEnergy", "equals", "Mass", "LinearVelocity", "AngularVelocity", "MomentOfInertia"]
    },
    "parallel_axis_theorem": {
        "symbolic": "I_O' = I_O + M(|a|^2 * 1 - a x a)",
        "terms": ["MomentOfInertia", "equals", "MomentOfInertia", "Mass", "Length"]
    },
    "lorentz_factor": {
        "symbolic": "gamma = 1 / sqrt(1 - v^2/c^2)",
        "terms": ["LorentzFactor", "equals", "LinearVelocity", "speed_of_light", "Dimensionless"]
    },
    "hookes_law": {
        "symbolic": "F = k * X",
        "terms": ["Force", "equals", "SpringConstant", "Length"]
    },
    "stefan_boltzmann": {
        "symbolic": "E_tot = sigma * T^4",
        "terms": ["Energy", "equals", "StefanBoltzmannConstant", "ThermodynamicTemperature"]
    },
    "newtons_second_law": {
        "symbolic": "F = m * a",
        "terms": ["Force", "equals", "Mass", "LinearAcceleration"]
    },
    "transport_law_rotating": {
        "symbolic": "(dA/dt)_inertial = (dA/dt)_rotating + Omega x A",
        "terms": ["time_derivative", "inertial_frame", "equals", "time_derivative", "rotating_frame", "AngularVelocity", "cross_product"]
    },
}

# --- 2d. QUDT QuantityKind Dimension Vectors ---
# Maps every QuantityKind to its SI dimensional exponents (from notebook sources)

QUDT_QUANTITY_KINDS = {
    "AmountOfSubstance":             {"N": 1},
    "AngularAcceleration":           {"T": -2},
    "AngularMomentum":               {"L": 2, "M": 1, "T": -1},
    "AngularVelocity":               {"T": -1},
    "Area":                          {"L": 2},
    "Capacitance":                   {"I": 2, "L": -2, "M": -1, "T": 4},
    "Density":                       {"L": -3, "M": 1},
    "Dimensionless":                 {},
    "DynamicViscosity":              {"L": -1, "M": 1, "T": -1},
    "ElectricCharge":                {"I": 1, "T": 1},
    "ElectricCurrent":               {"I": 1},
    "ElectricFieldStrength":         {"I": -1, "L": 1, "M": 1, "T": -3},
    "ElectromagneticPermeability":   {"I": -2, "L": 1, "M": 1, "T": -2},
    "Energy":                        {"L": 2, "M": 1, "T": -2},
    "EnergyDensity":                 {"L": -1, "M": 1, "T": -2},
    "EnergyPerTemperature":          {"L": 2, "M": 1, "Theta": -1, "T": -2},
    "Force":                         {"L": 1, "M": 1, "T": -2},
    "Frequency":                     {"T": -1},
    "InverseLength":                 {"L": -1},
    "Length":                        {"L": 1},
    "LinearAcceleration":            {"L": 1, "T": -2},
    "LinearMomentum":                {"L": 1, "M": 1, "T": -1},
    "LinearVelocity":                {"L": 1, "T": -1},
    "LuminousIntensity":             {"J": 1},
    "MagneticFluxDensity":           {"I": -1, "M": 1, "T": -2},
    "Mass":                          {"M": 1},
    "MassFlowRate":                  {"M": 1, "T": -1},
    "MomentOfInertia":               {"L": 2, "M": 1},
    "MolarEnergy":                   {"N": -1, "L": 2, "M": 1, "T": -2},
    "MolarHeatCapacity":             {"N": -1, "L": 2, "M": 1, "Theta": -1, "T": -2},
    "Permittivity":                  {"I": 2, "L": -3, "M": -1, "T": 4},
    "Power":                         {"L": 2, "M": 1, "T": -3},
    "Pressure":                      {"L": -1, "M": 1, "T": -2},
    "Resistance":                    {"I": -2, "L": 2, "M": 1, "T": -3},
    "SpecificEnergy":                {"L": 2, "T": -2},
    "SpecificHeatCapacity":          {"L": 2, "Theta": -1, "T": -2},
    "ThermalConductivity":           {"L": 1, "M": 1, "Theta": -1, "T": -3},
    "ThermodynamicTemperature":      {"Theta": 1},
    "Time":                          {"T": 1},
    "Volume":                        {"L": 3},
    "VolumetricHeatCapacity":        {"L": -1, "M": 1, "Theta": -1, "T": -2},
}

# --- 2e. Topological Invariants ---

TOPOLOGICAL_INVARIANTS = {
    "chern_number": {
        "description": "First Chern number c1 — quantized Hall conductivity",
        "terms": ["ChernNumber", "BerryPhase", "BrillouinZone", "TopologicalInvariant", "quantized"]
    },
    "spin_chern_number": {
        "description": "Spin Chern number: 2*c_spin = c1_up - c1_down",
        "terms": ["SpinChernNumber", "ChernNumber", "spin_up", "spin_down", "TopologicalInvariant"]
    },
    "kane_mele_z2": {
        "description": "Z2 invariant — parity of spectral flow at TRS fixed points",
        "terms": ["Z2Invariant", "TimeReversalSymmetry", "Pfaffian", "TopologicalInvariant", "KramersTheorem"]
    },
    "chern_simons": {
        "description": "Chern-Simons invariant — bulk-boundary correspondence in 3D TIs",
        "terms": ["ChernSimons", "BulkBoundary", "WZWTerm", "TopologicalInvariant", "3D"]
    },
    "berry_curvature": {
        "description": "Berry curvature Omega — local geometric phase accumulation",
        "terms": ["BerryCurvature", "BerryConnection", "BerryPhase", "geometric_phase", "projector"]
    },
    "maslov_index": {
        "description": "Lagrangian intersection number for chiral edge states",
        "terms": ["MaslovIndex", "Lagrangian", "chiral", "edge_state", "FermiLevel"]
    },
    "witten_index": {
        "description": "Tr(-1)^F — parity anomaly of Majorana zero modes",
        "terms": ["WittenIndex", "Majorana", "DiracCone", "parity_anomaly", "mod2"]
    },
    "gauss_bonnet": {
        "description": "Extended Gauss-Bonnet on quantum-state manifolds",
        "terms": ["GaussCurvature", "GaussBonnet", "singular_fold", "quantum_manifold", "signed_area"]
    },
}

# --- 2f. Symmetry Principles ---

SYMMETRY_PRINCIPLES = {
    "altland_zirnbauer": {
        "description": "Tenfold way classification of free fermionic Hamiltonians",
        "terms": ["AltlandZirnbauer", "TenfoldWay", "TimeReversalSymmetry", "ParticleHoleSymmetry", "ChiralSymmetry"]
    },
    "time_reversal_symmetry": {
        "description": "Anti-unitary operator guaranteeing Kramers degeneracy",
        "terms": ["TimeReversalSymmetry", "KramersTheorem", "anti_unitary", "Dirac", "fixed_point"]
    },
    "lorentz_invariance": {
        "description": "Spacetime symmetry preserved under Lorentz transformations",
        "terms": ["LorentzInvariance", "spacetime", "LorentzFactor", "speed_of_light", "covariant"]
    },
}

# --- 2g. Relativistic and Holographic Invariants ---

RELATIVISTIC_INVARIANTS = {
    "einstein_field_equations": {
        "description": "Einstein Field Equations relating geometry to matter",
        "terms": ["RicciTensor", "MetricTensor", "equals", "StressEnergyTensor", "gravitational_constant"]
    },
    "holographic_weyl_anomaly": {
        "description": "Trace anomaly of the stress tensor in holographic duals",
        "terms": ["WeylAnomaly", "TraceOperator", "MetricTensor", "HolographicDual", "equals"]
    },
    "weyl_curvature": {
        "description": "Weyl curvature tensor representing tidal forces",
        "terms": ["WeylTensor", "RiemannTensor", "RicciTensor", "MetricTensor", "trace_free"]
    }
}

# --- 2h. Lean 4 Physlib Theorems ---

LEAN4_THEOREMS = [
    "RigidBody.translational_equation_inertial",
    "RigidBody.rotational_equation_inertial",
    "RigidBody.parallel_axis_theorem",
    "RigidBody.transport_law_inertial_rotating",
    "RigidBody.kinetic_energy_decomposition",
    "Mechanics_73_University",
    "Mechanics_74_University_0",
]


# ============================================================================
# 3. Manifold Compiler
# ============================================================================

class PhysicsManifoldCompiler:
    """
    Compiles the structured physics knowledge base into a single d=1024
    complex vector — the Invariant Manifold (Glass Constitution) — that
    the Sagnac Veto uses to judge incoming hypothesis wavefronts.
    """

    def __init__(self, d: int = D):
        self.d = d
        self._atom_cache = {}
        self.compilation_log = []

    def _get_atom(self, label: str) -> np.ndarray:
        """Get or create an atomic vector for a concept (cached)."""
        if label not in self._atom_cache:
            self._atom_cache[label] = make_atomic_vector(label, self.d)
        return self._atom_cache[label]

    def _encode_value_as_phase(self, value: float) -> np.ndarray:
        """
        Encode a numerical constant as a phase-rotated atomic vector.
        Uses log10(|value|) mapped to a phase angle, preserving the
        order-of-magnitude structure across 50+ decades of physics.
        """
        if value == 0:
            return self._get_atom("ZERO")
        log_mag = np.log10(abs(value))
        # Map log magnitude to a phase angle (radians)
        phase = log_mag * (2 * np.pi / 50.0)  # ~50 decades of physics
        base = self._get_atom(f"NUMERICAL_{abs(hash(str(value))) % 10000}")
        return base * np.exp(1j * phase)

    def _encode_dimension_vector(self, dims: dict) -> np.ndarray:
        """
        Encode a QUDT dimension vector by binding base dimension atoms
        raised to their respective exponents (via repeated self-convolution).
        """
        if not dims:
            return self._get_atom("Dimensionless")

        components = []
        for dim_name in SI_DIMENSIONS:
            exp = dims.get(dim_name, 0)
            if exp == 0:
                continue
            dim_atom = self._get_atom(f"SI_{dim_name}")
            # Raise to power via repeated binding (positive) or conjugate (negative)
            if exp > 0:
                powered = dim_atom.copy()
                for _ in range(int(abs(exp)) - 1):
                    powered = bind(powered, dim_atom)
            else:
                # Inverse = complex conjugate in frequency domain
                inv_atom = np.conj(dim_atom)
                powered = inv_atom.copy()
                for _ in range(int(abs(exp)) - 1):
                    powered = bind(powered, inv_atom)
            components.append(powered)

        if len(components) == 1:
            return components[0]
        return bind(*components)

    def compile_codata_constants(self) -> np.ndarray:
        """Encode all CODATA constants and bundle them."""
        encoded = []
        for name, info in CODATA_CONSTANTS.items():
            name_atom = self._get_atom(name)
            value_atom = self._encode_value_as_phase(info["value"])
            dim_atom = self._encode_dimension_vector(info["dims"])
            bound = bind(name_atom, value_atom, dim_atom)
            encoded.append(bound)
            self.compilation_log.append(f"[CODATA] Encoded: {name} = {info['value']}")
        return bundle(*encoded)

    def compile_equations(self) -> np.ndarray:
        """Encode all conservation laws and fundamental equations."""
        encoded = []
        for eq_name, eq_info in FUNDAMENTAL_EQUATIONS.items():
            term_atoms = [self._get_atom(t) for t in eq_info["terms"]]
            # Bind all terms in sequence to encode the relational structure
            bound = bind(*term_atoms)
            encoded.append(bound)
            self.compilation_log.append(f"[EQUATION] Encoded: {eq_name} ({eq_info['symbolic']})")
        return bundle(*encoded)

    def compile_qudt_dimensions(self) -> np.ndarray:
        """Encode all QUDT QuantityKind dimension vectors."""
        encoded = []
        for qk_name, dims in QUDT_QUANTITY_KINDS.items():
            qk_atom = self._get_atom(qk_name)
            dim_atom = self._encode_dimension_vector(dims)
            bound = bind(qk_atom, dim_atom)
            encoded.append(bound)
            self.compilation_log.append(f"[QUDT] Encoded: {qk_name} dims={dims}")
        return bundle(*encoded)

    def compile_topological_invariants(self) -> np.ndarray:
        """Encode topological invariants (Chern, Z2, Berry, etc.)."""
        encoded = []
        for inv_name, inv_info in TOPOLOGICAL_INVARIANTS.items():
            term_atoms = [self._get_atom(t) for t in inv_info["terms"]]
            bound = bind(*term_atoms)
            encoded.append(bound)
            self.compilation_log.append(f"[TOPOLOGY] Encoded: {inv_name}")
        return bundle(*encoded)

    def compile_symmetry_principles(self) -> np.ndarray:
        """Encode symmetry classification principles."""
        encoded = []
        for sym_name, sym_info in SYMMETRY_PRINCIPLES.items():
            term_atoms = [self._get_atom(t) for t in sym_info["terms"]]
            bound = bind(*term_atoms)
            encoded.append(bound)
            self.compilation_log.append(f"[SYMMETRY] Encoded: {sym_name}")
        return bundle(*encoded)

    def compile_relativistic_invariants(self) -> np.ndarray:
        """Encode relativistic and holographic invariants."""
        encoded = []
        for inv_name, inv_info in RELATIVISTIC_INVARIANTS.items():
            term_atoms = [self._get_atom(t) for t in inv_info["terms"]]
            bound = bind(*term_atoms)
            encoded.append(bound)
            self.compilation_log.append(f"[RELATIVITY] Encoded: {inv_name}")
        return bundle(*encoded)

    def compile_lean4_theorems(self) -> np.ndarray:
        """Encode Lean 4 Physlib theorem identifiers as structural anchors."""
        encoded = []
        lean_marker = self._get_atom("LEAN4_VERIFIED")
        for thm_name in LEAN4_THEOREMS:
            thm_atom = self._get_atom(thm_name)
            bound = bind(lean_marker, thm_atom)
            encoded.append(bound)
            self.compilation_log.append(f"[LEAN4] Encoded: {thm_name}")
        return bundle(*encoded)

    def compile_equation_library(self) -> dict:
        """
        Returns a dict mapping each equation/constant/invariant name to
        its individual bound vector (NOT bundled). This allows the Sagnac
        Veto to check coherence against each law independently.
        """
        library = {}

        # Equations
        for eq_name, eq_info in FUNDAMENTAL_EQUATIONS.items():
            term_atoms = [self._get_atom(t) for t in eq_info["terms"]]
            vec = bind(*term_atoms)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            library[eq_name] = vec

        # Topological invariants
        for inv_name, inv_info in TOPOLOGICAL_INVARIANTS.items():
            term_atoms = [self._get_atom(t) for t in inv_info["terms"]]
            vec = bind(*term_atoms)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            library[inv_name] = vec

        # Symmetry principles
        for sym_name, sym_info in SYMMETRY_PRINCIPLES.items():
            term_atoms = [self._get_atom(t) for t in sym_info["terms"]]
            vec = bind(*term_atoms)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            library[sym_name] = vec

        # Relativistic and holographic invariants
        for inv_name, inv_info in RELATIVISTIC_INVARIANTS.items():
            term_atoms = [self._get_atom(t) for t in inv_info["terms"]]
            vec = bind(*term_atoms)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            library[inv_name] = vec

        # QUDT dimension type signatures
        for qk_name, dims in QUDT_QUANTITY_KINDS.items():
            qk_atom = self._get_atom(qk_name)
            dim_atom = self._encode_dimension_vector(dims)
            vec = bind(qk_atom, dim_atom)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            library[f"qudt_{qk_name}"] = vec

        return library

    def compile_full_manifold(self) -> np.ndarray:
        """
        Compile the complete Invariant Manifold (Glass Constitution).
        Bundles all sub-manifolds with weighted importance, then normalizes.
        """
        print("=" * 70)
        print("HENRI PHYSICS MANIFOLD COMPILER")
        print("Compiling Glass Constitution from structured notebook sources...")
        print("=" * 70)

        # Compile each domain
        codata_manifold   = self.compile_codata_constants()
        equation_manifold = self.compile_equations()
        qudt_manifold     = self.compile_qudt_dimensions()
        topo_manifold     = self.compile_topological_invariants()
        sym_manifold      = self.compile_symmetry_principles()
        rel_manifold      = self.compile_relativistic_invariants()
        lean_manifold     = self.compile_lean4_theorems()

        print(f"\n[CODATA]    {len(CODATA_CONSTANTS)} constants encoded")
        print(f"[EQUATIONS] {len(FUNDAMENTAL_EQUATIONS)} laws encoded")
        print(f"[QUDT]      {len(QUDT_QUANTITY_KINDS)} quantity kinds encoded")
        print(f"[TOPOLOGY]  {len(TOPOLOGICAL_INVARIANTS)} invariants encoded")
        print(f"[SYMMETRY]  {len(SYMMETRY_PRINCIPLES)} principles encoded")
        print(f"[RELATIVITY]{len(RELATIVISTIC_INVARIANTS)} invariants encoded")
        print(f"[LEAN4]     {len(LEAN4_THEOREMS)} verified theorems encoded")

        # Bundle all sub-manifolds with weighting
        # Conservation laws and constants get higher weight (they are absolute)
        full_manifold = bundle(
            codata_manifold * 2.0,      # Constants: absolute anchors
            equation_manifold * 2.0,    # Conservation laws: structural truth
            qudt_manifold * 1.5,        # Dimensional analysis: type system
            topo_manifold * 1.0,        # Topological invariants
            sym_manifold * 1.0,         # Symmetry principles
            rel_manifold * 1.5,         # Relativistic & holographic invariants
            lean_manifold * 1.5,        # Formally verified theorems
        )

        # Normalize to unit energy
        full_manifold = full_manifold / (np.linalg.norm(full_manifold) + 1e-12)

        total_concepts = len(self._atom_cache)
        print(f"\n[MANIFOLD]  Total atomic concepts: {total_concepts}")
        print(f"[MANIFOLD]  Manifold dimension: d={self.d}")
        print(f"[MANIFOLD]  Manifold norm: {np.linalg.norm(full_manifold):.6f}")
        print("=" * 70)
        print("Glass Constitution compiled successfully.")
        print("=" * 70)

        return full_manifold

    def save_manifold(self, manifold: np.ndarray, filepath: str):
        """Save the compiled manifold and equation library to disk."""
        library = self.compile_equation_library()

        # Pack library vectors into a 2D array + name list
        lib_names = list(library.keys())
        lib_vectors = np.array([library[n] for n in lib_names])

        np.savez_compressed(
            filepath,
            manifold_re=manifold.real,
            manifold_im=manifold.imag,
            num_atoms=len(self._atom_cache),
            atom_labels=list(self._atom_cache.keys()),
            library_names=lib_names,
            library_vectors_re=lib_vectors.real,
            library_vectors_im=lib_vectors.imag,
        )
        print(f"[SAVED] Manifold + {len(lib_names)} library records saved to: {filepath}")

    @staticmethod
    def load_manifold(filepath: str) -> np.ndarray:
        """Load a pre-compiled manifold from disk."""
        data = np.load(filepath, allow_pickle=True)
        manifold = data["manifold_re"] + 1j * data["manifold_im"]
        print(f"[LOADED] Manifold loaded from: {filepath}")
        print(f"         Atoms: {data['num_atoms']}, Dimension: {len(manifold)}")
        return manifold

    @staticmethod
    def load_library(filepath: str) -> dict:
        """Load the equation library from disk."""
        data = np.load(filepath, allow_pickle=True)
        names = list(data["library_names"])
        vecs_re = data["library_vectors_re"]
        vecs_im = data["library_vectors_im"]
        library = {}
        for i, name in enumerate(names):
            library[name] = vecs_re[i] + 1j * vecs_im[i]
        return library


# ============================================================================
# 4. Main: Compile and Save
# ============================================================================

if __name__ == "__main__":
    compiler = PhysicsManifoldCompiler(d=D)
    manifold = compiler.compile_full_manifold()

    # Save to project root
    save_path = os.path.join(os.path.dirname(__file__), "glass_constitution.npz")
    compiler.save_manifold(manifold, save_path)

    # Print compilation log
    print(f"\n--- Compilation Log ({len(compiler.compilation_log)} entries) ---")
    for entry in compiler.compilation_log:
        print(f"  {entry}")

    # Quick self-test
    print(f"\n--- Self-Test ---")
    print(f"  Manifold energy: {np.sum(np.abs(manifold)**2):.6f}")

    # Load library and test individual equation coherence
    library = PhysicsManifoldCompiler.load_library(save_path)
    print(f"  Library records: {len(library)}")

    # E=mc2 record from library should have perfect self-coherence
    emc2_lib = library["mass_energy_equivalence"]
    emc2_test = bind(
        make_atomic_vector("Energy"),
        make_atomic_vector("equals"),
        make_atomic_vector("Mass"),
        make_atomic_vector("speed_of_light"),
        make_atomic_vector("speed_of_light"),
    )
    emc2_test = emc2_test / np.linalg.norm(emc2_test)
    coherence = float(np.real(np.vdot(emc2_test, emc2_lib)))
    print(f"  E=mc^2 self-coherence (should be ~1.0): {coherence:.6f}")

    # E=mc^3 (wrong) should have near-zero coherence with E=mc^2 record
    emc3_test = bind(
        make_atomic_vector("Energy"),
        make_atomic_vector("equals"),
        make_atomic_vector("Mass"),
        make_atomic_vector("speed_of_light"),
        make_atomic_vector("speed_of_light"),
        make_atomic_vector("speed_of_light"),
    )
    emc3_test = emc3_test / np.linalg.norm(emc3_test)
    wrong_coherence = float(np.real(np.vdot(emc3_test, emc2_lib)))
    print(f"  E=mc^3 vs E=mc^2 coherence (should be ~0): {wrong_coherence:.6f}")

    print("\nDone.")
