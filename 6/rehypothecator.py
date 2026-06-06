import torch
import math

class ViscoelasticGovernor:
    """
    Implements the Viscoelastic Epsilon-Spine relaxation curve.
    Determines physical Sagnac acceptance thresholds, simmer voltages,
    and shockwave triggers to govern coherent plasticity in BTO crystals.
    """
    def __init__(self, e_floor=0.05, e_ceil=0.65, kappa=3.5):
        self.e_floor = e_floor
        self.e_ceil = e_ceil
        self.kappa = kappa

    def compute_epistemic_rigidity(self, resonance_scores):
        """
        Calculates prompt strictness R in [0, 1] based on its geometric 
        alignment with the Alpha Master (Index 0: Math/Precision/Logic).
        """
        # Softmax the resonance scores to get a clean probability distribution
        probabilities = torch.nn.functional.softmax(resonance_scores, dim=-1)
        # Rigidity is strictly tied to the Alpha Master's probability
        rigidity_R = probabilities[0, 0].item()  # Assumes [Batch, Master] and Batch=1
        return rigidity_R

    def calculate_sagnac_threshold(self, rigidity_R):
        """
        Executes the non-linear Epsilon-Spine relaxation curve.
        Delta_thresh = e_floor + (e_ceil - e_floor) * exp(-kappa * R^2)
        """
        variance_allowance = (self.e_ceil - self.e_floor) * math.exp(-self.kappa * (rigidity_R ** 2))
        dynamic_threshold = self.e_floor + variance_allowance
        return dynamic_threshold

    def determine_simmer_voltage(self, sagnac_delta, dynamic_threshold):
        """
        Calculates microheater duty-cycle voltage directives based on optical stress.
        """
        if sagnac_delta <= self.e_floor:
            # Perfect Crystalline Truth (Math) - Isothermal Clamp
            return 0.0
        elif self.e_floor < sagnac_delta <= dynamic_threshold:
            # Valid but slightly noisy. Maintain Coherent Plasticity via low simmer.
            gap = dynamic_threshold - sagnac_delta
            simmer_voltage = 0.4 * (gap / (self.e_ceil - self.e_floor))
            return max(0.1, simmer_voltage)  # Never simmer below 0.1V
        else:
            # sagnac_delta > dynamic_threshold -> VETO: Langevin Shockwave
            return 2.5


class MetacognitiveRehypothecator:
    """
    L3-pinned closed-loop telemetry evaluator. Catching physical error energy
    and using it to mathematically steer subsequent speculative generations.
    """
    def __init__(self, hrr_dim=4096):
        self.hrr_dim = hrr_dim
        self.base_temperature = 0.3

    def evaluate_telemetry(self, current_hrr_wave, sagnac_delta, epiplexity_score):
        """
        Analyzes the physical telemetry and determines the next Swarm Action.
        """
        # Ensure we have scalar float numbers for metrics
        if isinstance(sagnac_delta, torch.Tensor):
            sagnac_delta = sagnac_delta.item()
        if isinstance(epiplexity_score, torch.Tensor):
            epiplexity_score = epiplexity_score.item()

        # --- STATE 1: THE ATTRACTOR COLLAPSE (TRUTH) ---
        if sagnac_delta < 0.05 and epiplexity_score > 0.90:
            print("[+] Resonance Achieved. Logic Verified.")
            return {
                "action": "TRANSLATE_TO_USER",
                "langevin_heat": 0.0,
                "vector_shift": None,
                "prompt_injection": None
            }
            
        # --- STATE 2: THE LOGIC LOCK (STAGNATION) ---
        elif 0.05 <= sagnac_delta < 0.30:
            print("[-] Logic Lock Detected. Injecting Langevin Noise.")
            # Langevin heat is proportional to error
            injected_heat = sagnac_delta * 2.5
            
            # The vector shift pushes the next representation away from the error point
            vector_shift = -0.1 * current_hrr_wave
            
            return {
                "action": "RETRY_WITH_HEAT",
                "langevin_heat": injected_heat,
                "vector_shift": vector_shift,
                "prompt_injection": "System Directive: Your previous logic resulted in a structural dead-end. Introduce lateral variance."
            }
            
        # --- STATE 3: THE SAGNAC VETO (HALLUCINATION) ---
        else:
            print("[!] SAGNAC VETO. Catastrophic Logical Contradiction.")
            # Aggressively repel from this specific geometry using the error delta
            error_geometry = current_hrr_wave * sagnac_delta
            vector_shift = -1.0 * error_geometry
            
            return {
                "action": "HARD_RESET_AND_RETRY",
                "langevin_heat": 0.8,  # High thermodynamic shake to reset the phase topology
                "vector_shift": vector_shift,
                "prompt_injection": "CRITICAL ERROR: Your previous output violated core invariants. Discard previous syntactic approach and rewrite entirely."
            }
