import torch
from torch.utils.data import Dataset
import numpy as np

class ProceduralPhysicsDataset(Dataset):
    """
    Procedurally generates high-dimensional multi-modal states (Video/Kinematics and Audio).
    This acts as a scaffolding surrogate for massive datasets like Kinetics-400 or AudioSet,
    allowing immediate stress-testing of the HolographicTransducer.
    """
    def __init__(self, num_samples: int = 1000, target_dimension: int = 4096):
        self.num_samples = num_samples
        self.target_dimension = target_dimension
        
        # We split the 4096 target dimension to represent multi-modal fusion natively.
        # e.g., 2048 for visual kinematics, 1024 for audio, 1024 for proprioception/action context
        self.dim_vis = 2048
        self.dim_aud = 1024
        self.dim_act = 1024
        
        assert self.dim_vis + self.dim_aud + self.dim_act == self.target_dimension, "Dimensions must sum to target dimension."

    def __len__(self):
        return self.num_samples

    def _generate_visual_kinematics(self, step: int) -> torch.Tensor:
        """
        Simulates a continuous, high-dimensional visual state (e.g. object bounding boxes, fluid vectors).
        Uses sine/cosine waves to simulate continuous non-linear drift.
        """
        base_freq = 0.05
        t = step * base_freq
        
        # Create a mock feature map that evolves continuously over time
        indices = torch.arange(self.dim_vis, dtype=torch.float32)
        kinematics = torch.sin(t + indices * 0.1) * torch.cos(t * 0.5 - indices * 0.01)
        
        # Add random noise to simulate sensor artifacts
        noise = torch.randn(self.dim_vis) * 0.05
        return kinematics + noise

    def _generate_audio_spectrogram(self, step: int) -> torch.Tensor:
        """
        Simulates audio features (e.g. Mel-Frequency Cepstral Coefficients) that correlate 
        with the visual state (like a collision sound).
        """
        base_freq = 0.1
        t = step * base_freq
        
        indices = torch.arange(self.dim_aud, dtype=torch.float32)
        # Audio spikes periodically based on the visual wave frequency
        spectrogram = torch.exp(-((np.sin(t) - indices/self.dim_aud)**2) / 0.1)
        
        noise = torch.randn(self.dim_aud) * 0.02
        return spectrogram + noise

    def __getitem__(self, idx: int):
        """
        Returns:
            current_sensor_data (Tensor): The multi-modal state at t
            action_vector (Tensor): The action taken at t
            empirical_next_state (Tensor): The ground truth state at t+1
        """
        # Current state t
        vis_t = self._generate_visual_kinematics(idx)
        aud_t = self._generate_audio_spectrogram(idx)
        
        # Dummy action vector (could be simulated robot joint torques)
        act_t = torch.randn(self.dim_act)
        
        # Concatenate to fuse modalities into the 4096 latent space
        current_sensor_data = torch.cat([vis_t, aud_t, act_t], dim=0)
        
        # Action vector to pass to the JEPA core (padded to match target dimension)
        # Here we just project the action to the full space for circular convolution compatibility
        action_vector = torch.zeros(self.target_dimension)
        action_vector[-self.dim_act:] = act_t
        
        # Next state t+1
        vis_next = self._generate_visual_kinematics(idx + 1)
        aud_next = self._generate_audio_spectrogram(idx + 1)
        
        # The next state empirical observation doesn't strictly need the action that led to it 
        # in the observation itself, so we pad it.
        empirical_next_state = torch.cat([vis_next, aud_next, torch.zeros(self.dim_act)], dim=0)
        
        # Normalize vectors for the Holographic Transducer
        current_sensor_data = current_sensor_data / (torch.norm(current_sensor_data) + 1e-8)
        action_vector = action_vector / (torch.norm(action_vector) + 1e-8)
        empirical_next_state = empirical_next_state / (torch.norm(empirical_next_state) + 1e-8)
        
        return current_sensor_data, action_vector, empirical_next_state
