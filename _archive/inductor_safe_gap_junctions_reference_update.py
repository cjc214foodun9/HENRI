# Inside lean_darwinian_phase_swarm.py forward pass:
# DO NOT slice the tensor dynamically: active_theta = self.explorers_theta[:self.active_explorer_count]

# INSTEAD: Generate a continuous masking vector to preserve static graph dimensions for Triton
mask = torch.arange(self.max_explorers, device=self.device) < self.active_explorer_count
mask = mask.float().unsqueeze(1) # [10000, 1]

complex_active = torch.exp(1j * self.explorers_theta)
# The dormant experts' phase shifts are multiplied by 0, effectively turning them into identity operators
modulated_wave = incident_expanded * (complex_active ** (self.polarities.unsqueeze(1) * mask))