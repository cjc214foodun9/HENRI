# Quantized Phase Vortex & Topological Braiding Engine for Project HENRI  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class HenriTopologicalPhaseVortices(nn.Module):  
    def __init__(self, lattice_dim=64):  
        super().__init__()  
        self.dim = lattice_dim  # 64 x 64 = 4096-Dimensional VSA Space  
          
    def _reshape_to_lattice(self, unrolled_wave):  
        """  
        Transforms flat [Batch, 4096, 2] unrolled tensor into a 2D complex phase field  
        """  
        batch_size = unrolled_wave.size(0)  
        # Reconstitute real and imaginary components from the HDF5 data layer  
        real = unrolled_wave[..., 0].contiguous().view(batch_size, self.dim, self.dim)  
        imag = unrolled_wave[..., 1].contiguous().view(batch_size, self.dim, self.dim)  
        return torch.atan2(imag, real) # Returns continuous phase field [Batch, 64, 64]  
  
    def extract_vortex_charges(self, unrolled_wave):  
        """  
        Computes localized integer winding numbers (Q) over 2x2 plaquette loops  
        """  
        batch_size = unrolled_wave.size(0)  
        theta = self._reshape_to_lattice(unrolled_wave)  
          
        # Compute adjacent phase differences with periodic boundary conditions (Torus topology)  
        # Shift lattice along X and Y axes to build closed loop components  
        theta_x1 = theta  
        theta_x2 = torch.roll(theta, shifts=-1, dims=2)  
        theta_y1 = torch.roll(theta, shifts=-1, dims=1)  
        theta_y2 = torch.roll(torch.roll(theta, shifts=-1, dims=1), shifts=-1, dims=2)  
          
        # Calculate the 4 path differentials bounding every single 2x2 plaquette cell  
        # Phase steps must wrap cleanly within the [-pi, pi] circular domain  
        d1 = torch.remainder(theta_x2 - theta_x1 + torch.pi, 2 * torch.pi) - torch.pi  
        d2 = torch.remainder(theta_y2 - theta_x2 + torch.pi, 2 * torch.pi) - torch.pi  
        d3 = torch.remainder(theta_y1 - theta_y2 + torch.pi, 2 * torch.pi) - torch.pi  
        d4 = torch.remainder(theta - theta_y1 + torch.pi, 2 * torch.pi) - torch.pi  
          
        # Sum the wrapped differentials around the perimeter of the loop  
        loop_sum = d1 + d2 + d3 + d4  
          
        # Extract the quantized topological charge matrix by dividing by the 2*pi constant  
        # Rounding maps floating-point approximations directly to hard discrete integers  
        vortex_matrix = torch.round(loop_sum / (2.0 * torch.pi))  
        return vortex_matrix # Shape: [Batch, 64, 64] tracking -1, 0, and +1 charges  
  
    def forward(self, sequence_wavefronts):  
        """  
        sequence_wavefronts: [Batch, Sequence_Length, 4096, 2] tracking unrolled states  
        """  
        batch_size, seq_len, _, _ = sequence_wavefronts.size()  
        device = sequence_wavefronts.device  
          
        # Allocate tracking matrices to log vortex positions across the temporal axis  
        braid_trajectory = []  
          
        for t in range(seq_len):  
            current_frame = sequence_wavefronts[:, t, :, :] # [Batch, 4096, 2]  
            vortex_charges = self.extract_vortex_charges(current_frame) # [Batch, 64, 64]  
            braid_trajectory.append(vortex_charges.unsqueeze(1))  
              
        # Stack chronological matrices into a continuous coordinate space map  
        # Shape: [Batch, Sequence_Length, 64, 64]  
        braid_tensor_field = torch.cat(braid_trajectory, dim=1)  
          
        # Calculate Braid Coherence: evaluates the conservation of charges across time steps  
        # Pristine logical threads preserve total absolute charge configurations  
        total_charge_profile = torch.abs(braid_tensor_field).sum(dim=(2, 3)) # [Batch, Sequence_Length]  
        braid_variance = torch.var(total_charge_profile.to(torch.float32), dim=1) # [Batch]  
          
        # The braid signature serves as a coordinate-free regularizer score  
        return braid_tensor_field, braid_variance  
