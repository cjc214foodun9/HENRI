import torch
import torch.nn as torch_nn
import torch.nn.functional as F

class ContractiveHebbianLayer(torch_nn.Module):
    """
    1. Crystallized Embeddings via Contractive Hebbian Plasticity
    Uses Oja's Rule (a stabilized Hebbian mechanism) to continuously update
    weights based on input-output correlations, crystallizing the sensory 
    manifold without relying on backpropagation.
    """
    def __init__(self, in_features, out_features, learning_rate=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lr = learning_rate
        
        # Initialize weights randomly, but orthogonal for stability
        self.weight = torch_nn.Parameter(torch.empty(out_features, in_features))
        torch_nn.init.orthogonal_(self.weight)
        
    def forward(self, x):
        # x shape: (batch_size, in_features)
        y = F.linear(x, self.weight)
        
        if self.training:
            # Apply Oja's rule: dW = lr * (y * x^T - y^2 * W)
            # This forces the weight vectors to align with principal components
            # of the input space, acting as a contractive mechanism.
            with torch.no_grad():
                batch_size = x.size(0)
                y_unsqueeze = y.unsqueeze(2) # (batch, out, 1)
                x_unsqueeze = x.unsqueeze(1) # (batch, 1, in)
                
                # y * x^T
                hebbian_term = torch.bmm(y_unsqueeze, x_unsqueeze) 
                
                # y^2 * W (Contraction/Forgetting term to prevent explosion)
                y_sq = (y ** 2).unsqueeze(2) # (batch, out, 1)
                weight_unsqueeze = self.weight.unsqueeze(0).expand(batch_size, -1, -1)
                forget_term = y_sq * weight_unsqueeze
                
                # Average update over the batch
                delta_w = torch.mean(hebbian_term - forget_term, dim=0)
                self.weight.add_(self.lr * delta_w)
                
        return y


class FEPOrthogonalizer(torch_nn.Module):
    """
    2. Self-Orthogonalizing Attractors via the Free Energy Principle
    Implements Sanger's Rule (Generalized Hebbian Algorithm). This forces
    the network to partition the space into local, orthogonal inferences,
    maximizing mutual information and memory capacity.
    """
    def __init__(self, features, learning_rate=0.01):
        super().__init__()
        self.features = features
        self.lr = learning_rate
        self.weight = torch_nn.Parameter(torch.empty(features, features))
        torch_nn.init.eye_(self.weight) # Start as identity
        
        # Lower triangular mask for Sanger's sequential orthogonalization
        self.register_buffer('tril_mask', torch.tril(torch.ones(features, features)))

    def forward(self, x):
        y = F.linear(x, self.weight)
        
        if self.training:
            with torch.no_grad():
                batch_size = x.size(0)
                # Compute Sanger's update
                # dW_ij = lr * y_i * (x_j - sum_{k=1}^i y_k W_kj)
                
                y_sq_matrix = torch.bmm(y.unsqueeze(2), y.unsqueeze(1)) # y_i * y_k
                masked_y = y_sq_matrix * self.tril_mask.unsqueeze(0)
                
                lateral_inhibition = torch.bmm(masked_y, self.weight.unsqueeze(0).expand(batch_size, -1, -1))
                hebbian = torch.bmm(y.unsqueeze(2), x.unsqueeze(1))
                
                delta_w = torch.mean(hebbian - lateral_inhibition, dim=0)
                self.weight.add_(self.lr * delta_w)
                
        return y


class TopologicalClosureMemory(torch_nn.Module):
    """
    3. Topological Closure (Partial^2 = 0) & Memory-Amortized Inference (MAI)
    Tracks transient scaffolds (H_0 dots) over time. When a cycle is detected 
    (the network returns to a highly similar state), it "closes" the topology 
    by amortizing the sequence into a direct attractor mapping (H_1 cycle).
    """
    def __init__(self, state_dim, history_length=50, sim_threshold=0.95):
        super().__init__()
        self.state_dim = state_dim
        self.history_length = history_length
        self.sim_threshold = sim_threshold
        
        # Ring buffer for state trajectory (Dots / H_0)
        self.register_buffer('trajectory_buffer', torch.zeros(history_length, state_dim))
        self.ptr = 0
        self.buffer_filled = False
        
        # Amortized mapping: maps from a state directly to its cycle invariant
        self.amortized_projection = torch_nn.Linear(state_dim, state_dim)
        self.optimizer = torch.optim.Adam(self.amortized_projection.parameters(), lr=0.005)

    def forward(self, current_state):
        batch_mean_state = current_state.mean(dim=0).detach() # Track macro-state
        
        # 1. Check for topological closure (Cycles / H_1)
        if self.buffer_filled or self.ptr > 5:
            # Compare current state against history using Cosine Similarity
            history_valid = self.history_length if self.buffer_filled else self.ptr
            valid_history = self.trajectory_buffer[:history_valid]
            
            sims = F.cosine_similarity(batch_mean_state.unsqueeze(0), valid_history)
            
            # Find the most similar past state, ignoring the immediate previous step
            sims[-1] = 0.0 
            max_sim, max_idx = torch.max(sims, dim=0)
            
            if max_sim > self.sim_threshold:
                # Cycle detected! The topology has closed (partial^2 = 0 analogy).
                # Amortize the inference: Train the projection to jump directly to this invariant
                cycle_start_state = self.trajectory_buffer[max_idx]
                
                # Active Inference step: Minimize surprise by predicting the invariant
                with torch.enable_grad():
                    predicted_invariant = self.amortized_projection(cycle_start_state)
                    loss = F.mse_loss(predicted_invariant, batch_mean_state)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                
                # Substitute the current transient computation with the amortized invariant
                # This dynamically reshapes the topology based on stable regularities
                current_state = current_state + 0.1 * self.amortized_projection(current_state)

        # 2. Update transient scaffold (Dots / H_0)
        self.trajectory_buffer[self.ptr] = batch_mean_state
        self.ptr = (self.ptr + 1) % self.history_length
        if self.ptr == 0:
            self.buffer_filled = True
            
        return current_state


class EmergentManifold(torch_nn.Module):
    """
    Full integrated module replacing engineered vortexes.
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.crystallizer = ContractiveHebbianLayer(in_features, hidden_features)
        self.orthogonalizer = FEPOrthogonalizer(hidden_features)
        self.topology_closer = TopologicalClosureMemory(hidden_features)
        
    def forward(self, x):
        x = x.to(device=self.crystallizer.weight.device, dtype=self.crystallizer.weight.dtype)
        # 1. Crystallize structure from environmental signals
        embedded = self.crystallizer(x)
        embedded = torch.tanh(embedded)
        
        # 2. Orthogonalize attractors via Free Energy minimization
        orthogonal = self.orthogonalizer(embedded)
        orthogonal = torch.relu(orthogonal)
        
        # 3. Apply topological closure to lock in persistent regularities
        invariant_state = self.topology_closer(orthogonal)
        
        return invariant_state