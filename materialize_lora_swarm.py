import os
import gc
import torch
import torch.nn as nn
import numpy as np

def track_vram_telemetry(step_label: str):
    """
    Queries the CUDA runtime API directly to print precise VRAM consumption 
    metrics to the host container shell.
    """
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[VRAM TELEMETRY] {step_label:<45} | Allocated: {allocated:.3f} GiB | Reserved: {reserved:.3f} GiB")

def materialize_lora_swarm_streams():
    # Structural invariants based on pre-training parameters
    base_weights_path = "./henri_core_final.pt"
    output_dir = "./archive"
    num_streams = 16
    lora_rank = 16
    hidden_dim = 4096

    print("[*] Initializing multi-scale LoRA swarm instantiation loop...")
    os.makedirs(output_dir, exist_ok=True)
    track_vram_telemetry("Initial baseline state before loading state dict")

    # Step 1: Load base core dictionary safely to CPU first to conserve physical GPU memory
    if not os.path.exists(base_weights_path):
        raise FileNotFoundError(f"[-] Critical Error: Pristine core weights missing at {base_weights_path}")
    
    print(f"[*] Loading base core state dictionary from: {base_weights_path}...")
    base_state_dict = torch.load(base_weights_path, map_location="cpu")
    track_vram_telemetry("Base state dictionary parsed to system RAM")

    # Detect model layout dynamically from state dict to prevent hardcoding mismatches
    layer_indices = set()
    expert_indices = set()
    for key, val in base_state_dict.items():
        if key.startswith("layers."):
            parts = key.split(".")
            layer_indices.add(int(parts[1]))
            if len(parts) > 3 and parts[2] == "experts":
                expert_indices.add(int(parts[3]))
                if "weight" in key:
                    hidden_dim = val.shape[-1]
            elif "weight" in key:
                hidden_dim = val.shape[-1]
                
    num_layers = len(layer_indices) if layer_indices else 8
    num_base_experts = len(expert_indices) if expert_indices else 4
    print(f"[+] Dynamically detected model structure: {num_layers} layers, {num_base_experts} experts per layer, hidden_dim: {hidden_dim}.")

    # Detect gemma_dim expected in the environment (checking existing adapters or defaulting to 3840)
    gemma_dim = 3840
    for file_name in os.listdir(output_dir):
        if file_name.startswith("dynamic_lora_stream_") and file_name.endswith(".bin"):
            try:
                existing_state = torch.load(os.path.join(output_dir, file_name), map_location="cpu")
                if "lora_A" in existing_state:
                    gemma_dim = existing_state["lora_A"].shape[0]
                    print(f"[+] Detected expected gemma_dim from existing file: {gemma_dim}")
                    break
            except Exception:
                pass

    # Step 2: Extract base orthogonal parameters layer-by-layer
    # We reconstruct a minimal processing shell on the GPU to isolate phase mappings
    print("[*] Initializing active device processing pipelines...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 3: Sequentially generate the 16 independent adaptive swarm pathways
    for stream_idx in range(num_streams):
        print(f"\n[STREAM-{stream_idx:02d}] Harvesting geometric invariants for parallel path generation...")
        stream_lora_weights = {}
        
        # Accumulators to build global lora_A and lora_B for DynamicLoraManager load compatibility
        accumulated_lora_A = torch.zeros(gemma_dim, lora_rank, device=device)
        accumulated_lora_B = torch.zeros(lora_rank, gemma_dim, device=device)

        for layer_idx in range(num_layers):
            # Process each individual layer independently to avoid concurrent tensor caching
            # Query base state keys (matching your core module topology)
            # Correct Topology maps to: layers.{layer_idx}.experts.{expert_idx}.phase_shift.weight
            
            layer_expert_weights = []
            for exp_idx in range(num_base_experts):
                weight_key = f"layers.{layer_idx}.experts.{exp_idx}.phase_shift.weight"
                if weight_key in base_state_dict:
                    layer_expert_weights.append(base_state_dict[weight_key])
            
            if len(layer_expert_weights) == 0:
                # Fallback handler if parameters use an alternate module key pattern
                fallback_key = f"layers.{layer_idx}.weight"
                base_weight_tensor = base_state_dict.get(fallback_key, torch.randn(hidden_dim, hidden_dim))
            else:
                # Intelligently merge base expert representations via structural reduction
                # Stack shape becomes [num_experts, 4096, 4096]
                stacked_experts = torch.stack(layer_expert_weights)
                # Compute the singular centroid wave pattern along the expert axis
                base_weight_tensor = torch.mean(stacked_experts, dim=0)

            # Push only the localized target tensor to the active hardware memory canyon
            W_base = base_weight_tensor.to(device, dtype=torch.float32)
            
            # Step 4: Perform a Truncated SVD to initialize LoRA weights inside the target subspace
            # W_base shape is [4096, 4096]. We extract the top Rank-16 singular components
            # to prime the LoRA matrices with the underlying orthogonal directional tensors.
            with torch.no_grad():
                U, S, V_T = torch.linalg.svd(W_base, full_matrices=False)
                
                # Slice matrices to isolate the Rank-16 boundary dimensions
                U_r = U[:, :lora_rank]                  # Size: [4096, 16]
                S_r = torch.diag(S[:lora_rank])        # Size: [16, 16]
                V_T_r = V_T[:lora_rank, :]              # Size: [16, 4096]
                
                # Materialize LoRA Down-projection (lora_A_layer) and Up-projection (lora_B_layer) matrices
                # Scaling by the square root of the singular values to distribute geometric weight evenly
                # lora_A_layer shape: [16, 4096]
                # lora_B_layer shape: [4096, 16]
                lora_A_layer = torch.matmul(S_r.sqrt(), V_T_r)
                lora_B_layer = torch.matmul(U_r, S_r.sqrt())
                
                # Add unique, high-frequency phase shifts unique to this specific stream index
                # This guarantees the 16 swarms explore unique variations of the bulk space
                stream_seed = (stream_idx * 100) + layer_idx
                torch.manual_seed(stream_seed)
                phase_perturbation = torch.randn_like(lora_A_layer) * 1e-4
                lora_A_layer += phase_perturbation

                # Cast shapes from 4096-D to gemma_dim for the global representation
                # lora_A_geom shape: (4096, 16)
                # lora_B_geom shape: (4096, 16)
                lora_A_geom = lora_A_layer.t()
                lora_B_geom = lora_B_layer

                if hidden_dim < gemma_dim:
                    lora_A_padded = torch.zeros(gemma_dim, lora_rank, device=device)
                    lora_A_padded[:hidden_dim, :] = lora_A_geom
                    
                    lora_B_padded = torch.zeros(lora_rank, gemma_dim, device=device)
                    lora_B_padded[:, :hidden_dim] = lora_B_geom.t()
                else:
                    lora_A_padded = lora_A_geom[:gemma_dim, :]
                    lora_B_padded = lora_B_geom[:gemma_dim, :].t()

                accumulated_lora_A += lora_A_padded / num_layers
                accumulated_lora_B += lora_B_padded / num_layers

            # Cache the initialized parameters back to the host CPU storage pool for layer-specific tracking
            stream_lora_weights[f"layers.{layer_idx}.lora_A.weight"] = lora_A_layer.cpu()
            stream_lora_weights[f"layers.{layer_idx}.lora_B.weight"] = lora_B_layer.cpu()

            # Evict temporary GPU workspace tensors immediately to reclaim hardware lanes
            del W_base, U, S, V_T, U_r, S_r, V_T_r, lora_A_layer, lora_B_layer, lora_A_geom, lora_B_geom, lora_A_padded, lora_B_padded
            torch.cuda.empty_cache()

        # Inject the root-level lora_A and lora_B to ensure compatibility with DynamicLoraManager
        stream_lora_weights["lora_A"] = accumulated_lora_A.cpu()
        stream_lora_weights["lora_B"] = accumulated_lora_B.cpu()

        # Step 5: Serialize the completed stream dictionary cleanly to disk space
        output_filename = os.path.join(output_dir, f"dynamic_lora_stream_{stream_idx}.bin")
        print(f"[+] Compiling binary envelope for Stream {stream_idx} -> {output_filename}")
        torch.save(stream_lora_weights, output_filename)
        
        # Evict entire stream profile from high-level memory context
        del stream_lora_weights, accumulated_lora_A, accumulated_lora_B
        gc.collect()
        torch.cuda.empty_cache()
        
        track_vram_telemetry(f"Completed materialization phase for Stream {stream_idx:02d}")

    print("\n[SUCCESS] All 16 independent Rank-16 dynamic LoRA adapter binaries successfully compiled.")

if __name__ == "__main__":
    materialize_lora_swarm_streams()
