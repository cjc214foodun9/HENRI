import torch
import torch.nn as nn
import os
import sys

def scale_checkpoint(input_path, output_path, target_dim=4096, target_depth=32, target_num_fluid_states=16):
    print(f"Loading checkpoint from {input_path}...")
    checkpoint = torch.load(input_path, map_location='cpu')
    
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Invalid checkpoint format: expected a dict with 'model_state_dict'")
        
    cfg = checkpoint.get("config", {})
    orig_dim = cfg.get("dim", 256)
    orig_depth = cfg.get("depth", 2)
    orig_num_fluid_states = cfg.get("num_fluid_states", 4)
    orig_vocab_size = cfg.get("vocab_size", 32000)
    
    print(f"Original geometry: dim={orig_dim}, depth={orig_depth}, num_fluid_states={orig_num_fluid_states}, vocab_size={orig_vocab_size}")
    print(f"Target geometry: dim={target_dim}, depth={target_depth}, num_fluid_states={target_num_fluid_states}, vocab_size={orig_vocab_size}")
    
    orig_model_sd = checkpoint["model_state_dict"]
    new_model_sd = {}
    
    # 1. Scale layers
    for i in range(target_depth):
        i_orig = i % orig_depth
        
        # Scale output_binding_geometry
        orig_binding = orig_model_sd[f"layers.{i_orig}.output_binding_geometry"]
        new_binding = torch.zeros(1, target_dim, dtype=orig_binding.dtype)
        new_binding[0, :orig_dim] = orig_binding[0]
        new_model_sd[f"layers.{i}.output_binding_geometry"] = new_binding
        
        # Scale router.phase_attractors
        orig_attractors = orig_model_sd[f"layers.{i_orig}.router.phase_attractors"]
        new_attractors = torch.zeros(target_num_fluid_states, target_dim, dtype=orig_attractors.dtype)
        for r in range(target_num_fluid_states):
            r_orig = r % orig_num_fluid_states
            new_attractors[r, :orig_dim] = orig_attractors[r_orig]
        new_model_sd[f"layers.{i}.router.phase_attractors"] = new_attractors
        
        # Scale router.beta
        orig_beta = orig_model_sd[f"layers.{i_orig}.router.beta"]
        new_model_sd[f"layers.{i}.router.beta"] = orig_beta.clone()
        
        # Scale expert weights
        for j in range(target_num_fluid_states):
            j_orig = j % orig_num_fluid_states
            orig_expert_w = orig_model_sd[f"layers.{i_orig}.experts.{j_orig}.phase_shift.weight"]
            
            # Use identity matrix for the scaling padding
            new_expert_w = torch.eye(target_dim, dtype=orig_expert_w.dtype)
            new_expert_w[:orig_dim, :orig_dim] = orig_expert_w
            new_model_sd[f"layers.{i}.experts.{j}.phase_shift.weight"] = new_expert_w
            
    # 2. Scale final layer norm
    if "final_layer_norm.weight" in orig_model_sd:
        orig_ln_w = orig_model_sd["final_layer_norm.weight"]
        new_ln_w = torch.ones(target_dim, dtype=orig_ln_w.dtype)
        new_ln_w[:orig_dim] = orig_ln_w
        new_model_sd["final_layer_norm.weight"] = new_ln_w
        
    if "final_layer_norm.bias" in orig_model_sd:
        orig_ln_b = orig_model_sd["final_layer_norm.bias"]
        new_ln_b = torch.zeros(target_dim, dtype=orig_ln_b.dtype)
        new_ln_b[:orig_dim] = orig_ln_b
        new_model_sd["final_layer_norm.bias"] = new_ln_b
        
    # 3. Scale translation head
    new_trans_sd = {}
    orig_trans_sd = checkpoint.get("translation_head_state_dict")
    if orig_trans_sd is not None:
        print("Scaling translation head state dict...")
        orig_th_w = orig_trans_sd["weight"]
        new_th_w = torch.zeros(orig_vocab_size, target_dim, dtype=orig_th_w.dtype)
        new_th_w[:, :orig_dim] = orig_th_w
        new_trans_sd["weight"] = new_th_w
        
        if "bias" in orig_trans_sd:
            new_trans_sd["bias"] = orig_trans_sd["bias"].clone()
            
    # Cast all float tensors to bfloat16 to optimize storage footprint
    print("Casting parameters to bfloat16...")
    for k, v in new_model_sd.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            new_model_sd[k] = v.to(dtype=torch.bfloat16)
            
    if orig_trans_sd is not None:
        for k, v in new_trans_sd.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                new_trans_sd[k] = v.to(dtype=torch.bfloat16)
            
    # Assemble final checkpoint dict
    new_checkpoint = {
        "config": {
            "dim": target_dim,
            "depth": target_depth,
            "num_fluid_states": target_num_fluid_states,
            "vocab_size": orig_vocab_size
        },
        "model_state_dict": new_model_sd,
        "translation_head_state_dict": new_trans_sd if orig_trans_sd is not None else None
    }
    
    torch.save(new_checkpoint, output_path)
    print(f"Scaled checkpoint successfully saved to {output_path}!")

if __name__ == "__main__":
    src = "henri_core_final.pt"
    dst = "henri_core_final.pt"
    if len(sys.argv) > 1:
        src = sys.argv[1]
    if len(sys.argv) > 2:
        dst = sys.argv[2]
    scale_checkpoint(src, dst)
