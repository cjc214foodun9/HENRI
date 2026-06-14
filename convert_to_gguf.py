import os
import sys
import torch
import numpy as np
import gguf

# Import the custom model architecture
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from henri_core.core import ProprietaryHENRICore

def serialize_henri_to_gguf(pytorch_model_path=None, output_gguf_path="henri_core_7b.gguf", dim=4096, depth=32, num_fluid_states=16):
    print(f"[*] Initializing ProprietaryHENRICore model (7B architecture: dim={dim}, depth={depth}, experts={num_fluid_states})...")
    
    # 1. Instantiate the model
    model = ProprietaryHENRICore(dim=dim, depth=depth, num_fluid_states=num_fluid_states)
    
    # 2. Load PyTorch weights if provided
    if pytorch_model_path and os.path.exists(pytorch_model_path):
        print(f"[+] Loading PyTorch weights from: {pytorch_model_path}")
        model.load_state_dict(torch.load(pytorch_model_path, map_location="cpu"))
    else:
        print("[WARN] No PyTorch weight file provided. Serializing model with current initialization weights.")

    # 3. Create GGUFWriter
    print(f"[*] Initializing GGUFWriter for target: {output_gguf_path}")
    writer = gguf.GGUFWriter(output_gguf_path, "henri-core")
    
    # 4. Add custom model metadata
    writer.add_uint32("henri.dim", dim)
    writer.add_uint32("henri.depth", depth)
    writer.add_uint32("henri.num_fluid_states", num_fluid_states)
    writer.add_uint32("henri.target_dim", dim)
    writer.add_string("henri.architecture", "ThermodynamicMoE")
    
    # 5. Extract and add weights
    state_dict = model.state_dict()
    print(f"[*] Extracting {len(state_dict)} tensor layers...")
    
    for layer_name, tensor in state_dict.items():
        arr = tensor.cpu().numpy()
        
        # GGUF requires float32 or float16 for tensor data (convert double to float32)
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
            
        print(f"  [TENSOR] Writing {layer_name} | Shape: {arr.shape} | Dtype: {arr.dtype}")
        writer.add_tensor(layer_name, arr)
        
    # 6. Write GGUF file to disk
    print("[*] Writing GGUF binary data...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"[SUCCESS] Model successfully compiled and contained in GGUF file: {output_gguf_path}")

if __name__ == "__main__":
    # Example usage:
    # python convert_to_gguf.py --weights henri_weights.pt --output henri_core_7b.gguf
    import argparse
    parser = argparse.ArgumentParser(description="Convert ProprietaryHENRICore PyTorch weights to GGUF")
    parser.add_argument("--weights", type=str, default="", help="Path to PyTorch weights .pt file (if omitted, uses initialized weights)")
    parser.add_argument("--output", type=str, default="henri_core_7b.gguf", help="Output path for the compiled GGUF file")
    parser.add_argument("--dim", type=int, default=4096, help="Embedding dimension")
    parser.add_argument("--depth", type=int, default=32, help="Number of model layers")
    parser.add_argument("--experts", type=int, default=16, help="Number of fluid states / experts")
    args = parser.parse_args()
    
    serialize_henri_to_gguf(pytorch_model_path=args.weights, output_gguf_path=args.output, dim=args.dim, depth=args.depth, num_fluid_states=args.experts)
