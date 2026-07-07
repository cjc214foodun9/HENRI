import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Adjust import path for HolographicVectorLifter
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from holographic_vector_lifter import HolographicVectorLifter

def grid_to_string(grid):
    return "\n".join([" ".join(map(str, row)) for row in grid])

def main():
    print("=========================================================================")
    print("      PROJECT HENRI: ARC AGI 2 DISTILLATION SPRINT (INFERENCE/HARVEST)   ")
    print("=========================================================================")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Target Architecture: {device}")
    
    vocab_size = 32000
    dim = 4096
    
    print("[*] Initializing LLaMA Tokenizer and Holographic Vector Lifter...")
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    lifter = HolographicVectorLifter(vocab_size=vocab_size, dim=dim).to(device)
    
    # Paths assuming script is run from HENRI_CORE_V1
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    training_txt_path = os.path.join(base_dir, "ARC-AGI-2-main", "data", "training.txt")
    training_data_dir = os.path.join(base_dir, "ARC-AGI-2-main", "data", "training")
    
    if not os.path.exists(training_txt_path):
        print(f"[!] Error: Cannot find {training_txt_path}")
        return
        
    with open(training_txt_path, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
        
    print(f"[*] Found {len(task_ids)} ARC tasks in training.txt")
    
    coherent_vectors = []
    
    for idx, task_id in enumerate(task_ids):
        json_path = os.path.join(training_data_dir, f"{task_id}.json")
        if not os.path.exists(json_path):
            continue
            
        with open(json_path, 'r') as f:
            task_data = json.load(f)
            
        # We harvest both train and test outputs to act as Zone C attractors
        for split in ['train', 'test']:
            if split not in task_data:
                continue
                
            for example in task_data[split]:
                output_grid = example.get('output')
                if not output_grid:
                    continue
                    
                # Distill the output grid into a string semantic layout
                output_str = f"<|arc_output|>\n{grid_to_string(output_grid)}\n<|arc_end|>"
                
                tokens = tokenizer.encode(output_str, add_special_tokens=False)
                tokens = [min(t, vocab_size - 1) for t in tokens]
                
                # Split into chunks if too long (though most grids are small)
                seq_len = 256
                for i in range(0, max(1, len(tokens) - seq_len + 1), seq_len):
                    chunk = tokens[i : i + seq_len]
                    token_tensor = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
                    
                    with torch.no_grad():
                        phasors = lifter(token_tensor)
                        bound_wavefront = torch.prod(phasors, dim=1).squeeze(0)
                        normed = F.normalize(bound_wavefront, p=2, dim=-1)
                        
                        # Store pure real shadow for ADMA (simulates BTO threshold)
                        coherent_vectors.append(normed.real.float().cpu())
                
        if (idx + 1) % 50 == 0:
            print(f"[*] Processed {idx + 1}/{len(task_ids)} tasks... Harvested {len(coherent_vectors)} coherent vectors.")

    print(f"[*] Harvesting complete. Total coherent vectors harvested: {len(coherent_vectors)}")
    
    if coherent_vectors:
        tensor_db = torch.stack(coherent_vectors) # [N, 4096]
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arc_coherent_vectors.pt")
        torch.save(tensor_db, out_path)
        print(f"[*] Successfully serialized ARC Holographic Attractors to {out_path} with shape {tensor_db.shape}")

if __name__ == "__main__":
    main()
