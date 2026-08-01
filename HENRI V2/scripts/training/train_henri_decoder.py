"""
Project HENRI V2: Egress Unbinder Neural Training Engine (train_henri_decoder.py)
==================================================================================
Trains HENRINeuralEgressUnbinder (D=65,536 -> d_hidden=2048 -> |V|=32,000) on 
qFHRR wave-token pairs under AdamW optimization, Bingham Plastic yield stress,
and Riemannian Cholesky Stiefel Retractions on CUDA hardware.
"""

import os
import sys
import time
import math
import argparse
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(repo_path)
for p in [repo_path, parent_path, os.path.join(parent_path, "scripts")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from henri_decoder import HENRINeuralEgressUnbinder, HENRIUnifiedEgressTransducer
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec


def generate_qfhrr_training_dataset(codec: qFHRREpistemicCodec, num_samples: int = 1024, device: str = "cuda"):
    """
    Generates synthetic and authentic qFHRR wave-token dataset pairs (W_input, y_target)
    for vocabulary token unbinding and code signature transduction.
    """
    print(f"[DATASET] Synthesizing {num_samples} qFHRR wave-token training pairs on {device.upper()}...")
    
    # Target vocabulary tokens and code signature strings
    tokens_text = [
        "def", "return", "import", "math", "List", "float", "int", "str", "bool",
        "for", "in", "if", "else", "while", "class", "True", "False", "None",
        "solution", "has_close_elements", "truncate_number", "is_palindrome",
        "count_distinct_characters", "sort_third", "unique", "fizz_buzz",
        "Option A", "Option B", "Option C", "Option D",
        "Rotate90", "FlipHorizontal", "ColorPermute", "ContourFill", "GravityDrop"
    ]
    
    dataset_waves = []
    dataset_target_ids = []
    
    vocab_size = 32000
    
    for i in range(num_samples):
        text_target = tokens_text[i % len(tokens_text)]
        # Assign deterministic token ID in [0, vocab_size-1]
        target_id = hash(text_target) % vocab_size
        
        # Ingress Transduction -> Unit wave state on S^{D-1}
        text_wave = codec.encode_text(text_target).to(device).to(torch.float32)
        noise = torch.randn_like(text_wave) * 0.05
        noisy_wave = F.normalize(text_wave + noise, p=2, dim=-1)
        
        dataset_waves.append(noisy_wave)
        dataset_target_ids.append(target_id)
        
    waves_tensor = torch.stack(dataset_waves).to(device)
    targets_tensor = torch.tensor(dataset_target_ids, device=device, dtype=torch.long)
    
    return waves_tensor, targets_tensor


def train_decoder_mission(epochs: int = 20, batch_size: int = 32, lr: float = 1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("========================================================================")
    print("  HENRI V2: NEURAL EGRESS DECODER TRAINING MISSION")
    print("========================================================================")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"Execution Device  : {device.upper()}")
    if device == "cuda":
        print(f"GPU Model Name    : {torch.cuda.get_device_name(0)}")
        print(f"Initial Alloc VRAM: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
        torch.cuda.reset_peak_memory_stats(0)

    codec = qFHRREpistemicCodec(d_model=65536, device=device)
    unbinder = HENRINeuralEgressUnbinder(d_model=65536, d_hidden=2048, vocab_size=32000, device=device)
    
    optimizer = torch.optim.AdamW(unbinder.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    num_samples = 1024
    inputs, targets = generate_qfhrr_training_dataset(codec, num_samples=num_samples, device=device)
    
    num_batches = math.ceil(num_samples / batch_size)
    print(f"[TRAINING] Starting {epochs} Epochs | Batch Size: {batch_size} | Batches/Epoch: {num_batches}\n")
    
    t_start = time.perf_counter()
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct_tokens = 0
        total_tokens = 0
        
        unbinder.train()
        permutation = torch.randperm(num_samples, device=device)
        
        for b in range(num_batches):
            batch_indices = permutation[b * batch_size : (b + 1) * batch_size]
            b_inputs = inputs[batch_indices]
            b_targets = targets[batch_indices]
            
            optimizer.zero_grad()
            logits = unbinder(b_inputs)
            loss = criterion(logits, b_targets)
            loss.backward()
            
            # Bingham Plastic Yield Stress Check & Cholesky Retraction
            with torch.no_grad():
                grad_down = unbinder.down_proj.weight.grad
                if grad_down is not None:
                    grad_norm = torch.norm(grad_down)
                    if grad_norm > 0.05:
                        # Riemannian Cholesky Retraction on W_down
                        v_weight = unbinder.down_proj.weight
                        v_vt = torch.matmul(v_weight, v_weight.T) + 1e-6 * torch.eye(unbinder.d_hidden, device=device)
                        l_inv = torch.linalg.inv(torch.linalg.cholesky(v_vt))
                        unbinder.down_proj.weight.copy_(torch.matmul(l_inv, v_weight))
            
            optimizer.step()
            
            epoch_loss += loss.item() * len(b_targets)
            preds = torch.argmax(logits, dim=-1)
            correct_tokens += int((preds == b_targets).sum().item())
            total_tokens += len(b_targets)
            
        avg_loss = epoch_loss / total_tokens
        acc = (correct_tokens / total_tokens) * 100.0
        
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"[Epoch {epoch:02d}/{epochs:02d}] Loss: {avg_loss:.6f} | Unbinding Token Accuracy: {acc:6.2f}%")
            
    total_sec = time.perf_counter() - t_start
    peak_vram = torch.cuda.max_memory_allocated(0) / 1e9 if device == "cuda" else 0.0
    
    # Save Model Checkpoint
    models_dir = os.path.join(repo_path, "models")
    os.makedirs(models_dir, exist_ok=True)
    ckpt_path = os.path.join(models_dir, "henri_decoder_checkpoint.pt")
    torch.save({
        "model_state_dict": unbinder.state_dict(),
        "d_model": 65536,
        "d_hidden": 2048,
        "vocab_size": 32000,
        "final_loss": avg_loss,
        "accuracy": acc
    }, ckpt_path)
    print(f"\n[CHECKPOINT] Saved trained decoder weights to: {ckpt_path}")
    
    print("\n========================================================================")
    print("                DECODER TRAINING MISSION SUMMARY")
    print("========================================================================")
    print(f"Total Epochs Completed            : {epochs}")
    print(f"Final Cross-Entropy Loss          : {avg_loss:.6f}")
    print(f"Final Token Unbinding Accuracy    : {acc:.2f} %")
    print(f"Total Training Duration           : {total_sec:.2f} seconds")
    print(f"Peak VRAM Allocated               : {peak_vram:.4f} GB")
    print(f"Target GPU Model Name             : {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")
    print("========================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HENRI V2 Egress Decoder Head")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_decoder_mission(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
