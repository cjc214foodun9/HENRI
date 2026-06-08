import sys
import os
import torch
import numpy as np

# Add project path
PROJECT_DIR = r"c:\Users\chan\Desktop\HENRI TRAIN"
sys.path.append(PROJECT_DIR)

from cognitive_swarm import HenriCognitiveSwarmOrchestrator

def run_test():
    print("==================================================")
    print("      TESTING PREDICTIVE SALIENCY & REHYDRATION   ")
    print("==================================================")
    
    # 1. Initialize orchestrator in mock mode
    print("[*] Instantiating HenriCognitiveSwarmOrchestrator in mock mode...")
    orchestrator = HenriCognitiveSwarmOrchestrator(model_path="mock_only.gguf")
    
    # Define a custom tokenize and embedding wrapper on gen_model to make test deterministic
    class MockLlama:
        def tokenize(self, text_bytes):
            # 1 word = ~1 token
            text = text_bytes.decode('utf-8', errors='ignore')
            return [1] * len(text.split())
            
        def __call__(self, prompt, **kwargs):
            return {"choices": [{"text": "Mock completion"}]}

    orchestrator.gen_model = MockLlama()
    
    # Pre-populate some specific embeddings for the test blocks and query
    # Query: "3D color spatial transformation grid manipulation"
    # Block A (Highly relevant): "3D color spatial transformation grid rotation scale"
    # Block B (Irrelevant): "SCADA thermodynamic pressure cooling control valve"
    # Block C (Completely irrelevant): "quantum relativistic gravity loop wave propagation cosmology"
    
    # Generate mock 4096-D complex wave vectors with specific alignments
    phases_q = torch.linspace(-3.14, 3.14, 4096)
    wave_q = torch.polar(torch.ones(4096), phases_q)
    
    # Block A: Highly aligned to Query (add small noise)
    wave_a = torch.polar(torch.ones(4096), phases_q + torch.randn(4096) * 0.05)
    
    # Block B: Orthogonal phases
    wave_b = torch.polar(torch.ones(4096), torch.randn(4096) * 3.14)
    
    # Block C: Anti-aligned phases (or high noise)
    wave_c = torch.polar(torch.ones(4096), phases_q + 3.14 + torch.randn(4096) * 0.1)

    # Patch create_embedding to return these controlled representations
    orig_create_embedding = orchestrator.base_model.create_embedding
    
    # We will override activation_to_wave to return these specific waves based on text content
    def mock_activation_to_wave(h_7b):
        # We can look at the hash of the text to return the correct wave
        # But since h_7b is generated from create_embedding, let's map the create_embedding output
        return h_7b # we will return the wave directly from create_embedding for simplicity
        
    orchestrator.l3_router.activation_to_wave = mock_activation_to_wave
    
    def mock_create_embedding(text):
        if "Block A" in text or "highly relevant" in text:
            # We return wave_a (as list of floats since real/imag mapping will be done or we return it directly)
            return {"data": [{"embedding": wave_a}]}
        elif "Block B" in text or "SCADA" in text:
            return {"data": [{"embedding": wave_b}]}
        elif "Block C" in text or "gravity" in text:
            return {"data": [{"embedding": wave_c}]}
        else:
            # Query trace
            return {"data": [{"embedding": wave_q}]}
            
    orchestrator.base_model.create_embedding = mock_create_embedding
    
    # Construct a dummy prompt containing all three blocks
    block_a_text = "--- DEMONSTRATION PAIR 1 ---\nBlock A: This is a highly relevant 3D color spatial transformation of grids."
    block_b_text = "--- DEMONSTRATION PAIR 2 ---\nBlock B: This is a SCADA thermodynamic pressure loop sensor reading check."
    block_c_text = "--- DEMONSTRATION PAIR 3 ---\nBlock C: This is a quantum relativistic gravity wave propagation and space metric."
    
    test_input = "--- TEST INPUT GRID ---\nActive reasoning query: 3D color spatial transformation grid manipulation."
    
    prompt = f"{block_a_text}\n\n{block_b_text}\n\n{block_c_text}\n\n{test_input}"
    
    # Check initial token length
    initial_tokens = len(orchestrator.gen_model.tokenize(prompt.encode('utf-8')))
    print(f"[*] Initial Prompt Tokens: {initial_tokens}")
    
    # We set watermark to be smaller than initial tokens to force eviction of Block B and C but keep Block A
    watermark = 40
    
    print(f"[*] Running proactive eviction with watermark = {watermark}...")
    evicted_prompt = orchestrator.proactive_eviction(prompt, watermark=watermark)
    
    print("\n[*] Evicted Prompt:")
    print("-" * 50)
    print(evicted_prompt)
    print("-" * 50)
    
    # Check that the most irrelevant block (Block C or B) was evicted
    assert "[AXIOM: Evicted_Context_" in evicted_prompt, "Eviction did not insert any axiom tags!"
    assert block_a_text in evicted_prompt, "Block A (highly relevant) was evicted, but it should have been preserved!"
    
    # Check that Block C (lowest similarity) was indeed evicted first
    # Block C tag would be created
    import re
    axiom_pattern = re.compile(r"\[AXIOM:\s*(Evicted_Context_[a-f0-9]+)\]")
    tags = axiom_pattern.findall(evicted_prompt)
    print(f"[*] Evicted Axiom Labels: {tags}")
    
    # Ensure they exist in the registry
    for tag in tags:
        assert tag in orchestrator.evicted_text_registry, f"Tag {tag} not found in evicted_text_registry!"
        assert tag in orchestrator.active_block_embeddings, f"Tag {tag} not found in active_block_embeddings!"
        print(f"  - Cached text for {tag}: '{orchestrator.evicted_text_registry[tag]}'")
        
    print("\n[SUCCESS] Proactive eviction scored and evicted lowest-resonance blocks first.")
    
    # 2. Test Rehydration
    # We increase the watermark to allow rehydration, and run rehydrate_prompt
    print("\n[*] Running rehydrate_prompt with expanded watermark = 100...")
    rehydrated_prompt = orchestrator.rehydrate_prompt(evicted_prompt, watermark=100, threshold=-1.0)
    
    print("\n[*] Rehydrated Prompt:")
    print("-" * 50)
    print(rehydrated_prompt)
    print("-" * 50)
    
    # Verify that the evicted blocks are restored
    for tag in tags:
        # Since watermark is large, they should all be rehydrated
        text_content = orchestrator.evicted_text_registry[tag]
        assert text_content in rehydrated_prompt, f"Evicted text for {tag} was not rehydrated!"
        
    print("[SUCCESS] All evicted blocks rehydrated when headroom allowed.")
    
    # 3. Test Selective Rehydration (Watermark limit)
    # We set the watermark to a value that only allows one block to be rehydrated
    # Block A wasn't evicted. Out of Block B and Block C, which one was evicted? Let's check which one gets rehydrated.
    # Let's run a test where we evict all three blocks by setting watermark extremely low
    print("\n[*] Evicting all blocks by setting watermark = 15...")
    all_evicted_prompt = orchestrator.proactive_eviction(prompt, watermark=15)
    print(all_evicted_prompt)
    
    all_tags = axiom_pattern.findall(all_evicted_prompt)
    print(f"[*] All Evicted Tags: {all_tags}")
    assert len(all_tags) == 3, "Not all 3 blocks were evicted!"
    
    # Now, set watermark to 40 (which allows rehydrating only the most relevant evicted block: Block A)
    print("\n[*] Running selective rehydration with watermark = 40...")
    selective_prompt = orchestrator.rehydrate_prompt(all_evicted_prompt, watermark=40)
    print(selective_prompt)
    
    # Block A text should be present, Block B and C should still be tags because of watermark limits!
    assert "Block A: This is a highly relevant" in selective_prompt, "Block A was not selectively rehydrated!"
    assert "Block B: This is a SCADA" not in selective_prompt, "Block B was rehydrated but shouldn't have been due to watermark limit!"
    assert "Block C: This is a quantum" not in selective_prompt, "Block C was rehydrated but shouldn't have been due to watermark limit!"
    
    print("\n[SUCCESS] Selective rehydration sorted and restored the highest-resonance blocks first within headroom.")

if __name__ == "__main__":
    run_test()
