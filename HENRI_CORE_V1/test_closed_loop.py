import torch
from closed_loop_engine import ClosedLoopThermodynamicEngine

class MockPipeline:
    def __init__(self):
        self.call_count = 0
        
    def generate_compliant_sequence(self, active_wavefront, target_axiom=None, max_len=100):
        self.call_count += 1
        if self.call_count == 1:
            # First attempt: complete failure (all zeros)
            return "<|python_begin|>\nimport torch\nans = torch.zeros(10, 10)\n<|python_end|>"
        elif self.call_count == 2:
            # Second attempt: partial failure (some ones)
            return "<|python_begin|>\nimport torch\nans = torch.ones(10, 10) * 0.5\n<|python_end|>"
        else:
            # Third attempt: absolute topological resonance!
            return "<|python_begin|>\nimport torch\nans = torch.ones(10, 10)\n<|python_end|>"

def run_test():
    engine = ClosedLoopThermodynamicEngine(vocab_map={"a": 1})
    engine.pipeline = MockPipeline()
    
    # 10x10 target grid of all 1s
    target_grid = torch.ones(10, 10, dtype=torch.float32)
    initial_wavefront = torch.randn(4096, dtype=torch.bfloat16)
    
    success, best_code, cycles = engine.execute_viscoelastic_creep(
        initial_wavefront=initial_wavefront,
        target_grid=target_grid,
        task_context={}
    )
    
    print(f"\nFinal Result: Success={success}, Cycles={cycles}")
    print(f"Best Code:\n{best_code}")

if __name__ == "__main__":
    run_test()
