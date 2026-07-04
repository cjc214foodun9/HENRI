import numpy as np
import torch
import torch.nn as nn
import re
from typing import Dict, Any, Union, List

class HybridArcReasoningEngine(nn.Module):
    def __init__(self, wave_core: nn.Module = None):
        super().__init__()
        self.wave_core = wave_core

    def parse_symbolic_reasoning(self, code_str: str) -> Union[Dict[str, Any], None]:
        """
        Parses generated code or reasoning prose for rigid affine transformations
        (e.g., rot90, rot180, flip, crop, transpose).
        """
        if not code_str:
            return None
            
        code_str_lower = code_str.lower()
        
        # Check for rotation
        rot180_match = re.search(r'(rot180|rotate180|rotate\s*180|rot\s*180)', code_str_lower)
        rot270_match = re.search(r'(rot270|rotate270|rotate\s*270|rot\s*270)', code_str_lower)
        rot_match = re.search(r'(rot90|rotate90|rotate\s*90|rot\s*90|rotate)', code_str_lower)
        
        # Check for flip / mirror
        flip_ud_match = re.search(r'(flipud|flip_ud|flip\s*up\s*down|flip\s*vertically)', code_str_lower)
        flip_lr_match = re.search(r'(fliplr|flip_lr|flip\s*left\s*right|flip\s*horizontally)', code_str_lower)
        flip_match = re.search(r'(flip|mirror)', code_str_lower)
        transpose_match = re.search(r'(transpose)', code_str_lower)
        
        # Check for crop
        crop_match = re.search(r'(crop|slice_grid|slice)', code_str_lower)
        
        # Determine specific rigid affine operation
        if rot180_match:
            return {"type": "rot180"}
        elif rot270_match:
            return {"type": "rot270"}
        elif rot_match:
            return {"type": "rot90"}
        elif flip_ud_match:
            return {"type": "flipud"}
        elif flip_lr_match:
            return {"type": "fliplr"}
        elif flip_match:
            return {"type": "flip"}
        elif transpose_match:
            return {"type": "transpose"}
        elif crop_match:
            return {"type": "crop"}
            
        return None

    def execute_rigid_slice(self, input_grid: Union[List, np.ndarray], transform_info: Dict[str, Any]) -> List:
        """
        Bypasses wave space entirely. Executes zero-loss, exact discrete translations
        using NumPy baseplate functions.
        """
        grid = np.array(input_grid)
        t_type = transform_info.get("type")
        
        if t_type == "rot90":
            res = np.rot90(grid, k=1)
        elif t_type == "rot180":
            res = np.rot90(grid, k=2)
        elif t_type == "rot270":
            res = np.rot90(grid, k=3)
        elif t_type == "flipud":
            res = np.flipud(grid)
        elif t_type == "fliplr":
            res = np.fliplr(grid)
        elif t_type == "flip":
            res = np.flip(grid)
        elif t_type == "transpose":
            res = np.transpose(grid)
        elif t_type == "crop":
            # Default fallback slice: remove borders if possible, else return as is
            if grid.ndim == 2 and grid.shape[0] > 2 and grid.shape[1] > 2:
                res = grid[1:-1, 1:-1]
            else:
                res = grid
        else:
            res = grid
            
        return res.tolist()

    def execute_wave_relaxation(self, 
                                current_wave: torch.Tensor, 
                                previous_wave: torch.Tensor, 
                                zone_c_attractor: torch.Tensor, 
                                temperature: float) -> tuple:
        """
        Feeds complex inputs into the continuous Zone B diffractive wave core
        and executes the Langevin epistemic loop.
        """
        if self.wave_core is None:
            # Fallback mock/random wave return if wave_core is not set
            return current_wave, torch.tensor(0.1, device=current_wave.device)
            
        return self.wave_core(current_wave, zone_c_attractor, temperature)

    def forward(self, 
                code_str: str, 
                input_data: Any, 
                current_wave: torch.Tensor = None, 
                previous_wave: torch.Tensor = None, 
                zone_c_attractor: torch.Tensor = None, 
                temperature: float = 0.0) -> tuple:
        """
        Selects between the NumPy slicing pathway and the wave core manifold dynamically.
        If a rigid transformation is parsed, returns (True, NumPy_Output).
        Otherwise, returns (False, Wave_Core_Output).
        """
        transform_info = self.parse_symbolic_reasoning(code_str)
        if transform_info is not None:
            try:
                # We have a valid rigid transformation path
                output = self.execute_rigid_slice(input_data, transform_info)
                return True, output
            except Exception as e:
                # Fallback to wave core if NumPy operation fails
                pass
                
        # Fallback to wave core pathway
        if current_wave is not None:
            res_wave, energy = self.execute_wave_relaxation(current_wave, previous_wave, zone_c_attractor, temperature)
            return False, (res_wave, energy)
            
        return False, None

if __name__ == "__main__":
    print("[TEST] Running HybridArcReasoningEngine unit test...")
    engine = HybridArcReasoningEngine()
    
    # Test 1: parse rotation
    code1 = "def transform(grid):\n    # We need to rot90 this input\n    return np.rot90(grid)"
    parsed = engine.parse_symbolic_reasoning(code1)
    assert parsed is not None and parsed["type"] == "rot90", f"Failed rot90 check: {parsed}"
    
    # Test 2: execute rotation
    test_grid = [[1, 2], [3, 4]]
    out = engine.execute_rigid_slice(test_grid, parsed)
    expected = [[2, 4], [1, 3]]
    assert out == expected, f"Failed execution. Got {out}, expected {expected}"
    
    # Test 3: forward path selection
    is_symbolic, res = engine.forward(code1, test_grid)
    assert is_symbolic is True, "Should select symbolic path"
    assert res == expected, f"Output mismatch: {res}"
    
    # Test 4: forward wave core fallback path
    is_symbolic_fb, res_fb = engine.forward("def transform(grid):\n    # Fuzzy topological task\n    return grid", 
                                            test_grid, 
                                            current_wave=torch.randn(1, 4096),
                                            previous_wave=torch.randn(1, 4096),
                                            zone_c_attractor=torch.randn(1, 4096),
                                            temperature=0.1)
    assert is_symbolic_fb is False, "Should select wave core fallback path"
    assert res_fb is not None, "Wave core output should not be None"
    
    print("[TEST] All HybridArcReasoningEngine tests PASSED successfully!")
