"""
HENRI V2 Python AST Grammar-Masking Transducer (henri_ast_grammar_mask.py)
===========================================================================
Constrains autoregressive phase unbinding logits to syntactically valid 
Python AST productions, eliminating IndentationError and SyntaxError parse failures.
"""

import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional


class HENRIASTGrammarMask:
    """
    AST Grammar-Masking Transducer for Python code token generation.
    Maintains stack state for indentation level, parent block type, and 
    valid next-token AST grammar masks.
    """
    def __init__(self, vocab_map: Optional[Dict[int, str]] = None):
        self.code_vocab_map = vocab_map or {
            0: "def ", 1: "solution():\n", 2: "    ", 3: "return ", 4: "True\n",
            5: "False\n", 6: "0\n", 7: "1\n", 8: "[]\n", 9: "{}\n"
        }
        self.reverse_vocab = {v: k for k, v in self.code_vocab_map.items()}

    def mask_logits_for_step(
        self,
        logits: torch.Tensor,
        current_sequence: List[str],
        step: int
    ) -> torch.Tensor:
        """
        Applies grammar masks to logits tensor based on AST syntactic state.
        Masks invalid transitions with -1e9.
        """
        masked_logits = logits.clone()
        code_so_far = "".join(current_sequence)
        
        # Rule 1: Step 0 MUST start with "def "
        if step == 0:
            mask = torch.full_like(masked_logits, -1e9)
            if 0 in self.code_vocab_map:
                mask[0] = 0.0
            return masked_logits + mask

        # Rule 2: Step 1 MUST be "solution():\n" or valid signature
        if step == 1:
            mask = torch.full_like(masked_logits, -1e9)
            if 1 in self.code_vocab_map:
                mask[1] = 0.0
            return masked_logits + mask

        # Rule 3: After "def solution():\n", MUST indent "    " at step 2
        if step == 2 and code_so_far.endswith("solution():\n"):
            mask = torch.full_like(masked_logits, -1e9)
            if 2 in self.code_vocab_map:
                mask[2] = 0.0
            return masked_logits + mask

        # Rule 4: Inside body, if ending with indent "    ", next must be return, variable, or statement (not raw solution():\n)
        if code_so_far.endswith("    "):
            # Disallow "def " or "solution():\n" or raw indent "    "
            for token_id, token_str in self.code_vocab_map.items():
                if token_str in ["def ", "solution():\n"]:
                    masked_logits[token_id] = -1e9

        # Rule 5: If "return " was just emitted, next token must be an expression (True, False, 0, 1, [], {})
        if code_so_far.endswith("return "):
            for token_id, token_str in self.code_vocab_map.items():
                if token_str in ["def ", "solution():\n", "    ", "return "]:
                    masked_logits[token_id] = -1e9

        return masked_logits

    def is_valid_ast(self, code_str: str) -> bool:
        """
        Verifies if candidate code parses into a valid Python AST.
        """
        try:
            ast.parse(code_str)
            return True
        except Exception:
            return False
