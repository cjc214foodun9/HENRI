# Pure PyTorch Swarm Registry for HENRI
# All Gemma (12B/26B) and GGUF/llama_cpp components have been purged.

def classify_rigor_level(prompt_text: str) -> int:
    """
    Dynamically adjusts the required mathematical rigor.
    Returns 0 for retrieval/coefficient tasks, 1 for derivations/proofs.
    """
    prompt_lower = prompt_text.lower()
    
    # If the prompt explicitly demands proof/derivation, NEVER drop to Level 0
    level_1_vetoes = ["derive", "prove", "show that", "calculate the full", "demonstrate"]
    if any(veto in prompt_lower for veto in level_1_vetoes):
        return 1
        
    # Heuristics for literature retrieval or coefficient matching
    level_0_triggers = ["what is the fundamental constant", "list the known mass of", "fixed constants"]
    
    if any(trigger in prompt_lower for trigger in level_0_triggers):
        return 0
    return 1 # Default to strict mathematical proof requirement

HENRI_GBNF_GRAMMAR = r"""
root ::= (python-block | amcc-block | yield-block | text-with-code | text-block)

python-block ::= "<|python_begin" heat-tag? "|>" ws python-code "<|python_end|>"
heat-tag ::= ": heat=" [0-9] "." [0-9]+

python-code ::= (python-char)*
python-char ::= [ \t\n!#-~] | "\\" | "\"" | "'"

amcc-block ::= "<|amcc_override_begin|>" content "<|amcc_override_end|>"
yield-block ::= "<|epistemic_yield|>" content

text-with-code ::= content "```python" ws python-code "```" content
text-block ::= content

content ::= (content-char)*
content-char ::= [ \t\n!#-~] | "\\" | "\"" | "'"
ws ::= [ \t\n]*
"""
