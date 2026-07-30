"""
HENRI V2: Code-Cleaning Post-Processor for Egress Transduction
Subsystem: Markdown/Code Block Sanitizer

Strip Markdown code fence wrappers (e.g. ```python ... ```) from raw generated text
to produce pure, syntactically valid Python code for REPL sandbox evaluation.
"""

import re


def clean_generated_code(text: str) -> str:
    """
    Strips Markdown code fences and leading/trailing whitespace from raw generated text.
    """
    if not text:
        return ""

    # Pattern to extract code inside ```python ... ``` or ``` ... ```
    pattern = r"```(?:python)?\s*\n?(.*?)\n?```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no fences are present, strip lone backticks or return clean text
    cleaned = text.replace("```python", "").replace("```", "").strip()
    return cleaned
