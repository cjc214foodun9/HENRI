import os
import json
import re
import sys
import argparse
import numpy as np
from openai import OpenAI

# Reconfigure stdout/stderr for Unicode path handling
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

class EpistemicDataFoundry:
    def __init__(self, api_key, output_dir="./esc_compiled_dataset", raw_dir="c:/Users/chan/Desktop/HENRI 7B SWARM/HENRI/archive/raw_sources"):
        self.output_dir = output_dir
        self.raw_dir = raw_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize client pointing to the official DeepSeek API node
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key
        )
        
        # DeepSeek flagship reasoning/chat model
        self.model_name = "deepseek-chat"  # deepseek-chat matches V3/V4 standard endpoint
        self.target_dim = 4096

    def auto_classify_quadrant(self, file_name, file_content=""):
        name_lower = file_name.lower()
        content_lower = file_content.lower()[:2000] # Check first 2k chars

        # Alpha: PDEs, physics, waves, conformal maps, boundary conditions, Ricci, strouhal, vortex
        alpha_keywords = ["conformal", "pde", "elliptic", "poisson", "wave", "vortex", "ricci", "strouhal", "sin", "cos", "fourier", "dft", "sine"]
        # Beta: AST, compilation, schemas, trait, interface, protocol, code, typings, json, AWS, Stripe
        beta_keywords = ["ast", "estree", "ts-morph", "schema", "interface", "protocol", "trait", "compilation", "type", "api", "graphql", "rust", "typescript"]
        # Gamma: Gestalt, Modulor, visionOS, balance, aesthetic, visual, art, color, layout, design, spatial
        gamma_keywords = ["gestalt", "modulor", "visionos", "balance", "aesthetic", "visual", "art", "color", "layout", "design", "spatial", "brutalism", "weight"]

        # 1. Check filename matches
        for kw in alpha_keywords:
            if kw in name_lower:
                return "alpha"
        for kw in beta_keywords:
            if kw in name_lower:
                return "beta"
        for kw in gamma_keywords:
            if kw in name_lower:
                return "gamma"

        # 2. Check content matches
        alpha_score = sum(1 for kw in alpha_keywords if kw in content_lower)
        beta_score = sum(1 for kw in beta_keywords if kw in content_lower)
        gamma_score = sum(1 for kw in gamma_keywords if kw in content_lower)

        max_score = max(alpha_score, beta_score, gamma_score)
        if max_score > 0:
            if max_score == alpha_score:
                return "alpha"
            elif max_score == beta_score:
                return "beta"
            else:
                return "gamma"

        # Fallback to alpha default
        return "alpha"

    def generate_system_prompt(self, quadrant_type):
        base_prompt = (
            "You are an automated structural data foundry for a thermodynamic topology engine. "
            "Your task is to ingest raw research documents and extract ONLY the explicit mathematical invariants, "
            "abstract syntax tree mappings, type boundaries, or layout coordinates. "
            "Strip away ALL natural language filler, polite introductions, explanations, and conversational text. "
            "Output your response inside a single, strict JSON block matching this schema:\n"
            '{"boundary_type": "string", "invariant_structure": "string"}'
        )
        
        if quadrant_type == "alpha":
            return base_prompt + " Focus on isolating partial differential equations, Fourier shifting parameters, and grid value states."
        elif quadrant_type == "beta":
            return base_prompt + " Focus on isolating compiler AST expressions, interfaces, schema structures, and code syntax blocks."
        elif quadrant_type == "gamma":
            return base_prompt + " Focus on isolating spatial balance scales, Modulor sequences, visual weight, and visual equilibrium metrics."
        else:
            return base_prompt

    def forge_raw_source(self, file_path, quadrant_type=None):
        print(f"\n[FOUNDRY INGESTION] Processing high-density asset: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
        except UnicodeDecodeError:
            # Handle binary files like PDFs by scanning metadata/ascii content or skipping cleanly
            print(f"[FOUNDRY SKIP] Skipping binary/PDF file: {file_path}")
            return None

        if not quadrant_type:
            quadrant_type = self.auto_classify_quadrant(os.path.basename(file_path), raw_content)

        print(f"  Classified as Quadrant: {quadrant_type}")
        system_prompt = self.generate_system_prompt(quadrant_type)
        
        try:
            # Note: The DeepSeek standard chat model (deepseek-chat) is extremely powerful
            # and follows system extraction instructions precisely.
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract structural invariants from this data chunk:\n\n{raw_content}"}
                ],
                temperature=0.1
            )
            
            raw_output = response.choices[0].message.content
            return self._parse_and_vectorize(file_path, raw_output, quadrant_type)
            
        except Exception as e:
            print(f"[FOUNDRY EXCEPTION] Generation failure on file {file_path}: {e}")
            return None

    def _parse_and_vectorize(self, original_path, clean_output, quadrant_type):
        try:
            # Clean markdown JSON wraps
            if "```json" in clean_output:
                clean_output = clean_output.split("```json")[1].split("```")[0].strip()
            elif "```" in clean_output:
                clean_output = clean_output.split("```")[1].split("```")[0].strip()
                
            parsed_data = json.loads(clean_output)
            invariant_str = parsed_data.get("invariant_structure", "")
            
            # Map character structures directly to float integers for the HRRInputLayer
            char_array = [float(ord(char)) for char in invariant_str if ord(char) < 256]
            padded_vector = np.zeros(self.target_dim, dtype=np.float32)
            padded_vector[:min(len(char_array), self.target_dim)] = char_array[:self.target_dim]
            
            file_name = os.path.basename(original_path).split(".")[0]
            export_packet = {
                "metadata": {
                    "id": f"synthetic_{file_name}",
                    "quadrant": f"{quadrant_type}_forged"
                },
                "boundary_type": parsed_data.get("boundary_type", "unclassified"),
                "tensor_data": padded_vector.tolist()
            }
            
            output_file = os.path.join(self.output_dir, f"packet_{file_name}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(export_packet, f, indent=4)
                
            print(f"[FOUNDRY ANCHOR] Successfully compiled and staged tensor packet: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"[PARSING LOCK FAULT] Failed to de-serialize structured payload: {e}")
            return None

    def process_all_sources(self):
        if not os.path.exists(self.raw_dir):
            print(f"[ERROR] Raw sources directory does not exist: {self.raw_dir}")
            return
            
        files = [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir) if f.endswith(".md") or f.endswith(".txt")]
        print(f"[FOUNDRY START] Processing {len(files)} text/markdown files...")
        
        success = 0
        for idx, f in enumerate(files):
            print(f"\n[{idx+1}/{len(files)}] Processing...")
            result = self.forge_raw_source(f)
            if result:
                success += 1
                
        print(f"\n[FOUNDRY FINISHED] Successfully vectorized and compiled {success}/{len(files)} packets.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epistemic Data Foundry Pipeline")
    parser.add_argument("--verify", action="store_true", help="Run in mock/verify mode using mock file")
    parser.add_argument("--all", action="store_true", help="Process all files in the raw_sources folder")
    args = parser.parse_args()

    # Configure API Key securely from environment variable
    api_key = os.environ.get("DEEPSEEK_API_KEY", "sk-your-deepseek-production-token")
    
    if args.verify:
        print("[VERIFY] Starting Mock Verification Setup...")
        foundry = EpistemicDataFoundry(api_key="mock-key")
        mock_dir = "./notebooklm_source_pool"
        os.makedirs(mock_dir, exist_ok=True)
        mock_source = os.path.join(mock_dir, "continuous_pde_notes.txt")
        with open(mock_source, "w", encoding="utf-8") as f:
            f.write("Notes on Navier-Stokes boundary behavior... velocity field v(x,t) behaves at a fixed wall.")
        
        # Test vectorization directly with a mock payload
        mock_output = '{"boundary_type": "navier_stokes_wall", "invariant_structure": "v(x,t) = 0 at wall boundary"}'
        packet = foundry._parse_and_vectorize(mock_source, mock_output, "alpha")
        if packet and os.path.exists(packet):
            print("[VERIFY SUCCESS] Mock execution completed successfully.")
        else:
            print("[VERIFY FAIL] Mock execution failed to write packet.")
            sys.exit(1)
            
    elif args.all:
        if api_key == "sk-your-deepseek-production-token" or not api_key:
            print("[FATAL] Please set the DEEPSEEK_API_KEY environment variable before running in production mode.")
            sys.exit(1)
            
        foundry = EpistemicDataFoundry(api_key=api_key)
        foundry.process_all_sources()
        
    else:
        parser.print_help()
