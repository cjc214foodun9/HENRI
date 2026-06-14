import os
import sys
import re
import argparse
import time
from openai import OpenAI

# Reconfigure console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

class EpistemicDataExtrapolator:
    def __init__(self, api_key, raw_dir="c:/Users/chan/Desktop/HENRI 7B SWARM/HENRI/archive/raw_sources", output_dir="c:/Users/chan/Desktop/HENRI 7B SWARM/HENRI/archive/extrapolated_sources"):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=api_key
        )
        self.model_name = "deepseek-chat"

    def extrapolate_file(self, file_path, variations_count=5):
        print(f"\n[EXTRAPOLATOR] Loading seed: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            seed_content = f.read()

        file_basename = os.path.basename(file_path).split(".")[0]
        
        system_prompt = (
            "You are an expert scientific and architectural data synthesizer. "
            "Your task is to take a seed technical document and extrapolate it by generating a highly detailed, "
            "comprehensive expansion of its mathematical derivations, specific problem scenarios, edge cases, "
            "or code implementations. "
            "Write in a dense, textbook-like style. "
            "Strips away ALL conversational filler, introductions, greetings, and chat remarks. "
            "Output ONLY the raw technical text and equations."
        )

        for i in range(variations_count):
            print(f"  [VARIATION {i+1}/{variations_count}] Generating synthetic expansion...")
            user_content = (
                f"Based on the following seed document, generate a unique, non-overlapping technical expansion. "
                f"Introduce different boundary conditions, edge cases, or specific code applications:\n\n{seed_content}"
            )
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.7 # High temperature for creative technical diversity
                )
                
                synthetic_text = response.choices[0].message.content.strip()
                
                # Write to extrapolated folder
                out_path = os.path.join(self.output_dir, f"{file_basename}_variation_{i+1}.md")
                with open(out_path, "w", encoding="utf-8") as out_f:
                    out_f.write(f"# Expanded Variation {i+1} for: {file_basename}\n\n" + synthetic_text)
                
                print(f"    [SAVED] {out_path} ({len(synthetic_text)} characters)")
                time.sleep(1) # Small delay to respect rate limits
                
            except Exception as e:
                print(f"    [ERROR] Failed to generate variation {i+1}: {e}")

    def run_pipeline(self, variations_per_file=5):
        if not os.path.exists(self.raw_dir):
            print(f"[FATAL] Raw sources directory not found: {self.raw_dir}")
            return

        files = [os.path.join(self.raw_dir, f) for f in os.listdir(self.raw_dir) if f.endswith(".md") or f.endswith(".txt")]
        print(f"[START EXTRAPOLATION] Expanding {len(files)} files with {variations_per_file} variations each (Total expected files: {len(files) * variations_per_file})...")

        for idx, f in enumerate(files):
            print(f"\n[{idx+1}/{len(files)}] Processing base file...")
            self.extrapolate_file(f, variations_count=variations_per_file)

        print("[FINISHED] Data extrapolation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Epistemic Data Extrapolator")
    parser.add_argument("--variations", type=int, default=5, help="Number of synthetic variations to generate per seed file")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("[FATAL] Please set the DEEPSEEK_API_KEY environment variable.")
        sys.exit(1)

    extrapolator = EpistemicDataExtrapolator(api_key=api_key)
    extrapolator.run_pipeline(variations_per_file=args.variations)
