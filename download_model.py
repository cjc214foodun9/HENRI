import sys
from huggingface_hub import hf_hub_download

print("Starting model download...")
try:
    path = hf_hub_download(
        repo_id="mradermacher/Huihui-gemma-4-12B-it-abliterated-GGUF",
        filename="Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf",
        local_dir="/dev/shm",
        local_dir_use_symlinks=False
    )
    print(f"Success! Model downloaded to {path}")
except Exception as e:
    print(f"Error downloading model: {e}")
    sys.exit(1)
