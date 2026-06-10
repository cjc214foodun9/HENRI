import llama_cpp
try:
    model = llama_cpp.Llama(
        model_path="Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf",
        n_ctx=8192,
        n_batch=512,
        n_gpu_layers=0,
        flash_attn=True,
        verbose=True
    )
    print("Success loading 30 layers with flash_attn")
    model.create_completion("Hello, how are you?", max_tokens=10)
    print("Success generating")
except Exception as e:
    print(f"Error: {e}")
