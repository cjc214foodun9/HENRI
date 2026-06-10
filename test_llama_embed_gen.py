import llama_cpp
try:
    model = llama_cpp.Llama(
        model_path="Huihui-gemma-4-12B-it-abliterated.Q8_0.gguf",
        n_ctx=8192,
        n_batch=512,
        n_gpu_layers=-1,
        flash_attn=True,
        embedding=True,
        verbose=True
    )
    print("Success loading model with embedding=True")
    
    # Test generation
    res_gen = model.create_completion("Hello, how are you?", max_tokens=10)
    print("Generation successful:", res_gen["choices"][0]["text"].strip())
    
    # Test embedding
    res_emb = model.create_embedding("Hello, how are you?")
    print("Embedding successful, dim:", len(res_emb["data"][0]["embedding"]))
except Exception as e:
    print(f"Error: {e}")
