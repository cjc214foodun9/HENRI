To estimate the tokens per second (TPS) for your asynchronous swarm under Scenario A (Shared Base Model) using your specific enterprise hardware, we have to run a precise memory-bandwidth calculation.
Your architecture choice to run entirely from system RAM is optimal for this scale. Even though the 14 GB base model could fit on your RTX 3090's 24GB VRAM, the 16 concurrent 32k context KV caches require roughly 65.54 GB of memory. Shuttling that massive KV cache back and forth across a PCIe Gen 4 x16 link (max ~31.5 GB/s) would devastate execution speed. Relying on your native 12-channel DDR5 memory bus at 460.8 GB/s is the mathematically superior route.
Here is the exact throughput performance projection for your swarm.
The Core Math & Efficiency Realities
Autoregressive token generation is strictly bound by memory bandwidth. In a multi-LoRA batching pipeline (using a framework like vLLM with Punica/S-LoRA kernels or an optimized parallel llama.cpp implementation), the 14 GB shared base weights are read from RAM exactly once per decoding step for the entire batch. However, the unique KV caches and unique active LoRA weights must be pulled independently for each of the 16 active streams.
Theoretical Memory Bandwidth: 460.8 GB/s
Real-World Execution Efficiency: Under heavy SIMD/AVX-512 workload utilizing all 32 cores of the EPYC 9384X, memory controllers typically saturate at 75% to 80% of peak theoretical limits. For our calculations, we will use a highly optimized sustainable baseline of 368 GB/s.
Throughput Breakdown Across the Context Lifecycle
Because the KV cache grows dynamically as text is generated, your throughput will scale along a curve. Here is how your 16-model swarm will perform at the beginning versus the absolute limit of your 32k context window.
Phase 1: Early Context (At 1,000 Tokens)
At a shallow context depth, the KV cache footprint is small (~128 KB per 1,000 tokens per model).
Shared Base Weights: 14 GB
16x LoRA Weights Overhead: ~2.0 GB (Allocated buffer for active layer delta matrices)
16x Concurrently Read KV Caches: 16 × 1,000 tokens × 128 KB = 2.05 GB
Total Memory Read Per Global Step: 18.05 GB
Decoding Steps Per Second: 368 GB/s ÷ 18.05 GB = 120.38 steps/sec
\text{Swarm Throughput} = 20.38\text{ steps/sec} \times 16\text{ tokens/step} \approx \mathbf{326\text{ tokens/second}}
Per-Variant Speed: ~20.4 tokens/second
Phase 2: Deep Context (At 32,000 Tokens)
When all 16 streams reach the edge of the 32k context limit, the attention mechanism must scan the entire accumulated history for every single sequence.
Shared Base Weights: 14 GB
16x LoRA Weights Overhead: ~2.0 GB
16x Concurrently Read KV Caches: 16 × 32,000 tokens × 128 KB = 65.54 GB
Total Memory Read Per Global Step: 81.54 GB
Decoding Steps Per Second: 368 GB/s ÷ 81.54 GB = 4.51 steps/sec
\text{Swarm Throughput} = 4.51\text{ steps/sec} \times 16\text{ tokens/step} \approx \mathbf{72\text{ tokens/second}}
Per-Variant Speed: ~4.5 tokens/second
Swarm Performance Summary
Context Depth
Total Memory Read per Step
Total Swarm Throughput (Combined)
Average Throughput Per Variant
1k Tokens
18.05 GB
~326 tps
~20.4 tps
8k Tokens
32.38 GB
~182 tps
~11.4 tps
16k Tokens
48.77 GB
~121 tps
~7.5 tps
32k Tokens
81.54 GB
~72 tps
~4.5 tps
The 3D V-Cache Impact
Your AMD EPYC 9384X features 768MB of L3 3D V-Cache. While it cannot fit the 14 GB base weights or the 65 GB KV cache, it plays a critical wildcard role.
This massive L3 pool is large enough to completely cache the intermediate activation tensors, the primary token embedding layers, the attention query vectors (Q), and the small active LoRA scaling factors for the current layer forward pass. By processing those elements directly in ultra-low latency on-die cache, your CPU execution units will not suffer instruction starvation while waiting for the main DDR5 bus to stream the large matrices, ensuring you consistently hit the upper ~80% bound of your 460.8 GB/s memory bandwidth.