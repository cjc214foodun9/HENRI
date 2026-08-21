#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>

namespace {

__constant__ float kCosineLut[256];
std::once_flag g_lut_once;

void init_lut() {
    float host_lut[256];
    for (int i = 0; i < 256; ++i) {
        const float theta = 6.28318530717958647692f * static_cast<float>(i) / 256.0f;
        host_lut[i] = cosf(theta);
    }
    const cudaError_t err = cudaMemcpyToSymbol(
        kCosineLut, host_lut, sizeof(host_lut), 0, cudaMemcpyHostToDevice);
    TORCH_CHECK(err == cudaSuccess,
                "failed to initialize phase cosine LUT: ", cudaGetErrorString(err));
}

__global__ void sagnac_score_kernel(
    const uint8_t* __restrict__ candidates,
    const uint8_t* __restrict__ codebook,
    float* __restrict__ output,
    int64_t candidate_count,
    int64_t codebook_count,
    int64_t dimension) {
    const int64_t candidate = static_cast<int64_t>(blockIdx.x);
    const int lane = threadIdx.x;
    if (candidate >= candidate_count) {
        return;
    }

    float local_sum = 0.0f;
    const int64_t candidate_base = candidate * dimension;
    for (int64_t n = 0; n < codebook_count; ++n) {
        const int64_t codebook_base = n * dimension;
        for (int64_t d = lane; d < dimension; d += blockDim.x) {
            const unsigned int lhs = static_cast<unsigned int>(candidates[candidate_base + d]);
            const unsigned int rhs = static_cast<unsigned int>(codebook[codebook_base + d]);
            const unsigned int phase_delta = (lhs - rhs) & 255u;
            local_sum += kCosineLut[phase_delta];
        }
    }

    __shared__ float reduction[256];
    reduction[lane] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            reduction[lane] += reduction[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        output[candidate] = reduction[0] /
            static_cast<float>(codebook_count * dimension);
    }
}

torch::Tensor sagnac_mcts_cuda(
    torch::Tensor candidates,
    torch::Tensor codebook) {
    TORCH_CHECK(candidates.is_cuda(), "candidates must be a CUDA tensor");
    TORCH_CHECK(codebook.is_cuda(), "codebook must be a CUDA tensor");
    TORCH_CHECK(candidates.device() == codebook.device(),
                "candidates and codebook must use the same CUDA device");
    TORCH_CHECK(candidates.scalar_type() == torch::kUInt8,
                "candidates must have dtype torch.uint8");
    TORCH_CHECK(codebook.scalar_type() == torch::kUInt8,
                "codebook must have dtype torch.uint8");
    TORCH_CHECK(candidates.dim() == 2 && codebook.dim() == 2,
                "expected candidates [C,D] and codebook [N,D]");
    TORCH_CHECK(candidates.size(1) == codebook.size(1),
                "candidate and codebook dimensions must match");
    TORCH_CHECK(candidates.is_contiguous() && codebook.is_contiguous(),
                "candidate and codebook tensors must be contiguous");
    TORCH_CHECK(candidates.size(0) > 0 && codebook.size(0) > 0 && candidates.size(1) > 0,
                "candidate and codebook tensors must be non-empty");
    TORCH_CHECK(candidates.size(0) <= 2147483647,
                "candidate count exceeds the CUDA grid limit");

    std::call_once(g_lut_once, init_lut);
    auto output = torch::empty(
        {candidates.size(0)},
        candidates.options().dtype(torch::kFloat32));
    const cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(candidates.get_device()).stream();
    sagnac_score_kernel<<<static_cast<unsigned int>(candidates.size(0)), 256, 0, stream>>>(
        candidates.data_ptr<uint8_t>(),
        codebook.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        candidates.size(0),
        codebook.size(0),
        candidates.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

TORCH_LIBRARY(henri, m) {
    m.def("sagnac_mcts(Tensor candidates, Tensor codebook) -> Tensor");
}

TORCH_LIBRARY_IMPL(henri, CUDA, m) {
    m.impl("sagnac_mcts", &sagnac_mcts_cuda);
}
