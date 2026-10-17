#include "ops/launcher/gemma4_vision_attention.h"

#include "core/device.h"
#include "ops/kernel/gemma4_vision_attention.cuh"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) { return; }
    std::ostringstream message;
    message << operation << " failed with cuBLAS status " << static_cast<int>(status);
    throw std::runtime_error(message.str());
}

cublasHandle_t vision_handle() {
    static cublasHandle_t handle = [] {
        cublasHandle_t created = nullptr;
        check_cublas(cublasCreate(&created), "cublasCreate");
        check_cublas(cublasSetMathMode(created, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
        void* workspace = nullptr;
        CUDA_CHECK(cudaMalloc(&workspace, 64ULL << 20));
        check_cublas(cublasSetWorkspace(created, workspace, 64ULL << 20), "cublasSetWorkspace");
        return created;
    }();
    return handle;
}

} // namespace

void gemma4_vision_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor* mask, Tensor& scores, Tensor& weights, Tensor& out,
                                    cudaStream_t stream) {
    const int tokens = q.ne[2];
    const int dim    = kGemma4VisionHeadDim;
    const int heads  = kGemma4VisionHeads;
    const int lda    = dim * heads;
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    auto* handle      = vision_handle();
    check_cublas(cublasSetStream(handle, stream), "cublasSetStream");
    check_cublas(
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, tokens, tokens, dim, &alpha,
                                   q.data, CUDA_R_16BF, lda, dim, k.data, CUDA_R_16BF, lda, dim,
                                   &beta, scores.data, CUDA_R_32F, tokens,
                                   static_cast<long long>(tokens) * tokens, heads,
                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmStridedBatchedEx QK");
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(heads));
    gemma4_vision_softmax_kernel<<<grid, 32, 0, stream>>>(
        static_cast<const float*>(scores.data), static_cast<__nv_bfloat16*>(weights.data),
        mask == nullptr ? nullptr : static_cast<const std::int32_t*>(mask->data), tokens);
    CUDA_CHECK(cudaGetLastError());
    check_cublas(
        cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, dim, tokens, tokens, &alpha,
                                   v.data, CUDA_R_16BF, lda, dim, weights.data, CUDA_R_16BF, tokens,
                                   static_cast<long long>(tokens) * tokens, &beta, out.data,
                                   CUDA_R_16BF, lda, dim, heads, CUBLAS_COMPUTE_32F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmStridedBatchedEx PV");
}

} // namespace ninfer::ops::detail
