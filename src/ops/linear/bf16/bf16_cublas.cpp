#include "ninfer/ops/linear.h"

#include "core/device.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <sstream>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) { return; }
    std::ostringstream message;
    message << operation << " failed with cuBLAS status " << static_cast<int>(status);
    throw std::runtime_error(message.str());
}

} // namespace

struct Bf16GemmContext::Impl {
    cublasHandle_t handle      = nullptr;
    cudaStream_t stream        = nullptr;
    void* workspace            = nullptr;
    std::size_t workspace_bytes = 0;
};

Bf16GemmContext::Bf16GemmContext(cudaStream_t stream, std::size_t workspace_bytes)
    : impl_(std::make_unique<Impl>()) {
    if (stream == nullptr) { throw std::invalid_argument("BF16 GEMM context requires a stream"); }
    check_cublas(cublasCreate(&impl_->handle), "cublasCreate");
    try {
        check_cublas(cublasSetStream(impl_->handle, stream), "cublasSetStream");
        check_cublas(cublasSetMathMode(impl_->handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
        if (workspace_bytes > 0) {
            CUDA_CHECK(cudaMalloc(&impl_->workspace, workspace_bytes));
            impl_->workspace_bytes = workspace_bytes;
        }
        check_cublas(cublasSetWorkspace(impl_->handle, impl_->workspace, impl_->workspace_bytes),
                     "cublasSetWorkspace");
        impl_->stream = stream;
    } catch (...) {
        if (impl_->workspace != nullptr) { (void)cudaFree(impl_->workspace); }
        (void)cublasDestroy(impl_->handle);
        throw;
    }
}

Bf16GemmContext::~Bf16GemmContext() {
    if (!impl_) { return; }
    if (impl_->workspace != nullptr) { (void)cudaFree(impl_->workspace); }
    if (impl_->handle != nullptr) { (void)cublasDestroy(impl_->handle); }
}

void Bf16GemmContext::launch(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (stream != impl_->stream) {
        throw std::invalid_argument("BF16 GEMM context used with a different stream");
    }
    const float alpha             = 1.0F;
    const float beta              = 0.0F;
    const cudaDataType_t out_type = out.dtype == DType::FP32 ? CUDA_R_32F : CUDA_R_16BF;
    check_cublas(cublasGemmEx(impl_->handle, CUBLAS_OP_T, CUBLAS_OP_N, w.n, x.ne[1], w.k, &alpha,
                              w.qdata, CUDA_R_16BF, w.k, x.data, CUDA_R_16BF, w.k, &beta, out.data,
                              out_type, w.n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "cublasGemmEx");
}

} // namespace ninfer::ops
