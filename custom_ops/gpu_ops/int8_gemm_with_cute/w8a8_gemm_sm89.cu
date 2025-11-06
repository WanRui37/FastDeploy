// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/algorithm/gemm.hpp>
#include "cute/algorithm/copy.hpp"
#include <cute/atom/mma_traits.hpp>

// SM89架构的头文件
#include "mma_sm89.hpp"
#include "mma_traits_sm89.hpp"

// 导入正确的GEMM配置
#include "kernel_traits_gemm.h"

#include "paddle/extension.h"
#include "helper.h"

using namespace cute;

// Shared storage structure for GEMM
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct GemmSharedStorage {
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> smem_A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> smem_B;
};

// 纯粹的GEMM实现，不包含Flash Attention相关代码
template <typename KernelTraits>
__global__ void w8a8_gemm_kernel(
    const typename KernelTraits::ElementA_* A,  // Activation matrix (M x K)
    const typename KernelTraits::ElementB_* B,  // Weight matrix (K x N)  
    typename KernelTraits::ElementC_* C,        // Output matrix (M x N)
    int M, int N, int K) {
    
    using ElementA = typename KernelTraits::ElementA_;
    using ElementB = typename KernelTraits::ElementB_;
    using ElementC = typename KernelTraits::ElementC_;
    using ElementAccum = typename KernelTraits::ElementAccum;
    
    using TiledMma = typename KernelTraits::TiledMma;
    using SmemLayoutA = typename KernelTraits::SmemLayoutA;
    using SmemLayoutB = typename KernelTraits::SmemLayoutB;
    using GmemTiledCopyA = typename KernelTraits::GmemTiledCopyA;
    using GmemTiledCopyB = typename KernelTraits::GmemTiledCopyB;
    
    static constexpr int kBlockM = KernelTraits::kBlockM;
    static constexpr int kBlockN = KernelTraits::kBlockN;
    static constexpr int kBlockK = KernelTraits::kBlockK;
    static constexpr int kNumThreads = KernelTraits::kNumThreads;

  // Preconditions
    static_assert(rank(TiledMma{}) == 3, "TiledMma should have rank 3");
    
    // Problem shape
    auto problem_shape = make_shape(Int<M>{}, Int<N>{}, Int<K>{});
    auto cta_tiler = make_shape(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});

  // Represent the full tensors
    Tensor gA = make_tensor(make_gmem_ptr(A), make_shape(M, K), make_stride(K, _1{}));
    Tensor gB = make_tensor(make_gmem_ptr(B), make_shape(K, N), make_stride(N, _1{}));
    Tensor gC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, _1{}));

  // Get the appropriate blocks for this thread block
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA_tile = local_tile(gA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB_tile = local_tile(gB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC_tile = local_tile(gC, cta_tiler, cta_coord, Step<_1, _1, X>{});

  // Shared memory buffers
  extern __shared__ char shared_memory[];
    using SharedStorage = GemmSharedStorage<ElementA, ElementB, SmemLayoutA, SmemLayoutB>;
    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);
    
    Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_A.begin()), SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_B.begin()), SmemLayoutB{});

    // Tiled copies
    GmemTiledCopyA gmem_tiled_copy_A{};
    GmemTiledCopyB gmem_tiled_copy_B{};
    
    auto thr_copy_A = gmem_tiled_copy_A.get_slice(threadIdx.x);
    auto thr_copy_B = gmem_tiled_copy_B.get_slice(threadIdx.x);
    
    // Partition global and shared memory
    Tensor tAgA = thr_copy_A.partition_S(gA_tile);
    Tensor tAsA = thr_copy_A.partition_D(sA);
    Tensor tBgB = thr_copy_B.partition_S(gB_tile);
    Tensor tBsB = thr_copy_B.partition_D(sB);
    
    // Thread-level MMA
    TiledMma tiled_mma{};
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCrA = thr_mma.partition_fragment_A(sA);
    auto tCrB = thr_mma.partition_fragment_B(sB);
    auto tCcC = thr_mma.partition_fragment_C(gC_tile);
    
    // Clear accumulator
    clear(tCcC);

    // Main GEMM loop over K dimension
    int k_tile_count = (K + kBlockK - 1) / kBlockK;
    
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        // Load A and B tiles to shared memory
        if (k_tile < k_tile_count - 1) {
            // Prefetch next tile
            cute::copy(gmem_tiled_copy_A, tAgA(_, _, k_tile + 1), tAsA(_, _, k_tile + 1));
            cute::copy(gmem_tiled_copy_B, tBgB(_, _, k_tile + 1), tBsB(_, _, k_tile + 1));
        }
        
        // Wait for current tile to be loaded
        if (k_tile > 0) {
            cute::cp_async_wait<0>();
    __syncthreads();
        }
        
        // Load current tile to registers and perform MMA
        cute::copy(gmem_tiled_copy_A, tAgA(_, _, k_tile), tAsA(_, _, k_tile));
        cute::copy(gmem_tiled_copy_B, tBgB(_, _, k_tile), tBsB(_, _, k_tile));
        
        cute::cp_async_wait<0>();
        __syncthreads();
        
        // Load from shared memory to registers and perform GEMM
        cute::copy(tiled_mma, sA(_, _, k_tile % 2), tCrA);
        cute::copy(tiled_mma, sB(_, _, k_tile % 2), tCrB);
        cute::gemm(tiled_mma, tCrA, tCrB, tCcC);
      }
    
    // Store result to global memory
    cute::copy(tiled_mma, tCcC, gC_tile);
}

// Host-side GEMM function
template <typename KernelTraits>
void w8a8_gemm_impl(
    const typename KernelTraits::ElementA_* A,
    const typename KernelTraits::ElementB_* B,
    typename KernelTraits::ElementC_* C,
    int M, int N, int K,
    cudaStream_t stream = 0) {
    
    static constexpr int kBlockM = KernelTraits::kBlockM;
    static constexpr int kBlockN = KernelTraits::kBlockN;
    static constexpr int kBlockK = KernelTraits::kBlockK;
    static constexpr int kNumThreads = KernelTraits::kNumThreads;
    static constexpr int kSmemSize = KernelTraits::kTotalSmemSize;
    
    // Calculate grid dimensions
    dim3 grid((M + kBlockM - 1) / kBlockM, (N + kBlockN - 1) / kBlockN, 1);
    dim3 block(kNumThreads, 1, 1);

    // Launch kernel
    w8a8_gemm_kernel<KernelTraits><<<grid, block, kSmemSize, stream>>>(A, B, C, M, N, K);
}

// Paddle operator implementation for W8A8 GEMM
std::vector<paddle::Tensor> W8A8GemmCute(
    const paddle::Tensor& activations,    // int8 activations (M x K)
    const paddle::Tensor& weights,        // int8 weights (K x N)
    const paddle::Tensor& scales_a,       // float scales for activations (M,)
    const paddle::Tensor& scales_b,       // float scales for weights (N,)
    int M, int N, int K) {
    
    // Validate input dimensions
    PADDLE_ENFORCE_EQ(activations.dtype(), paddle::DataType::INT8,
                     "Activations must be int8");
    PADDLE_ENFORCE_EQ(weights.dtype(), paddle::DataType::INT8,
                     "Weights must be int8");
    PADDLE_ENFORCE_EQ(scales_a.dtype(), paddle::DataType::FLOAT32,
                     "Scales A must be float32");
    PADDLE_ENFORCE_EQ(scales_b.dtype(), paddle::DataType::FLOAT32,
                     "Scales B must be float32");
    
    // Create intermediate int32 output
    paddle::Tensor output_int32 = paddle::empty({M, N}, paddle::DataType::INT32, activations.place());
    
    // Perform W8A8 GEMM (int8 x int8 -> int32)
    using KernelTraits = W8A8Gemm_128x128x32_4warps;
    
    w8a8_gemm_impl<KernelTraits>(
        reinterpret_cast<const int8_t*>(activations.data()),
        reinterpret_cast<const int8_t*>(weights.data()),
        reinterpret_cast<int32_t*>(output_int32.data()),
        M, N, K,
        activations.stream());
    
    // Apply scaling (int32 -> float32 with scaling)
    paddle::Tensor output_fp32 = paddle::empty({M, N}, paddle::DataType::FLOAT32, activations.place());
    
    // Simple scaling kernel (in real implementation, this would be optimized)
    int num_elements = M * N;
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    
    auto* scales_a_data = scales_a.data<float>();
    auto* scales_b_data = scales_b.data<float>();
    auto* int32_data = output_int32.data<int32_t>();
    auto* fp32_data = output_fp32.data<float>();
    
    // Launch scaling kernel
    auto scaling_kernel = [] __global__ (
        const int32_t* input, float* output, const float* scales_a, const float* scales_b,
        int M, int N, int num_elements) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_elements) return;
        
        int row = idx / N;
        int col = idx % N;
        
        float scale = scales_a[row] * scales_b[col];
        output[idx] = static_cast<float>(input[idx]) * scale;
    };
    
    scaling_kernel<<<num_blocks, block_size, 0, activations.stream()>>>(
        int32_data, fp32_data, scales_a_data, scales_b_data, M, N, num_elements);
    
    return {output_fp32};
}

// Shape inference function
std::vector<std::vector<int64_t>> W8A8GemmShape(
    const std::vector<int64_t>& activations,
    const std::vector<int64_t>& weights,
    const std::vector<int64_t>& scales_a,
    const std::vector<int64_t>& scales_b) {
    
    PADDLE_ENFORCE_EQ(
        activations.size(), 2,
        phi::errors::InvalidArgument("Activations should be 2D"));
    PADDLE_ENFORCE_EQ(
        weights.size(), 2,
        phi::errors::InvalidArgument("Weights should be 2D"));
    
    int64_t m = activations[0];
    int64_t n = weights[1];
    
    return {{m, n}};
}

// Dtype inference function
std::vector<paddle::DataType> W8A8GemmDtype(
    const paddle::DataType& activations,
    const paddle::DataType& weights,
    const paddle::DataType& scales_a,
    const paddle::DataType& scales_b,
    const std::string& out_dtype) {
    
    paddle::DataType output_dtype;
    
    if (out_dtype == "bfloat16") {
        output_dtype = paddle::DataType::BFLOAT16;
    } else if (out_dtype == "float16") {
        output_dtype = paddle::DataType::FLOAT16;
    } else if (out_dtype == "float32") {
        output_dtype = paddle::DataType::FLOAT32;
    } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Unsupported output dtype: %s", out_dtype.c_str()));
    }
    
    return {output_dtype};
}

// PD_KERNEL binding
PD_BUILD_STATIC_OP(w8a8_gemm)
    .Inputs({"activations", "weights", "scales_a", "scales_b"})
    .Outputs({"outputs"})
    .Attrs({"out_dtype: std::string"})
    .SetKernelFn(PD_KERNEL(W8A8GemmCute))
    .SetInferShapeFn(PD_INFER_SHAPE(W8A8GemmShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(W8A8GemmDtype));