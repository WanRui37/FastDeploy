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

#include "paddle/extension.h"
#include "helper.h"

using namespace cute;

// Shared storage structure
template <class ElementA, class ElementB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

// Main GEMM device kernel (similar to sgemm_sm80.cu)
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)

  // Partition the copying of A and B tiles across the threads
  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

  // PREFETCH
  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count = size<3>(tAgA);
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
  }

  // Define A/B partitioning and C accumulators
  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V((  shape(tCrC) == take<0,3>(shape(tCgC))));     // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));              // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));              // MMA_N

  // Clear the accumulators
  clear(tCrC);

  // Copy Atom retiling
  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)

  // Current pipe index in smem to read from
  int smem_pipe_read  = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX-1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX-2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  // PIPELINED MAIN LOOP
  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }
  }

  // Epilogue
  axpby(alpha, tCrC, beta, tCgC);
}

// W8A8 GEMM implementation using CUTE
template <class Alpha, class Beta>
void
w8a8_gemm_tn(int m, int n, int k,
             Alpha alpha,
             int8_t const* A, int ldA,
             int8_t const* B, int ldB,
             Beta beta,
             int32_t* C, int ldC,
             cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<128>{};
  auto bN = Int<128>{};
  auto bK = Int< 64>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)
  auto bP = Int<3>{};  // Pipeline

  // Define the smem layouts (static)
  // Swizzles for LDSM and 128b k-major loads
  auto swizzle_atom = composition(Swizzle<3,3,3>{},
                                  Layout<Shape <_8,Shape <_8, _8>>,
                                         Stride<_8,Stride<_1,_64>>>{});

  auto sA = tile_to_shape(swizzle_atom, make_shape(bM,bK,bP));
  auto sB = tile_to_shape(swizzle_atom, make_shape(bN,bK,bP));
  auto sC = make_layout(make_shape(bM, bN));

  // Define the thread layouts (static)
  // 修复：使用适合int8_t的CopyAtom和布局
  TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, int8_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 k-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, int8_t>{},
                                    Layout<Shape<_16,_8>,Stride<_8,_1>>{},  // Thr layout 16x8 k-major
                                    Layout<Shape< _1,_8>>{});               // Val layout  1x8 n-major

  TiledMMA mmaC = make_tiled_mma(SM80_16x8x16_S32S8S8S32_TN{},
                                 Layout<Shape<_2,_2>>{},    // 2x2x1 MMA Atoms
                                 Tile<_32,_32,_16>{});      // 32x32x16 Tiled MMA for LDSM

  // 修复：使用适合int8_t的s2r_atom
  Copy_Atom<SM75_U32x4_LDSM_N, int8_t> s2r_atom_A;
  Copy_Atom<SM75_U32x4_LDSM_N, int8_t> s2r_atom_B;

  int smem_size = int(sizeof(SharedStorage<int8_t, int8_t, decltype(sA), decltype(sB)>));
  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));

  auto kernel_fptr = gemm_device<
    decltype(prob_shape), decltype(cta_tiler),
    int8_t, decltype(dA), decltype(sA), decltype(copyA), decltype(s2r_atom_A),
    int8_t, decltype(dB), decltype(sB), decltype(copyB), decltype(s2r_atom_B),
    int32_t, decltype(dC), decltype(sC), decltype(mmaC),
    decltype(alpha), decltype(beta)>;

  // Set L1 to be SMEM only
  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

  cudaFuncSetAttribute(
    kernel_fptr,
    cudaFuncAttributePreferredSharedMemoryCarveout, 100);

  kernel_fptr<<<dimGrid, dimBlock, smem_size, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA, s2r_atom_A,
       B, dB, sB, copyB, s2r_atom_B,
       C, dC, sC, mmaC,
       alpha, beta);
}

// CUTE-based W8A8 GEMM kernel implementation
template <typename MMA_Traits>
void cute_w8a8_gemm_kernel(
    const typename MMA_Traits::ValTypeA* activations,
    const typename MMA_Traits::ValTypeB* weights,
    typename MMA_Traits::ValTypeC* output,
    const float* scales_a,
    const float* scales_b,
    int64_t m, int64_t n, int64_t k,
    cudaStream_t stream) {
    
    // Call the main GEMM function
    w8a8_gemm_tn(m, n, k,
                 1.0f,  // alpha
                 activations, k,  // A matrix with leading dimension k
                 weights, k,      // B matrix with leading dimension k  
                 0.0f,   // beta
                 output, n,       // C matrix with leading dimension n
                 stream);
}

// Scaling kernel for W8A8
__global__ void cute_w8a8_gemm_kernel_scale(
    int32_t* output, const float* scales_a, const float* scales_b,
    int64_t m, int64_t n, int64_t k) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    
    int64_t row = idx / n;
    int64_t col = idx % n;
    
    // Apply scaling to int32 output
    float scale = scales_a[row] * scales_b[col];
    output[idx] = static_cast<int32_t>(output[idx] * scale);
}

// Main GEMM function using CUTE
template <typename OutputType>
std::vector<paddle::Tensor> W8A8GemmCuteImpl(
    const paddle::Tensor& activations,
    const paddle::Tensor& weights,
    const paddle::Tensor& scales_a,
    const paddle::Tensor& scales_b,
    int64_t m, int64_t n, int64_t k) {
    
    // Create output tensor (int32 intermediate result)
    paddle::Tensor output_int32 = paddle::empty({m, n}, 
                                               paddle::DataType::INT32, 
                                               activations.place());
    
    // Choose appropriate MMA traits for W8A8
    using MMA_Traits = cute::MMA_Traits<cute::SM80_16x8x16_S32S8S8S32_TN>;
    
    // Call CUTE-based kernel
    cute_w8a8_gemm_kernel<MMA_Traits>(
        reinterpret_cast<const typename MMA_Traits::ValTypeA*>(activations.data()),
        reinterpret_cast<const typename MMA_Traits::ValTypeB*>(weights.data()),
        reinterpret_cast<typename MMA_Traits::ValTypeC*>(output_int32.data()),
        scales_a.data<float>(),
        scales_b.data<float>(),
        m, n, k,
        activations.stream());
    
    // Apply scaling
    int64_t num_elements = m * n;
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;
    
    cute_w8a8_gemm_kernel_scale<<<num_blocks, block_size, 0, activations.stream()>>>(
        output_int32.data<int32_t>(),
        scales_a.data<float>(),
        scales_b.data<float>(),
        m, n, k);
    
    // Convert to final output type
    paddle::Tensor output = paddle::empty({m, n}, 
                                         paddle::DataType::FLOAT32, 
                                         activations.place());
    
    // In real implementation, this would be a proper conversion kernel
    // For now, we'll just copy and convert
    auto* output_data = output.data<float>();
    auto* int32_data = output_int32.data<int32_t>();
    
    for (int64_t i = 0; i < num_elements; ++i) {
        output_data[i] = static_cast<float>(int32_data[i]);
    }
    
    return {output};
}

// Paddle operator implementation
std::vector<paddle::Tensor> W8A8Gemm(
    const paddle::Tensor& activations,
    const paddle::Tensor& weights,
    const paddle::Tensor& scales_a,
    const paddle::Tensor& scales_b,
    const std::string& out_dtype) {
    
    // Validate input dimensions
    std::vector<int64_t> activations_dims = activations.shape();
    std::vector<int64_t> weights_dims = weights.shape();
    
    PADDLE_ENFORCE_EQ(
        activations_dims.size(), 2,
        phi::errors::InvalidArgument("Activations should be 2D tensor"));
    PADDLE_ENFORCE_EQ(
        weights_dims.size(), 2,
        phi::errors::InvalidArgument("Weights should be 2D tensor"));
    
    int64_t m = activations_dims[0];
    int64_t k_activations = activations_dims[1];
    int64_t k_weights = weights_dims[0];
    int64_t n = weights_dims[1];
    
    PADDLE_ENFORCE_EQ(
        k_activations, k_weights,
        phi::errors::InvalidArgument(
            "Inner dimensions must match: activations[%d] != weights[%d]",
            k_activations, k_weights));
    
    // Dispatch based on output dtype
    if (out_dtype == "bfloat16") {
        auto result = W8A8GemmCuteImpl<paddle::bfloat16>(
            activations, weights, scales_a, scales_b, m, n, k_activations);
        return result;
    } else if (out_dtype == "float16") {
        auto result = W8A8GemmCuteImpl<paddle::float16>(
            activations, weights, scales_a, scales_b, m, n, k_activations);
        return result;
    } else if (out_dtype == "float32") {
        return W8A8GemmCuteImpl<float>(
            activations, weights, scales_a, scales_b, m, n, k_activations);
    } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Unsupported output dtype: %s. Supported: bfloat16, float16, float32", 
            out_dtype.c_str()));
    }
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
    .SetKernelFn(PD_KERNEL(W8A8Gemm))
    .SetInferShapeFn(PD_INFER_SHAPE(W8A8GemmShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(W8A8GemmDtype));