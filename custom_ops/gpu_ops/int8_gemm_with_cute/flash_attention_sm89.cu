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
#include "mma_sm89.hpp"
#include "mma_traits_sm89.hpp"

using namespace cute;

// Flash attention GEMM functions
// Blocks until all but N previous cp.async.commit_group operations have committed.
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// NOTE: A矩阵已经在寄存器中的gemm封装
template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
inline __device__ void gemm_A_in_regs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                      TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                      ThrCopy smem_thr_copy_B) {
    // NOTE: 符合M N K描述: A[M, K] @ B[N, K] = C[M, N]
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    // NOTE: retile 成拷贝需要的大小
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

template<typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
inline __device__ void gemm_smem(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    // NOTE: s -> reg
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

// copy from S to D with tiled_copy
template <typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        #pragma unroll
        for (int k = 0; k < size<2>(S); ++k) {
          cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        }
    }
}

template<typename ToType, typename Fragment>
inline __device__ auto convert_float32_to_fp8(Fragment const &acc_fp32) {
  Tensor acc_fp8 = make_tensor<ToType>(shape(acc_fp32));
  using convert_type = std::conditional_t<
                            std::is_same_v<ToType, cutlass::float_e5m2_t>,
                            __nv_fp8x2_e5m2,
                            __nv_fp8x2_e4m3
                        >;
  {
    Tensor acc_fp32x2 = recast< float2>(acc_fp32);
    Tensor acc_fp8x2 = recast<convert_type>(acc_fp8);
    for (int i = 0; i < size(acc_fp32x2); ++i) { 
      acc_fp8x2(i) = convert_type(acc_fp32x2(i)); 
    }
  }
  return acc_fp8;
}

// Convert acc_layout from (MMA=(2,2), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
};

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max);

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum);

// Apply the exp to all the elements.
template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        // If max is -inf, then all elements must have been -inf (possibly due to masking).
        // We don't want (-inf - (-inf)) since that would give NaN.
        // If we don't have float around M_LOG2E the multiplication is done in fp64.
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
            // Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
            // max * log_2(e)) This allows the compiler to use the ffma
            // instruction instead of fadd and fmul separately.
            tensor(mi, ni) = expf(tensor(mi, ni) * scale - max_scaled);
        }
    }
}

// scores:((2, MMA_M),(2, MMA_N))，经过了 causal 之后的 Q_i 和 k_j^T 的乘积，
// scores_max:(2 * MMA_N), rowmax 的结果
// scores_sum:(2 * MMA_N)， rowsum 的结果
// acc_o:((2, 2),(MMA_M, MMA_N))， 最后的计算结果
template<bool Is_first, typename Tensor0, typename Tensor1, typename Tensor2>
inline __device__ void softmax_rescale_o(Tensor0 &scores, Tensor1 &scores_max, Tensor1 &scores_sum,
                                         Tensor2 &acc_o, float softmax_scale_log2) {
    if (Is_first) {
        // NOTE: 第一次softmax不需要rescale, 只需要记录 Sij(kblockM, kblockN) 的 rowmax 和 rowsum
        reduce_max</*zero_init=*/true>(scores, scores_max);
        scale_apply_exp2(scores, scores_max, softmax_scale_log2);
        reduce_sum(scores, scores_sum);
    } else {
        // 记录上一次的 rowmax
        Tensor scores_max_prev = make_fragment_like(scores_max); // 相当于公式中的 m_i^{j-1}
        cute::copy(scores_max, scores_max_prev);
        // NOTE: 计算最新的 max 
        // reduce_max包含步:
        //  1. 求当前thread内max: 遍历
        //  2. reduce thread间的max: 使用线程数洗牌指令做 all reduce，每个线程都获得了最大值
        reduce_max</*zero_init=*/false>(scores, scores_max); // scores_max 变成最新的最大值，相当于公式中的 m_i^{j}
        // Reshape acc_o from ((2,2), MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        // 将acc_o转换成符合2D直觉的(nrow, ncol)的形状
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size(scores_max); ++mi) { // 遍历每一行
            // NOTE: 辅助变量: 当前行max
            float scores_max_cur = scores_max(mi); // 当前行的最大值
            // NOTE: 计算上一次 score_sum 的 rescale 值
            float scores_scale = expf((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2); // 想当于公式中的 e^{m_i^{j-1} - m_i^{j}}.
            scores_sum(mi) *= scores_scale; // 想当于公式中的  e^{m_i^{j-1} - m_i^{j}}l_i^{j-1}
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) { acc_o_rowcol(mi, ni) *= scores_scale; } // 想当于公式中的 e^{m_i^{j-1} - m_i^{j}}O_i^{j-1}
        }
        // NOTE: Apply the exp to all the elements with new max value， 这里相当于论文公式里的 P_i^_j
        scale_apply_exp2(scores, scores_max, softmax_scale_log2);

        Tensor scores_sum_cur = make_fragment_like(scores_sum);  // l_i^{j} = e^{m_i^{j-1} - m_i^{j}}O_i^{j-1}
        // NOTE: 累计求和
        reduce_sum(scores, scores_sum_cur); // rowsum(P_i^_j)
        // NOTE: 新分母累加到旧分母
        #pragma unroll
        for (int mi = 0; mi < size(scores_sum); ++mi) { scores_sum(mi) += scores_sum_cur(mi); } // l{ij} = e^{m_i^{j-1} - m_i^{j}}O_i^{j-1} + rowsum(P_i^_j)
    }
};

template<typename T>
struct MaxOp {
__device__ inline T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ inline float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ inline T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

// tensor:((2, MMA_M),(2, MMA_N))
// summary:(2 * MMA_N)
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void thread_reduce_(Tensor<Engine0, Layout0> const &tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(summary) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); mi++) {
        summary(mi) = zero_init ? tensor(mi, 0) : op(summary(mi), tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ni++) {
            summary(mi) = op(summary(mi), tensor(mi, ni));
        }
    }
}

// summary:(2 * MMA_N)
// summary:(2 * MMA_N)
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void quad_allreduce_(Tensor<Engine0, Layout0> &dst, Tensor<Engine1, Layout1> &src, Operator &op) {
    CUTE_STATIC_ASSERT_V(size(dst) == size(src));
    #pragma unroll
    for (int i = 0; i < size(dst); i++){
        // NOTE: 4表示4个线程, 因为在SM80_16x8x16_F32F16F16F32_TN中,
        // 每组每行就是4个线程处理8个value的, 每个线程处理2个value
        dst(i) = Allreduce<4>::run(src(i), op); // SM89_16x8x32_F32F8F8F32_E4M3_TN 同
    }
}

// tensor:((2, MMA_M),(2, MMA_N))
// summary:(2 * MMA_N)
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void reduce_(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &summary, Operator &op) {
    // NOTE: 遍历tensor每行, 记录到summary中
    // reduce 当前thread的max
    thread_reduce_<zero_init>(tensor, summary, op);
    // NOTE: 二分法对summary[]进行reduce
    // reduce thread间的max
    quad_allreduce_(summary, summary, op);
}

// scores:((2, MMA_M),(2, MMA_N))
// scores_max:(2 * MMA_N)
template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max){
    MaxOp<float> max_op;
    reduce_<zero_init>(tensor, max, max_op);
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum){
    SumOp<float> sum_op;
    reduce_(tensor, sum, sum_op);
}

// Shared Storage with Aligned addresses for Flash Attention.
template <class ElementType, class SmemLayoutQ, class SmemLayoutK, class SmemLayoutV>
struct FlashAttentionSharedStorage {
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<ElementType, cute::cosize_v<SmemLayoutV>> smem_v;
};

// Flash attention kernel traits for SM89
template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type>
struct Flash_kernel_traits_sm89 {

    using Element = elem_type;
    static constexpr bool Has_cp_async = true;

    using ElementAccum = float;
    using index_t = uint32_t;

    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, cutlass::float_e5m2_t>,
        MMA_Atom<SM89_16x8x32_F32F8F8F32_E5M2_TN>,
        MMA_Atom<SM89_16x8x32_F32F8F8F32_E4M3_TN>
    >;

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type,
         typename Base=Flash_kernel_traits_sm89<kHeadDim_, kBlockM_, kBlockN_, kNWarps_, elem_type> >
struct Flash_fwd_kernel_traits_sm89 : public Base {
    using Element = typename Base::Element;
    using ElementAccum = typename Base::ElementAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;

    // The number of threads.
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;

    static_assert(kHeadDim % 32 == 0);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _32>>;
        
    using SmemLayoutAtom = decltype(
        composition(Swizzle<kSwizzle, 4, 3>{},
                    Layout<Shape<_16, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtom{},
        Shape<Int<kHeadDim>, Int<kBlockN>>{}));

    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzle, 4, 3>{},
                    Layout<Shape<Int<16>, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));

    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;

    static constexpr int kSmemQCount = size(SmemLayoutQ{});
    static constexpr int kSmemKVCount = size(SmemLayoutK{}) + size(SmemLayoutV{});
    static constexpr int kSmemQSize = kSmemQCount * sizeof(Element);
    static constexpr int kSmemKVSize = kSmemKVCount * sizeof(Element);
    static constexpr int kSmemSize = kSmemQSize + kSmemKVSize;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDim % kGmemElemsPerLoad == 0, "kHeadDim must be a multiple of kGmemElemsPerLoad");

    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0, "kNThreads must be a multiple of kGmemThreadsPerRow");
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;

    using Gmem_copy_struct = std::conditional_t<
        Has_cp_async,
        SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
        DefaultCopy
    >;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));
};

struct Flash_fwd_params {
    int bs;
    int head;
    int q_seqlen;
    int dim;
    int k_head;
    int k_seqlen;
    int bs_stride;
    int head_stride;
    int seqlen_stride;
    int dim_stride;
    float softmax_scale;
    float softmax_scale_log2;
    bool is_causal;
    bool is_fp8_e5m2;
    void* softmax_lse_ptr;
    void* q_ptr;
    void* k_ptr;
    void* v_ptr;
    void* out_ptr;
};

// Flash attention GEMM implementation for SM89
template <typename Kernel_traits, bool Is_causal=false, typename Params>
__global__ void flash_attention_v2_sm89_kernel(const Params params) {
  using namespace cute;

  // m block index
  const int m_block = blockIdx.x;

  // bs * head
  const int base_id = blockIdx.y;
  // The thread index.
  const int tidx = threadIdx.x;

  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using TiledMma = typename Kernel_traits::TiledMma;
  using index_t = typename Kernel_traits::index_t;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutK;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutV;

  constexpr int kNWarps = Kernel_traits::kNWarps;
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  // Shared memory.
  extern __shared__ char smem_[];
  using SharedStorage = FlashAttentionSharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>;
  SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_);

  const int bs_head_offset = base_id * params.head_stride;

  // NOTE: convert C pointer to Tensor for convenience
  Tensor Q = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + bs_head_offset),
      make_shape(params.q_seqlen, Int<kHeadDim>{}),
      make_stride(Int<kHeadDim>{}, Int<1>{}));
  Tensor K = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + bs_head_offset),
      make_shape(params.k_seqlen, Int<kHeadDim>{}),
      make_stride(Int<kHeadDim>{}, Int<1>{}));
  Tensor V = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + bs_head_offset),
      make_shape(Int<kHeadDim>{}, params.k_seqlen),
      make_stride(params.k_seqlen, Int<1>{}));
  Tensor O = make_tensor(
      make_gmem_ptr(reinterpret_cast<Element *>(params.out_ptr) + bs_head_offset),
      make_shape(params.q_seqlen, Int<kHeadDim>{}),
      make_stride(Int<kHeadDim>{}, Int<1>{}));

  // 加载Q, K, V分块
  Tensor gQ = local_tile(Q, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));
  // NOTE: loading流水线, 初次加载所需K, V
  Tensor gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(0, _));
  Tensor gV = local_tile(V, make_tile(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_, 0));

  // 获取MMA抽象
  TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_slice(tidx);

  // Construct SMEM tensors.
  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_k.data()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(shared_storage.smem_v.data()), SmemLayoutV{});

  // NOTE: copy抽象
  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  // NOTE: 定义gmem -> smem拷贝的src, dst
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ(_, _, 0));
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  // NOTE: 定义smem -> reg拷贝的dst
  Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK  = thr_mma.partition_fragment_B(sK);
  Tensor tOrV  = thr_mma.partition_fragment_B(sV);
 
  // NOTE: 准备拷贝Q, K, V到 reg 的copy对象 (smem --> reg)
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsV = smem_thr_copy_V.partition_S(sV);

  // 流水线加载初始Q, K
  copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  copy(gmem_tiled_copy_QKV, tKgK, tKsK);
  cute::cp_async_fence();

  Tensor rAccOut = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

  // NOTE: K, V分块的数量: 处理的区间
  const int n_block_min = 0;
  int seqlen_start = m_block * kBlockM;
  int seqlen_end = (m_block + 1) * kBlockM;
  int n_block_max = Is_causal ? cute::ceil_div(seqlen_end, kBlockN) : cute::ceil_div(params.k_seqlen, kBlockN); 

  // NOTE: 需要记录的max
  Tensor scores_max = make_tensor<ElementAccum>(Shape<Int<2 * size<1>(rAccOut)>>{});

  // NOTE: 需要记录的 softmax 分母
  Tensor scores_sum = make_fragment_like(scores_max);

  clear(rAccOut);

  for (int nbi = n_block_min; nbi < n_block_max; nbi++) {
    auto rAccScore = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(rAccScore);

    // 等待Q, K的gmem -> smem拷贝完成
    cp_async_wait<0>();
    __syncthreads();

    // gemm的同时异步加载V
    gV = local_tile(V, make_tile(Int<kHeadDim>{}, Int<kBlockN>{}), make_coord(_,  nbi));
    tVgV = gmem_thr_copy_QKV.partition_S(gV(_, _, 0));
    copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    cute::cp_async_fence();

    // O = Q@K.T
    gemm_smem(rAccScore, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
        smem_thr_copy_Q, smem_thr_copy_K);

    // NOTE: Convert from layout C to layout A
    Tensor scores = make_tensor(rAccScore.data(), convert_layout_acc_rowcol(rAccScore.layout()));

    // NOTE: 2. mask within N BLOCKs
    if (Is_causal ==  true && nbi * kBlockN >= seqlen_start) {
      // Masking implementation would go here
    }
  
    // NOTE: 等待V加载完成
    cp_async_wait<0>();
    __syncthreads();
    
    // advance K
    if (nbi != n_block_max - 1) {
      gK = local_tile(K, make_tile(Int<kBlockN>{}, Int<kHeadDim>{}), make_coord(nbi + 1, _));
      tKgK = gmem_thr_copy_QKV.partition_S(gK(_, _, 0));
      copy(gmem_tiled_copy_QKV, tKgK, tKsK);
      cute::cp_async_fence();
    }
    
    // 计算softmax
    nbi == 0 ? softmax_rescale_o<true>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale) :
      softmax_rescale_o<false>(scores, scores_max, scores_sum, rAccOut, params.softmax_scale);

    // 计算完成后， scores 相当于公式中的 P_i^j
    Tensor rP = convert_float32_to_fp8<Element>(rAccScore);
    
    // NOTE: Convert from layout C to layout A
    auto tOrPLayout = Layout<Shape<Shape<_4, _2, _2>, _1, _2>>{};
    Tensor tOrP = make_tensor(rP.data(), tOrPLayout);
    
    // rAccOut:((2, 2),(MMA_M, MMA_N))
    gemm_A_in_regs(rAccOut, tOrP, tOrV, tOsV, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
  } 
  
  
  // Epilogue
  Tensor acc_o_rowcol = make_tensor(rAccOut.data(), convert_layout_acc_rowcol(rAccOut.layout()));
  
  #pragma unroll
  for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
    float sum = scores_sum(mi);
    float inv_sum = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
    float scale = inv_sum;
    
    #pragma unroll
    for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
      acc_o_rowcol(mi, ni) *= scale;
    }
  }

  // Convert acc_o from fp32 to fp8
  Tensor rO = convert_float32_to_fp8<Element>(rAccOut);
  Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});

  // Partition sO to match the accumulator partitioning
  auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

  Tensor gO = local_tile(O, make_tile(Int<kBlockM>{}, Int<kHeadDim>{}), make_coord(m_block, _));

  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO(_, _, 0));

  __syncthreads();

  cute::copy(gmem_tiled_copy_O, tOsO, tOgO);
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_fwd_sm89(Flash_fwd_params &params, cudaStream_t stream) {
  using Element = typename Kernel_traits::Element;
  using SmemLayoutQ = typename Kernel_traits::SmemLayoutQ;
  using SmemLayoutK = typename Kernel_traits::SmemLayoutK;
  using SmemLayoutV = typename Kernel_traits::SmemLayoutV;

  const int num_m_block =
      (params.q_seqlen + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;

  dim3 grid(num_m_block, params.bs * params.head, 1);
  dim3 block(Kernel_traits::kNThreads);

  int smem_size = int(sizeof(FlashAttentionSharedStorage<Element, SmemLayoutQ, SmemLayoutK, SmemLayoutV>));

  auto kernel = &flash_attention_v2_sm89_kernel<Kernel_traits, Is_causal, Flash_fwd_params>;
  if (smem_size >= 48 * 1024) {
      CUDA_ERROR_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  }
  
  kernel<<<grid, block, smem_size, stream>>>(params);
}

template<typename T, int Headdim>
void run_flash_fwd_sm89_(Flash_fwd_params &params, cudaStream_t stream) {
    // Using the same block sizes as in the original implementation
    run_flash_fwd_sm89<Flash_fwd_kernel_traits_sm89<Headdim, 64, 64, 4, T>, false>(params, stream);
}