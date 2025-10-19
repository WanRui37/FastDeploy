// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cute/tensor.hpp"

#include "cutlass/gemm/collective/collective_builder.hpp"

// #include "named_barrier.hpp"
#include "utils.hpp"


using namespace cute;
template <typename Ktraits>
struct CollectiveMainloopFwd {

    using Element = typename Ktraits::Element;
    using ElementOutput = typename Ktraits::ElementOutput;
    using TileShape_MNK = typename Ktraits::TileShape_MNK;
    using TileShape_MNK_TAIL = typename Ktraits::TileShape_MNK_TAIL;
    using ClusterShape = typename Ktraits::ClusterShape_MNK;
    using ElementAccum = typename Ktraits::ElementAccum;

    static constexpr int kStages = Ktraits::kStages;
    static constexpr int kBlockM = Ktraits::kBlockM;
    static constexpr int kBlockN = Ktraits::kBlockN;
    static constexpr int TAIL_N = Ktraits::TAIL_N;
    static constexpr int kBlockK = Ktraits::kBlockK;
    static constexpr int NumCopyThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int kTiles = Ktraits::kTiles;
    static constexpr int M = Ktraits::M;
    static constexpr int TokenPackSize = Ktraits::TokenPackSize;
    static constexpr int NumMmaThreads = Ktraits::NumMmaThreads;

    // 使用适用于sm80+的共享内存布局，不使用TMA
    using SmemLayoutA = typename Ktraits::SmemLayoutA;
    using SmemLayoutB = typename Ktraits::SmemLayoutB;
    using SmemLayoutC = typename Ktraits::SmemLayoutC;
    using SmemLayoutB_TAIL = typename Ktraits::SmemLayoutB_TAIL;

    using ShapeT = cute::Shape<int64_t, int64_t, int64_t>;
    using StrideT = cute::Shape<int64_t, _1, int64_t>;
    using LayoutT = cute::Layout<ShapeT, StrideT>;

    using MainloopPipeline = typename Ktraits::MainloopPipeline;
    using PipelineParams = typename MainloopPipeline::Params;
    using PipelineState = typename MainloopPipeline::PipelineState;
    using SmemCopyAtomAB = typename Ktraits::SmemCopyAtomAB;
    using SmemCopyAtomC = typename Ktraits::SmemCopyAtomC;
    using TiledCopyC = typename Ktraits::TiledCopyC;

    struct Arguments {
        Element const* ptr_A;
        LayoutT layout_A;
        Element const* ptr_B;
        LayoutT layout_B;
        ElementOutput * ptr_C;
        LayoutT layout_C;
        const float *weight_scale;
        const float *input_scale;
        const int64_t * tokens;
    };

    struct Params {
        Element const* ptr_A;
        LayoutT layout_A;
        Element const* ptr_B;
        LayoutT layout_B;
        ElementOutput * ptr_C;
        LayoutT layout_C;
        const float *weight_scale;
        const float *input_scale;
        const int64_t * tokens;
    };

    Params static
    to_underlying_arguments(Arguments const& args) {
        return {args.ptr_A, args.layout_A, args.ptr_B, args.layout_B,
                args.ptr_C, args.layout_C, args.weight_scale, args.input_scale, args.tokens};
    }

    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params) {
        // 移除TMA描述符预取，使用常规内存访问
    }

    // 简化load函数，使用常规的全局内存加载
    template <typename SharedStorage, typename Pipeline, typename PipelineState>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params,
         Pipeline& pipeline,
         PipelineState& smem_pipe_write,
         SharedStorage& shared_storage,
         const int64_t tokens,
         const int64_t pre_fix_tokens,
         const int bidm,
         const int bidn,
         const int bidb,
         const int tidx) {

        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_a.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_b.data()), SmemLayoutB{});

        // 使用常规的全局内存加载，不使用TMA
        Tensor gA = make_tensor(make_gmem_ptr(mainloop_params.ptr_A), mainloop_params.layout_A);
        Tensor gB = make_tensor(make_gmem_ptr(mainloop_params.ptr_B), mainloop_params.layout_B);

        // 计算全局内存偏移
        const int64_t batch_offset = bidb * M * (TokenPackSize == 0 ? 1 : TokenPackSize);
        const int64_t row_offset = bidm * kBlockM;
        const int64_t col_offset = bidn * kBlockN;

        // 简化加载逻辑：直接复制数据到共享内存
        const int kIters = kTiles / kStages;

        if (tidx < NumCopyThreads) {
            #pragma unroll
            for (int kiter = 0; kiter < kIters; ++kiter) {
                #pragma unroll
                for (int s = 0; s < kStages; s++) {
                    const int i = kiter * kStages + s;
                    pipeline.producer_acquire(smem_pipe_write);

                    // 加载A矩阵数据
                    #pragma unroll
                    for (int m = tidx; m < kBlockM; m += NumCopyThreads) {
                        for (int k = 0; k < kBlockK; ++k) {
                            const int64_t global_idx_A = batch_offset + (row_offset + m) * kBlockK + k;
                            const int smem_idx_A = m * kBlockK * kStages + k * kStages + s;
                            if (global_idx_A < size(gA)) {
                                shared_storage.smem_a[smem_idx_A] = gA(global_idx_A);
                            }
                        }
                    }

                    // 加载B矩阵数据
                    #pragma unroll
                    for (int n = tidx; n < kBlockN; n += NumCopyThreads) {
                        for (int k = 0; k < kBlockK; ++k) {
                            const int64_t global_idx_B = batch_offset + (col_offset + n) * kBlockK + k;
                            const int smem_idx_B = n * kBlockK * kStages + k * kStages + s;
                            if (global_idx_B < size(gB)) {
                                shared_storage.smem_b[smem_idx_B] = gB(global_idx_B);
                            }
                        }
                    }

                    ++smem_pipe_write;
                }
            }
        }
    }

    // 完整的mma函数实现
    template <int CUR_N, typename SharedStorage, typename FrgTensorO, typename TiledMma>
    CUTLASS_DEVICE void
    mma(Params const& mainloop_params,
            TiledMma tiled_mma,
            MainloopPipeline pipeline,
            PipelineState& smem_pipe_read,
            SharedStorage& shared_storage,
            FrgTensorO &tSrS,
            const int tidx) {

        using sMemBLayout = std::conditional_t<
            CUR_N == kBlockN,
            SmemLayoutB,
            SmemLayoutB_TAIL
        >;

        Tensor sA = make_tensor(make_smem_ptr(shared_storage.smem_a.data()), SmemLayoutA{});
        Tensor sB = make_tensor(make_smem_ptr(shared_storage.smem_b.data()), sMemBLayout{});

        tiled_mma.accumulate_ = GMMA::ScaleOut::One;

        auto threadMma = tiled_mma.get_thread_slice(tidx);

        auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtomAB{}, tiled_mma);
        auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(tidx);

        Tensor tSrA = threadMma.partition_fragment_A(sA(_, _, 0));
        Tensor tSrB = threadMma.partition_fragment_B(sB);

        auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);
        };

        const int kIters = kTiles / kStages;

        constexpr int B_STEPS = CUR_N == 0 ? 1 : (kBlockN / CUR_N);

        #pragma unroll
        for (int kiter = 0; kiter < kIters; ++kiter) {
            #pragma unroll
            for (int s = 0; s < kStages; s++) {
                Tensor tSsA = smem_thr_copy_A.partition_S(sA(_, _, s));
                consumer_wait(pipeline, smem_pipe_read);
                gemm</*wg_wait=*/0>(tiled_mma, tSrA, tSsA, tSrB(_, _, _, s * B_STEPS), tSrS, smem_tiled_copy_A, smem_thr_copy_A);
                pipeline.consumer_release(smem_pipe_read);
                ++smem_pipe_read;
            }
        }

        #pragma unroll
        for (int i = 0; i < kTiles % kStages; ++i) {
            Tensor tSsA = smem_thr_copy_A.partition_S(sA(_, _, i));
            consumer_wait(pipeline, smem_pipe_read);

            gemm</*wg_wait=*/0>(tiled_mma, tSrA, tSsA, tSrB(_, _, _, i * B_STEPS), tSrS, smem_tiled_copy_A, smem_thr_copy_A);
            pipeline.consumer_release(smem_pipe_read);
            ++smem_pipe_read;
        }
    }

    // 完整的store函数实现
    template <int CUR_N, typename SharedStorage, typename FrgTensorO, typename TiledMma>
    CUTLASS_DEVICE void
    store(Params const& mainloop_params,
        FrgTensorO & tOrO,
        SharedStorage& shared_storage,
        TiledMma tiled_mma,
        const float *input_scale,
        const float *weight_scale,
        const int64_t tokens,
        const int64_t pre_fix_tokens,
        const int bidm,
        const int bidn,
        const int bidb,
        const int tidx) {

        // 反量化逻辑：output = (accum * input_scale * weight_scale)
        #pragma unroll
        for (int i = 0; i < size(tOrO); i++) {
            float dequantized = static_cast<float>(tOrO[i]) * input_scale[0] * weight_scale[0];
            tOrO[i] = static_cast<ElementOutput>(dequantized);
        }

        // 简化存储逻辑：直接写入全局内存
        const int64_t batch_offset = bidb * M * (TokenPackSize == 0 ? 1 : TokenPackSize);
        const int64_t row_offset = bidm * kBlockM;
        const int64_t col_offset = bidn * kBlockN;

        Tensor gC = make_tensor(make_gmem_ptr(mainloop_params.ptr_C), mainloop_params.layout_C);

        #pragma unroll
        for (int i = tidx; i < size(tOrO); i += NumMmaThreads) {
            const int64_t global_idx = batch_offset + (row_offset + i / CUR_N) * M + (col_offset + i % CUR_N);
            if (global_idx < size(gC)) {
                gC(global_idx) = tOrO[i];
            }
        }
    }
};
