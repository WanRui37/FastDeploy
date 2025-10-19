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

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

template <int kStages, class GemmType, class OutputType, class SmemLayoutA,
          class SmemLayoutB, class SmemLayoutC>
struct SharedStorage {
    union {
        struct {
            cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutA>> smem_a;
            cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutB>> smem_b;
        };
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutC>> smem_c;
    };
};

template<int kBlockM_, int kBlockN_, int kBlockK_,
        int kNWarps_, int kStages_,
        int kTiles_, int M_,
        int TokenPackSize_,
        int TAIL_N_ = 0,
        int kClusterM_ = 1,
        typename elem_type=cutlass::int8_t,
        typename OutputType = cutlass::bfloat16_t>
struct Kernel_traits {
    using Element = elem_type;
    using ElementAccum = int32_t;
    using ElementOutput = OutputType;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumMmaThreads = kNThreads - NumProducerThreads;

    static_assert(kNWarps_ == 12 || kNWarps_ == 16);

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;
    static constexpr int kTiles = kTiles_;
    static constexpr int TokenPackSize = TokenPackSize_;
    static constexpr int M = M_;
    static constexpr int TAIL_N = TAIL_N_;

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>;
    using TileShape_MNK_TAIL = Shape<Int<kBlockM>, Int<TAIL_N>, Int<kBlockK>>;

    static constexpr int kClusterM = kClusterM_;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    static constexpr int kStages = kStages_;
    static_assert(kStages > 1);

    // 使用适用于int8的MMA操作
    using TiledMma = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        Layout<Shape<Int<kBlockM / 64>, _1, _1>>{}));

    using TiledMma_TAIL = decltype(cute::make_tiled_mma(
        cute::GMMA::rs_op_selector<Element, Element, ElementAccum, TileShape_MNK_TAIL>(),
        Layout<Shape<Int<kBlockM / 64>, _1, _1>>{}));

    // 使用适用于sm80+的共享内存布局
    using SmemLayoutAtomA = decltype(
        cutlass::gemm::collective::detail::rs_smem_selector<
            GMMA::Major::K, Element, Int<kBlockM>, Int<kBlockK>>());

    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtomA{},
            make_shape(Int<kBlockM>{}, Int<kBlockK>{}, Int<kStages>{})));

    using SmemLayoutAtomB = decltype(
        cutlass::gemm::collective::detail::rs_smem_selector<
            GMMA::Major::K, Element, Int<kBlockN>, Int<kBlockK>>());

    using SmemLayoutB = decltype(
        tile_to_shape(SmemLayoutAtomB{},
            make_shape(Int<kBlockN>{}, Int<kBlockK>{}, Int<kStages>{})));

    // 使用适用于sm80+的复制原子操作
    using SmemCopyAtomAB = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomC = Copy_Atom<cute::SM75_U32x4_STSM_N, ElementOutput>;

    using SharedStorage = SharedStorage<
        kStages, Element, ElementOutput, SmemLayoutA, SmemLayoutB, SmemLayoutC>;

    // 使用通用的流水线实现，不依赖TMA
    using MainloopPipeline = typename cutlass::PipelineAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;

    // 简化输出复制逻辑
    using TiledCopyCAtom = cute::Copy_Atom<cute::UniversalCopy<cutlass::uint128_t>, OutputType>;
    using TiledCopyC = decltype(make_tiled_copy(
        TiledCopyCAtom{},
        Layout<Shape<Int<NumMmaThreads / 4>, Int<4>>>{},
        Layout<Shape<_1{}, Int<kBlockN / 4>>>{}
    ));
};
