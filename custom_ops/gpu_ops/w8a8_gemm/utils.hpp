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

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>


using namespace cute;

template<typename T>
struct PackedHalf;

template<>
struct PackedHalf<cutlass::half_t> {
    using Type = __half2;
};

template<>
struct PackedHalf<cutlass::bfloat16_t> {
    using Type = nv_bfloat162;
};


template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <int numel>
__forceinline__ __device__ void convert_c4_2_fp8(const int32_t * src, int32_t * dst1, int32_t * dst2) {
    #pragma unroll
    for (int i = 0; i < numel; ++i) {
        dst1[i] = (src[i] >> 4) & 0x0f0f0f0f;
        dst2[i] = src[i] & 0x0f0f0f0f;
    }
}

template <int numel>
__forceinline__ __device__ void convert_int8_2_int32(const int8_t * src, int32_t * dst) {
    #pragma unroll
    for (int i = 0; i < numel; ++i) {
        dst[i] = static_cast<int32_t>(src[i]);
    }
}

// 简化GEMM函数，适配int8计算
template <bool arrive=true, bool commit=true, typename Tensor0, typename Tensor1,
    typename Tensor2, typename Tensor3, typename TiledMma>
__forceinline__ __device__ void gemm(
        TiledMma &tiled_mma,
        Tensor0 &tCrA,
        Tensor1 &tCsA,
        Tensor2 const &tCrB,
        Tensor3 &tCrC) {

    warpgroup_fence_operand(tCrC);
    if constexpr (arrive) {
        warpgroup_arrive();
    }

    CUTLASS_PRAGMA_UNROLL
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // 直接进行int8矩阵乘法，累积到int32
        cute::gemm(tiled_mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }

    if constexpr (commit) {
        warpgroup_commit_batch();
    }

    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrC);
}
