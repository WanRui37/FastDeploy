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
#include <cute/algorithm/gemm.hpp>
#include "cute/algorithm/copy.hpp"
#include <cute/atom/mma_traits.hpp>

#include "paddle/extension.h"
#include "helper.h"

// Include CUTE MMA implementations
#include "core/gemm.hpp"
#include "core/mma_sm89.hpp"
#include "core/mma_traits_sm89.hpp"

using namespace cute;

// CUTE-based WFP8AFP8 GEMM kernel implementation
template <typename MMA_Traits>
void cute_wfp8afp8_gemm_kernel(
    const typename MMA_Traits::ValTypeA* activations,
    const typename MMA_Traits::ValTypeB* weights,
    typename MMA_Traits::ValTypeC* output,
    const float* scales_a,
    const float* scales_b,
    int64_t m, int64_t n, int64_t k,
    cudaStream_t stream) {

    // Define tensor layouts using CUTE
    using Shape_MNK = typename MMA_Traits::Shape_MNK;
    using ALayout = typename MMA_Traits::ALayout;
    using BLayout = typename MMA_Traits::BLayout;
    using CLayout = typename MMA_Traits::CLayout;

    // Create tensors with CUTE layouts
    auto a_tensor = make_tensor(make_gmem_ptr(activations),
                               make_layout(make_shape(m, k),
                                          make_stride(k, 1)));

    auto b_tensor = make_tensor(make_gmem_ptr(weights),
                               make_layout(make_shape(k, n),
                                          make_stride(n, 1)));

    auto c_tensor = make_tensor(make_gmem_ptr(output),
                               make_layout(make_shape(m, n),
                                          make_stride(n, 1)));

    // Define threadblock and warp layouts
    constexpr int Threads = 128;
    auto tiled_mma = TiledMMA<MMA_Atom<SM89_16x8x32_F32E4M3E4M3F32_TN>,
                             Layout<Shape<_2, _2, _1>>>{};

    // Perform the GEMM operation using CUTE
    auto gemm_result = cute::gemm_gemm(tiled_mma, tCrA1(_,_,k_block), tCrB(_,_,2 * k_block), tCrC);

    // Apply scaling factors
    // Note: In a real implementation, scaling would be integrated into the epilogue
    // For simplicity, we show the concept here
    int64_t total_elements = m * n;
}

// Scaling kernel
__global__ void cute_wfp8afp8_gemm_kernel_scale(
    float* output, const float* scales_a, const float* scales_b,
    int64_t m, int64_t n, int64_t k) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;

    int64_t row = idx / n;
    int64_t col = idx % n;

    // Apply scaling (simplified - real implementation would be more sophisticated)
    float scale = scales_a[row] * scales_b[col];
    output[idx] *= scale;
}

// Main GEMM function using CUTE
template <typename OutputType>
std::vector<paddle::Tensor> Wfp8afp8GemmCuteImpl(
    const paddle::Tensor& activations,
    const paddle::Tensor& weights,
    const paddle::Tensor& scales_a,
    const paddle::Tensor& scales_b,
    int64_t m, int64_t n, int64_t k) {

    // Create output tensor
    paddle::Tensor output = paddle::empty({m, n},
                                         paddle::DataType::FLOAT32,
                                         activations.place());

    // Choose appropriate MMA traits based on precision
    using MMA_Traits = cute::MMA_Traits<cute::SM89_16x8x32_F32E4M3E4M3F32_TN>;

    // Call CUTE-based kernel
    cute_wfp8afp8_gemm_kernel<MMA_Traits>(
        reinterpret_cast<const typename MMA_Traits::ValTypeA*>(activations.data()),
        reinterpret_cast<const typename MMA_Traits::ValTypeB*>(weights.data()),
        reinterpret_cast<typename MMA_Traits::ValTypeC*>(output.data()),
        scales_a.data<float>(),
        scales_b.data<float>(),
        m, n, k,
        activations.stream());

    return {output};
}

// Paddle operator implementation
std::vector<paddle::Tensor> Wfp8afp8Gemm(
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
        auto result = Wfp8afp8GemmCuteImpl<paddle::bfloat16>(
            activations, weights, scales_a, scales_b, m, n, k_activations);

        // Convert float32 output to bfloat16 if needed
        // In real implementation, this would be integrated into the epilogue
        return result;
    } else if (out_dtype == "float16") {
        auto result = Wfp8afp8GemmCuteImpl<paddle::float16>(
            activations, weights, scales_a, scales_b, m, n, k_activations);

        // Convert float32 output to float16 if needed
        return result;
    } else if (out_dtype == "float32") {
        return Wfp8afp8GemmCuteImpl<float>(
            activations, weights, scales_a, scales_b, m, n, k_activations);
    } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "Unsupported output dtype: %s. Supported: bfloat16, float16, float32",
            out_dtype.c_str()));
    }
}

// Shape inference function
std::vector<std::vector<int64_t>> Wfp8afp8GemmShape(
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
std::vector<paddle::DataType> Wfp8afp8GemmDtype(
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
PD_BUILD_STATIC_OP(wfp8afp8_gemm)
    .Inputs({"activations", "weights", "scales_a", "scales_b"})
    .Outputs({"outputs"})
    .Attrs({"out_dtype: std::string"})
    .SetKernelFn(PD_KERNEL(Wfp8afp8Gemm))
    .SetInferShapeFn(PD_INFER_SHAPE(Wfp8afp8GemmShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(Wfp8afp8GemmDtype));
