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

#include "w8a8_group_gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>
#include <algorithm>

// Helper template for dtype conversion
template<paddle::DataType D>
struct CutlassDtypeTraits;

template<>
struct CutlassDtypeTraits<paddle::DataType::BFLOAT16> {
    using DataType = cutlass::bfloat16_t;
};

template<>
struct CutlassDtypeTraits<paddle::DataType::FLOAT16> {
    using DataType = cutlass::half_t;
};

template<>
struct CutlassDtypeTraits<paddle::DataType::FLOAT32> {
    using DataType = float;
};

// Paddle operator implementation
template <paddle::DataType D, typename T>
void RunW8A8GroupGemm(const int8_t *activations,
                    const int8_t *weights,
                    T *output,
                    const float *scales_a,
                    const float *scales_b,
                    int lda, int ldb, int ldc, int ldd,
                    int m,
                    int k,
                    int n,
                    cudaStream_t stream) {

    // Get dimensions
    using ElementA = int8_t;
    using LayoutA = cutlass::layout::RowMajor;
    using ElementB = int8_t;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementC = typename CutlassDtypeTraits<D>::DataType;
    using LayoutC = cutlass::layout::RowMajor;
    using ElementAccumulator = float;
    using ArchTag = cutlass::arch::Sm80;

    static int const kStages = 5;
    static int const kAlignmentAB = 128 / cutlass::sizeof_bits<ElementA>::value;
    static int const kAlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

    int M1 = 64, N1 = 64, K1 = 64, M2 = 32, N2 = 32, K2 = 64;

    // Call the launcher function directly
    bool success = cutlass::W8A8GroupGemmLauncher<M1, N1, K1, M2, N2, K2,
                                    ElementA,
                                    LayoutA,
                                    ElementB,
                                    LayoutB,
                                    ElementC,
                                    LayoutC,
                                    ElementAccumulator,
                                    ArchTag,
                                    kStages,
                                    kAlignmentAB,
                                    kAlignmentC>(
        activations,
        weights,
        reinterpret_cast<ElementC*>(output),
        scales_a,
        scales_b,
        lda, ldb, ldc, ldd,
        m, n, k,
        stream);

    PADDLE_ENFORCE_EQ(success, true,
                     phi::errors::Fatal("cutlass W8A8GroupGemm runtime error"));
}

std::vector<paddle::Tensor> W8A8GroupGemm(const paddle::Tensor &activations,
                                        const paddle::Tensor& weights,
                                        const paddle::Tensor& scales_a,
                                        const paddle::Tensor& scales_b,
                                        const std::string &out_dtype) {
    std::vector<int64_t> activations_dims = activations.shape(), weights_dims = weights.shape();
    PADDLE_ENFORCE_EQ(
        activations_dims[activations_dims.size() - 1],
        weights_dims[weights_dims.size() - 1],
        phi::errors::InvalidArgument(
            "The last dimension of activations and weights should be equal. But "
            "received activations[%d] != weights[%d].",
            activations_dims[activations_dims.size() - 1],
            weights_dims[weights_dims.size() - 1]));

    int64_t m = activations_dims[activations_dims.size() - 2];
    int64_t k = activations_dims[activations_dims.size() - 1];
    int64_t n = weights_dims[weights_dims.size() - 2];

    int64_t rank = activations.size();
    int64_t lda = activations.dims()[rank - 1];
    int64_t ldb = weights.dims()[rank - 1];
    int64_t ldc = 0;
    int64_t ldd = weights.dims()[rank - 1];

    if (out_dtype == "bfloat16") {
        paddle::Tensor out =
            paddle::empty({m, n}, paddle::DataType::BFLOAT16, activations.place());
        RunW8A8GroupGemm<paddle::DataType::BFLOAT16, paddle::bfloat16>(
            activations.data<int8_t>(),
            weights.data<int8_t>(),
            out.data<paddle::bfloat16>(),
            scales_a.data<float>(),
            scales_b.data<float>(),
            m, k, n,
            activations.stream());
        return {out};
    } else if (out_dtype == "float16") {
        paddle::Tensor out =
            paddle::empty({m, n}, paddle::DataType::FLOAT16, activations.place());
        RunW8A8GroupGemm<paddle::DataType::FLOAT16, paddle::float16>(
            activations.data<int8_t>(),
            weights.data<int8_t>(),
            out.data<paddle::float16>(),
            scales_a.data<float>(),
            scales_b.data<float>(),
            m, k, n,
            activations.stream());
        return {out};
    } else if (out_dtype == "float32") {
        paddle::Tensor out =
            paddle::empty({m, n}, paddle::DataType::FLOAT32, activations.place());
        RunW8A8GroupGemm<paddle::DataType::FLOAT32, paddle::float32>(
            activations.data<int8_t>(),
            weights.data<int8_t>(),
            out.data<paddle::float32>(),
            scales_a.data<float>(),
            scales_b.data<float>(),
            m, k, n,
            activations.stream());
        return {out};
    } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "only support bfloat16, float16, float32, but got %s", out_dtype));
    }
}

std::vector<std::vector<int64_t>> W8A8GroupGemmShape(
    const std::vector<int64_t>& activations_shape,
    const std::vector<int64_t>& weights_shape,
    const std::vector<int64_t>& scales_a_shape,
    const std::vector<int64_t>& scales_b_shape,
    const std::vector<int64_t>& expert_offsets_shape) {

    PADDLE_ENFORCE_EQ(activations_shape.size(), 2,
                     phi::errors::InvalidArgument("Activations must be 2D"));
    PADDLE_ENFORCE_EQ(weights_shape.size(), 2,
                     phi::errors::InvalidArgument("Weights must be 2D"));

    int m = activations_shape[0];
    int n = weights_shape[0];

    return {{m, n}};
}

std::vector<paddle::DataType> W8A8GroupGemmDtype(
    const paddle::DataType& activations_dtype,
    const paddle::DataType& weights_dtype,
    const paddle::DataType& scales_a_dtype,
    const paddle::DataType& scales_b_dtype,
    const paddle::DataType& expert_offsets_dtype) {

    return {paddle::DataType::BFLOAT16};
}

// PD_KERNEL binding
PD_BUILD_STATIC_OP(w8a8_group_gemm)
    .Inputs({"activations", "weights", "scales_a", "scales_b", "expert_offsets"})
    .Outputs({"outputs"})
    .SetKernelFn(PD_KERNEL(W8A8GroupGemm))
    .SetInferShapeFn(PD_INFER_SHAPE(W8A8GroupGemmShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(W8A8GroupGemmDtype));
