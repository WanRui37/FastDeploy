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

#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

struct W8A8GroupGemmParams {
    const int8_t* activations;
    const int8_t* weights;
    const float* dequant_scales_a;
    const float* dequant_scales_b;
    cutlass::bfloat16_t* outputs;
    int32_t* workspace;

    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<int64_t> leading_dimensions_A;
    std::vector<int64_t> leading_dimensions_B;
    std::vector<int64_t> leading_dimensions_C;

    std::vector<int64_t> scale_dims_a;
    std::vector<int64_t> scale_dims_b;

    int num_groups;
    cudaStream_t stream;
};

class W8A8GroupGemmLauncher {
public:
    struct Config {
        using ElementA = int8_t;
        using LayoutA = cutlass::layout::RowMajor;
        using ElementB = int8_t;
        using LayoutB = cutlass::layout::ColumnMajor;
        using ElementC = cutlass::bfloat16_t;
        using LayoutC = cutlass::layout::RowMajor;
        using ElementAccumulator = int32_t;
        using ElementCompute = float;
        using ElementScale = float;

        using OperatorClass = cutlass::arch::OpClassTensorOp;
        using ArchTag = cutlass::arch::Sm80;

        using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
        using WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
        using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

        static const int kStages = 3;
        static const int kAlignmentA = 16;
        static const int kAlignmentB = 16;
        static const int kAlignmentC = 8;

        static const int kElementsPerAccess = kAlignmentC / cutlass::sizeof_bits<ElementC>::value;
    };

    class DualScaleEpilogueOp {
    public:
        using ElementOutput = Config::ElementC;
        using ElementAccumulator = Config::ElementAccumulator;
        using ElementCompute = Config::ElementCompute;
        using ElementScale = Config::ElementScale;

        static int const kCount = Config::kElementsPerAccess;

        struct Params {
            ElementCompute alpha;
            ElementCompute beta;

            Params(ElementCompute alpha_ = ElementCompute(1),
                   ElementCompute beta_ = ElementCompute(0))
                : alpha(alpha_), beta(beta_) {}
        };

        Params params;

        CUTLASS_HOST_DEVICE
        DualScaleEpilogueOp(Params const& params_ = Params()) : params(params_) {}

        CUTLASS_HOST_DEVICE
        ElementOutput operator()(
            ElementAccumulator accumulator,
            ElementCompute linear_combination,
            ElementScale scale_a,
            ElementScale scale_b) const {

            ElementCompute dequantized = ElementCompute(accumulator) * scale_a * scale_b;
            return ElementOutput(dequantized);
        }

        CUTLASS_HOST_DEVICE
        ElementOutput operator()(
            ElementAccumulator accumulator,
            ElementScale scale_a,
            ElementScale scale_b) const {
            return (*this)(accumulator, ElementCompute(1), scale_a, scale_b);
        }
    };

    using GemmKernel = typename cutlass::gemm::kernel::DefaultW8A8GemmGrouped<
        Config::ElementA,
        Config::LayoutA,
        Config::kAlignmentA,
        Config::ElementB,
        Config::LayoutB,
        Config::kAlignmentB,
        Config::ElementC,
        Config::LayoutC,
        Config::ElementAccumulator,
        Config::OperatorClass,
        Config::ArchTag,
        Config::ThreadblockShape,
        Config::WarpShape,
        Config::InstructionShape,
        DualScaleEpilogueOp,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        Config::kStages,
        cutlass::gemm::kernel::GroupScheduleMode,
        cutlass::arch::OpMultiplyAdd
    >::GemmKernel;

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    static cutlass::Status launch(
        const W8A8GroupGemmParams& params,
        int multi_processor_count);

    static size_t get_workspace_size(const W8A8GroupGemmParams& params);

    static bool validate_scale_dims(const W8A8GroupGemmParams& params);
};

std::vector<paddle::Tensor> W8A8GroupGemm(const paddle::Tensor& activations,
                                          const paddle::Tensor& weights,
                                          const paddle::Tensor& scales_a,
                                          const paddle::Tensor& scales_b,
                                          const paddle::Tensor& expert_offsets);

std::vector<std::vector<int64_t>> W8A8GroupGemmShape(
    const std::vector<int64_t>& activations_shape,
    const std::vector<int64_t>& weights_shape,
    const std::vector<int64_t>& scales_a_shape,
    const std::vector<int64_t>& scales_b_shape,
    const std::vector<int64_t>& expert_offsets_shape);

std::vector<paddle::DataType> W8A8GroupGemmDtype(
    const paddle::DataType& activations_dtype,
    const paddle::DataType& weights_dtype,
    const paddle::DataType& scales_a_dtype,
    const paddle::DataType& scales_b_dtype,
    const paddle::DataType& expert_offsets_dtype);
