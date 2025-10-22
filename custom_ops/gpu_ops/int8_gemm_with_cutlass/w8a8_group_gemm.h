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
#include "w8a8_group_gemm_template.h"

namespace phi {

// W8A8 Grouped GEMM parameters structure with dual scales
struct W8A8GroupGemmParams {
    const int8_t* activations;          // W8: int8 activations
    const int8_t* weights;              // A8: int8 weights
    const float* dequant_scales_a;      // Dequantization scales for activations
    const float* dequant_scales_b;      // Dequantization scales for weights
    cutlass::bfloat16_t* outputs;       // BF16 output
    int32_t* workspace;                 // Workspace for grouped GEMM

    // Group information
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<int64_t> leading_dimensions_A;
    std::vector<int64_t> leading_dimensions_B;
    std::vector<int64_t> leading_dimensions_C;

    // Scale dimensions
    std::vector<int64_t> scale_dims_a;  // Dimensions for activation scales
    std::vector<int64_t> scale_dims_b;  // Dimensions for weight scales

    int num_groups;
    cudaStream_t stream;
};

// W8A8 Grouped GEMM launcher class with dual-scale support
class W8A8GroupGemmLauncher {
public:
    // Configuration for W8A8 GEMM with dual scales
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

        // Epilogue configuration for dual-scale dequantization
        static const int kElementsPerAccess = kAlignmentC / cutlass::sizeof_bits<ElementC>::value;
    };

    // Custom epilogue operator for dual-scale dequantization
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

            // Apply dual-scale dequantization: output = (accumulator * scale_a * scale_b)
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

    // Grouped GEMM kernel type
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
        Config::kStages
    >::GemmKernel;

    // Grouped GEMM device interface
    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    // Launch W8A8 grouped GEMM with dual scales
    static cutlass::Status launch(
        const W8A8GroupGemmParams& params,
        int multi_processor_count);

    // Get workspace size
    static size_t get_workspace_size(const W8A8GroupGemmParams& params);

    // Helper function to validate scale dimensions
    static bool validate_scale_dims(const W8A8GroupGemmParams& params);
};

}  // namespace phi
