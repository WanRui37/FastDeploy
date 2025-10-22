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

#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/numeric_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue.h"

namespace cutlass {
namespace gemm {
namespace kernel {

// Custom epilogue for W8A8 dequantization with dual scales
template <
    typename ElementOutput_,           // bfloat16_t for output
    int Count,
    typename ElementAccumulator_,      // int32_t for accumulation
    typename ElementCompute_,          // float for computation
    typename ElementScale_ = float     // float for scales
>
class W8A8DequantEpilogue {
public:
    using ElementOutput = ElementOutput_;
    using ElementAccumulator = ElementAccumulator_;
    using ElementCompute = ElementCompute_;
    using ElementScale = ElementScale_;

    static int const kCount = Count;

    // Parameters structure
    struct Params {
        ElementCompute alpha;
        ElementCompute beta;

        Params(ElementCompute alpha_ = ElementCompute(1),
               ElementCompute beta_ = ElementCompute(0))
            : alpha(alpha_), beta(beta_) {}
    };

    // Functor operator
    CUTLASS_HOST_DEVICE
    ElementOutput operator()(
        ElementAccumulator accumulator,
        ElementCompute linear_combination,
        ElementScale scale_a,
        ElementScale scale_b) const {

        // Apply dequantization: output = (accumulator * scale_a * scale_b) converted to output type
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

// W8A8 Grouped GEMM with dual-scale dequantization to BF16
template <
    typename ElementA_,           // int8_t for activation
    typename LayoutA_,            // RowMajor for activation
    int kAlignmentA,
    typename ElementB_,           // int8_t for weight
    typename LayoutB_,            // ColumnMajor for weight
    int kAlignmentB,
    typename ElementC_,           // bfloat16_t for output
    typename LayoutC_,            // RowMajor for output
    typename ElementAccumulator_,  // int32_t for accumulation
    typename OperatorClass_,
    typename ArchTag_,
    typename ThreadblockShape_,
    typename WarpShape_,
    typename InstructionShape_,
    typename EpilogueOutputOp_,
    typename ThreadblockSwizzle_,
    int Stages,
    typename GroupScheduleMode_ = cutlass::gemm::kernel::GroupScheduleMode,
    typename Operator_ = cutlass::arch::OpMultiplyAdd
>
struct DefaultW8A8GemmGrouped {

    using ElementA = ElementA_;
    using LayoutA = LayoutA_;
    using ElementB = ElementB_;
    using LayoutB = LayoutB_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using ElementAccumulator = ElementAccumulator_;
    using OperatorClass = OperatorClass_;
    using ArchTag = ArchTag_;
    using ThreadblockShape = ThreadblockShape_;
    using WarpShape = WarpShape_;
    using InstructionShape = InstructionShape_;
    using EpilogueOutputOp = EpilogueOutputOp_;
    using ThreadblockSwizzle = ThreadblockSwizzle_;
    using Operator = Operator_;
    using GroupScheduleMode = GroupScheduleMode_;

    // Define the MMA (Matrix Multiply Accumulate) operation
    using Mma = typename cutlass::gemm::threadblock::DefaultMma<
        ElementA,
        LayoutA,
        kAlignmentA,
        ElementB,
        LayoutB,
        kAlignmentB,
        ElementAccumulator,
        LayoutC,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        Stages,
        Operator,
        false,  // Use zfill
        cutlass::gemm::SharedMemoryClearOption::kNone,
        false,  // GatherA
        false   // GatherB
    >::ThreadblockMma;

    // Define the epilogue with dual-scale dequantization support
    using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
        ThreadblockShape,
        typename Mma::Operator,
        1,  // kPartitionsK
        EpilogueOutputOp,
        EpilogueOutputOp::kCount
    >::Epilogue;

    // Define the grouped GEMM kernel
    using GemmKernel = cutlass::gemm::kernel::GemmGrouped<
        Mma,
        Epilogue,
        ThreadblockSwizzle,
        ArchTag,
        GroupScheduleMode
    >;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
