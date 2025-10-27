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
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/arch.h"

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/extension.h"
#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/common/flags.h"

namespace cutlass {

template <int M1, int N1, int K1, int M2, int N2, int K2,
        typename ElementA_,
        typename LayoutA_,
        typename ElementB_,
        typename LayoutB_,
        typename ElementC_,
        typename LayoutC_,
        typename ElementAccumulator_,
        typename ArchTag_,
        int kStages_,
        int kAlignmentAB_,
        int kAlignmentC_>
bool W8A8GroupGemmLauncher(const ElementA_* A,
                        const ElementB_* B,
                        ElementC_* C,
                        const ElementAccumulator_* act_scales,
                        const ElementAccumulator_* weight_scales,
                        int lda, int ldb, int ldc, int ldd,
                        int m, int n, int k,
                        cudaStream_t stream) {
    using ElementA = ElementA_;
    using LayoutA = LayoutA_;
    int kAlignmentA = kAlignmentAB_;
    using ElementB = ElementB_;
    using LayoutB = LayoutB_;
    int kAlignmentB = kAlignmentAB_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    int kAlignmentC = kAlignmentC_;
    using ElementAccumulator = ElementAccumulator_;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = ArchTag_;

    using ThreadblockShape = cutlass::gemm::GemmShape<M1, N1, K1>;
    using WarpShape = cutlass::gemm::GemmShape<M2, N2, K2>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
            ElementC, kAlignmentC, ElementAccumulator, ElementAccumulator>;

    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
    int kStages = kStages_;

    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        ElementA,
        LayoutA,
        cutlass::ComplexTransform::kNone,
        kAlignmentA,
        ElementB,
        LayoutB,
        cutlass::ComplexTransform::kNone,
        kAlignmentB,
        ElementC,
        LayoutC,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOutputOp,
        ThreadblockSwizzle,
        kStages,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

    using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

    typename GemmGrouped::EpilogueOp::Params epilogue_op(ElementAccumulator(1.f),
                                            ElementAccumulator(0.f));
    // Create problem sizes
    cutlass::gemm::GemmCoord problem_sizes(m, n, k);
    int problem_count = problem_sizes.size();
    int threadblock_count = GemmGrouped::sufficient(problem_sizes.data(), problem_count);

    // Create arguments
    typename GemmGrouped::Arguments args(
        problem_sizes_device,
        problem_count,
        threadblock_count,
        reinterpret_cast<const ElementA*>(A),
        reinterpret_cast<const ElementB*>(B),
        reinterpret_cast<ElementC*>(C),
        reinterpret_cast<ElementC*>(C),
        lda,
        ldb,
        ldc,
        ldd,
        problem_sizes
    );

    GemmGrouped gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

    Status init_status = gemm_op.initialize(args, workspace.get());
    if (init_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to initialize cutlass variable batched gemm. Error: " +
            std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[W8A8GroupGemm Runner] " + err_msg);
    }

    auto run_status = gemm_op.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to run cutlass variable batched gemm. Error: " +
            std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[W8A8GroupGemm Runner] " + err_msg);
}

    return true;
};

} // namespace cutlass
