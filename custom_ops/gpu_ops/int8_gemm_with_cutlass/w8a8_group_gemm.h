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
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"

#include "cutlass/gemm/kernel/gemm_grouped_per_group_scale.h"
#include "cutlass/gemm/kernel/default_gemm_grouped_per_group_scale.h"
#include "cutlass/gemm/device/gemm_grouped.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/tensor_ref.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/util/tensor_view_io.h"
#include <cutlass/util/device_memory.h>
#include "cutlass/util/host_tensor.h"

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
bool W8A8GroupGemmLauncher_sm80(const ElementA_* A,
                        const ElementB_* B,
                        ElementC_* C,
                        const ElementAccumulator_* act_scales,
                        const ElementAccumulator_* weight_scales,
                        int lda, int ldb, int ldc, int ldd,
                        int m, int n, int k,
                        cudaStream_t stream) {
    using ElementA = ElementA_;
    using LayoutA = LayoutA_;
    static constexpr int kAlignmentA = kAlignmentAB_;
    using ElementB = ElementB_; 
    using LayoutB = LayoutB_;
    static constexpr int kAlignmentB = kAlignmentAB_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    static constexpr int kAlignmentC = kAlignmentC_;
    using ElementAccumulator = ElementAccumulator_;
    
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = ArchTag_;
    
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
            ElementC, kAlignmentC, ElementAccumulator, ElementAccumulator>;
    
    using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
    static constexpr int kStages = kStages_;
    
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

    typename GemmGrouped::EpilogueOutputOp::Params epilogue_op(1.f, 0.f);
    
    // Create problem sizes
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    problem_sizes.push_back(cutlass::gemm::GemmCoord(m, n, k));
    int problem_count = int(problem_sizes.size());
    int threadblock_count = GemmGrouped::sufficient(problem_sizes.data(), problem_count);
    
    // Allocate and copy problem sizes to device
    cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;
    problem_sizes_device.reset(problem_count);
    problem_sizes_device.copy_from_host(problem_sizes.data());

    std::vector<int64_t> lda_host(problem_count, lda);
    std::vector<int64_t> ldb_host(problem_count, ldb);
    std::vector<int64_t> ldc_host(problem_count, ldc);
    std::vector<int64_t> ldd_host(problem_count, ldd);
    // Allocate and copy leading dimensions to device
    cutlass::DeviceAllocation<int64_t> lda_device;
    cutlass::DeviceAllocation<int64_t> ldb_device;
    cutlass::DeviceAllocation<int64_t> ldc_device;
    cutlass::DeviceAllocation<int64_t> ldd_device;
    
    lda_device.reset(problem_count);
    ldb_device.reset(problem_count);
    ldc_device.reset(problem_count);
    ldd_device.reset(problem_count);
    
    lda_device.copy_from_host(lda_host.data(), problem_count);
    ldb_device.copy_from_host(ldb_host.data(), problem_count);
    ldc_device.copy_from_host(ldc_host.data(), problem_count);
    ldd_device.copy_from_host(ldd_host.data(), problem_count);

    // Create host arrays for pointers - 关键修复：使用传入的参数A, B, C
    std::vector<ElementA *> ptr_A_host(problem_count);
    std::vector<ElementB *> ptr_B_host(problem_count);
    std::vector<ElementC *> ptr_C_host(problem_count);
    std::vector<ElementC *> ptr_D_host(problem_count);
    
    // 正确设置指针数组 - 使用传入的参数
    for (int i = 0; i < problem_count; ++i) {
        ptr_A_host.at(i) = const_cast<ElementA*>(A);  // 使用传入的A指针
        ptr_B_host.at(i) = const_cast<ElementB*>(B);  // 使用传入的B指针
        ptr_C_host.at(i) = C;                         // 使用传入的C指针
        ptr_D_host.at(i) = C;                         // 输出也使用C指针
    }

    // Allocate and copy pointer arrays to device
    cutlass::DeviceAllocation<ElementA *> ptr_A;
    cutlass::DeviceAllocation<ElementB *> ptr_B;
    cutlass::DeviceAllocation<ElementC *> ptr_C;
    cutlass::DeviceAllocation<ElementC *> ptr_D;
    
    ptr_A.reset(problem_count);
    ptr_B.reset(problem_count);
    ptr_C.reset(problem_count);
    ptr_D.reset(problem_count);
    
    ptr_A.copy_from_host(ptr_A_host.data());
    ptr_B.copy_from_host(ptr_B_host.data());
    ptr_C.copy_from_host(ptr_C_host.data());
    ptr_D.copy_from_host(ptr_D_host.data());

    // Create arguments - 使用正确的设备指针
    typename GemmGrouped::Arguments args(
        problem_sizes_device.get(),
        problem_count,
        threadblock_count,
        epilogue_op,
        ptr_A.get(),    // 使用设备指针数组
        ptr_B.get(),    // 使用设备指针数组
        ptr_C.get(),    // 使用设备指针数组
        ptr_D.get(),    // 使用设备指针数组
        lda_device.get(),  // 使用设备lda数组
        ldb_device.get(),  // 使用设备ldb数组
        ldc_device.get(),  // 使用设备ldc数组
        ldd_device.get(),  // 使用设备ldd数组
        problem_sizes.data()
    );

    GemmGrouped gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::DeviceAllocation<uint8_t> workspace;
    workspace.reset(workspace_size);

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

// template <int M1, int N1, int K1, int M2, int N2, int K2,
//         typename ElementA_,
//         typename LayoutA_,
//         typename ElementB_,
//         typename LayoutB_,
//         typename ElementC_,
//         typename LayoutC_,
//         typename ElementAccumulator_,
//         typename ArchTag_,
//         int kStages_,
//         int kAlignmentAB_,
//         int kAlignmentC_>
// bool W8A8GroupGemmLauncher_sm89(const ElementA_* A,
//                         const ElementB_* B,
//                         ElementC_* C,
//                         const ElementAccumulator_* act_scales,
//                         const ElementAccumulator_* weight_scales,
//                         int lda, int ldb, int ldc, int ldd,
//                         int m, int n, int k,
//                         cudaStream_t stream) {
//     using ElementA = ElementA_;
//     using LayoutA = LayoutA_;
//     static constexpr int kAlignmentA = kAlignmentAB_;  // 改为constexpr
//     using ElementB = ElementB_; 
//     using LayoutB = LayoutB_;
//     static constexpr int kAlignmentB = kAlignmentAB_;  // 改为constexpr
//     using ElementC = ElementC_;
//     using LayoutC = LayoutC_;
//     static constexpr int kAlignmentC = kAlignmentC_;   // 改为constexpr
//     using ElementAccumulator = ElementAccumulator_;
    
//     using OperatorClass = cutlass::arch::OpClassTensorOp;
//     using ArchTag = ArchTag_;
    
//     using ThreadblockShape = cutlass::gemm::GemmShape<64, 128, 64>;
//     using WarpShape = cutlass::gemm::GemmShape<64, 32, 64>;
//     using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

//     using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
//             ElementC, kAlignmentC, ElementAccumulator, ElementAccumulator>;
        
//     using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
//     static constexpr int kStages = kStages_;  // 改为constexpr
    
//     using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGroupedPerGroupScale<
//         ElementA,
//         LayoutA,
//         cutlass::ComplexTransform::kNone,
//         kAlignmentA,  // 直接使用常量
//         ElementB,
//         LayoutB,
//         cutlass::ComplexTransform::kNone,
//         kAlignmentB,  // 直接使用常量
//         ElementC,
//         LayoutC,
//         ElementAccumulator,
//         OperatorClass,
//         ArchTag,
//         ThreadblockShape,
//         WarpShape,
//         InstructionShape,
//         EpilogueOutputOp,
//         ThreadblockSwizzle,
//         kStages>::GemmKernel;
    
//     using GemmGrouped = cutlass::gemm::device::GemmGrouped<GemmKernel>;

//     typename GemmGrouped::EpilogueOp::Params epilogue_op(ElementAccumulator(1.f),
//                                             ElementAccumulator(0.f));
//     // Create problem sizes
//     std::vector<cutlass::gemm::GemmCoord> problem_sizes;
//     problem_sizes.push_back(cutlass::gemm::GemmCoord(m, n, k));
//     int problem_count = int(problem_sizes.size());
//     int threadblock_count = GemmGrouped::sufficient(problem_sizes.data(), problem_count);
    
//     cutlass::DeviceAllocation<cutlass::gemm::GemmCoord> problem_sizes_device;
//     problem_sizes_device.reset(problem_count);
//     problem_sizes_device.copy_from_host(problem_sizes.data());

//     // Create arguments
//     typename GemmGrouped::Arguments args(
//         problem_sizes_device,
//         problem_count,
//         threadblock_count,
//         reinterpret_cast<const ElementA*>(A),
//         reinterpret_cast<const ElementB*>(B),
//         reinterpret_cast<ElementC*>(C),
//         reinterpret_cast<ElementC*>(C),
//         lda,
//         ldb,
//         ldc,
//         ldd,
//         problem_sizes
//     );

//     GemmGrouped gemm_op;

//     size_t workspace_size = gemm_op.get_workspace_size(args);
//     // Allocator::AllocationPtr workspace;
//     // workspace = allocator->Allocate(workspace_size);
//     cutlass::DeviceAllocation<uint8_t> workspace;
//     workspace.reset(workspace_size);

//     Status init_status = gemm_op.initialize(args, workspace.get());
//     if (init_status != cutlass::Status::kSuccess) {
//         std::string err_msg =
//             "Failed to initialize cutlass variable batched gemm. Error: " +
//             std::string(cutlassGetStatusString(init_status));
//         throw std::runtime_error("[W8A8GroupGemm Runner] " + err_msg);
//     }

//     auto run_status = gemm_op.run(stream);
//     if (run_status != cutlass::Status::kSuccess) {
//         std::string err_msg =
//             "Failed to run cutlass variable batched gemm. Error: " +
//             std::string(cutlassGetStatusString(run_status));
//         throw std::runtime_error("[W8A8GroupGemm Runner] " + err_msg);
// }

//     return true;
// };

} // namespace cutlass