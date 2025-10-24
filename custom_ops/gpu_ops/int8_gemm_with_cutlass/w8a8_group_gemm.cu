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
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"
#include "get_group_starts.cuh"
#include <vector>
#include <algorithm>

cutlass::Status W8A8GroupGemmLauncher::launch(
    const W8A8GroupGemmParams& params,
    int multi_processor_count) {

    // Validate scale dimensions
    if (!validate_scale_dims(params)) {
        return cutlass::Status::kErrorInvalidProblem;
    }

    // Create grouped GEMM device interface
    GemmGrouped gemm_grouped;

    // Prepare arguments for grouped GEMM
    typename GemmGrouped::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    args.problem_count = params.num_groups;
    args.problem_sizes = params.problem_sizes.data();
    args.lda = params.leading_dimensions_A.data();
    args.ldb = params.leading_dimensions_B.data();
    args.ldc = params.leading_dimensions_C.data();
    args.ldd = params.leading_dimensions_C.data();

    // Set pointers
    args.A = const_cast<int8_t*>(params.activations);
    args.B = const_cast<int8_t*>(params.weights);
    args.C = nullptr;  // No initial C matrix
    args.D = params.outputs;

    // Set scales for dual-scale dequantization
    args.alpha = 1.0f;
    args.beta = 0.0f;

    // Set stream
    args.stream = params.stream;

    // Initialize the operation
    cutlass::Status status = gemm_grouped.initialize(args, params.workspace);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    // Run the operation
    return gemm_grouped.run();
}

size_t W8A8GroupGemmLauncher::get_workspace_size(const W8A8GroupGemmParams& params) {
    GemmGrouped gemm_grouped;
    typename GemmGrouped::Arguments args;
    args.mode = cutlass::gemm::GemmUniversalMode::kGrouped;
    args.problem_count = params.num_groups;
    args.problem_sizes = params.problem_sizes.data();
    args.lda = params.leading_dimensions_A.data();
    args.ldb = params.leading_dimensions_B.data();
    args.ldc = params.leading_dimensions_C.data();
    args.ldd = params.leading_dimensions_C.data();

    return gemm_grouped.get_workspace_size(args);
}

bool W8A8GroupGemmLauncher::validate_scale_dims(const W8A8GroupGemmParams& params) {
    // Validate activation scales dimensions
    if (params.scale_dims_a.size() != 1 && params.scale_dims_a.size() != params.num_groups) {
        return false;
    }

    // Validate weight scales dimensions
    if (params.scale_dims_b.size() != 1 && params.scale_dims_b.size() != params.num_groups) {
        return false;
    }

    return true;
}

// Paddle operator implementation
std::vector<paddle::Tensor> W8A8GroupGemm(const paddle::Tensor& activations,
                                          const paddle::Tensor& weights,
                                          const paddle::Tensor& scales_a,
                                          const paddle::Tensor& scales_b,
                                          const paddle::Tensor& expert_offsets) {
    // Validate input tensors
    PADDLE_ENFORCE_EQ(activations.dtype(), paddle::DataType::INT8,
                     phi::errors::InvalidArgument("Activations must be int8"));
    PADDLE_ENFORCE_EQ(weights.dtype(), paddle::DataType::INT8,
                     phi::errors::InvalidArgument("Weights must be int8"));
    PADDLE_ENFORCE_EQ(scales_a.dtype(), paddle::DataType::FLOAT32,
                     phi::errors::InvalidArgument("Scales A must be float32"));
    PADDLE_ENFORCE_EQ(scales_b.dtype(), paddle::DataType::FLOAT32,
                     phi::errors::InvalidArgument("Scales B must be float32"));
    PADDLE_ENFORCE_EQ(expert_offsets.dtype(), paddle::DataType::INT64,
                     phi::errors::InvalidArgument("Expert offsets must be int64"));

    // Get dimensions
    auto activations_shape = activations.shape();
    auto weights_shape = weights.shape();
    auto scales_a_shape = scales_a.shape();
    auto scales_b_shape = scales_b.shape();
    auto expert_offsets_shape = expert_offsets.shape();

    int num_experts = expert_offsets_shape[0];
    int m = activations_shape[0];  // batch size
    int k = activations_shape[1];  // input dimension
    int n = weights_shape[0];     // output dimension

    // Create output tensor
    paddle::Tensor outputs = paddle::empty({m, n}, paddle::DataType::BFLOAT16, activations.place());

    // Prepare parameters
    W8A8GroupGemmParams params;
    params.activations = activations.data<int8_t>();
    params.weights = weights.data<int8_t>();
    params.dequant_scales_a = scales_a.data<float>();
    params.dequant_scales_b = scales_b.data<float>();
    params.outputs = reinterpret_cast<cutlass::bfloat16_t*>(outputs.data<paddle::bfloat16>());
    params.num_groups = num_experts;
    params.stream = activations.stream();

    // Prepare problem sizes and leading dimensions
    for (int i = 0; i < num_experts; ++i) {
        params.problem_sizes.push_back(cutlass::gemm::GemmCoord(m, n, k));
        params.leading_dimensions_A.push_back(k);
        params.leading_dimensions_B.push_back(k);
        params.leading_dimensions_C.push_back(n);
    }

    // Prepare scale dimensions
    params.scale_dims_a = {scales_a_shape[0]};
    params.scale_dims_b = {scales_b_shape[0]};

    // Get workspace size and allocate
    size_t workspace_size = W8A8GroupGemmLauncher::get_workspace_size(params);
    paddle::Tensor workspace = paddle::empty({static_cast<int64_t>(workspace_size)},
                                            paddle::DataType::INT32, activations.place());
    params.workspace = workspace.data<int32_t>();

    // Launch the operation
    int multi_processor_count = 0;
    cudaDeviceGetAttribute(&multi_processor_count, cudaDevAttrMultiProcessorCount, activations.device().index());

    cutlass::Status status = W8A8GroupGemmLauncher::launch(params, multi_processor_count);
    PADDLE_ENFORCE_EQ(status, cutlass::Status::kSuccess,
                     phi::errors::Fatal("W8A8 grouped GEMM launch failed"));

    return {outputs};
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

// PD_KERNEL binding similar to gemm_dequant.cu
PD_BUILD_STATIC_OP(w8a8_group_gemm)
    .Inputs({"activations", "weights", "scales_a", "scales_b", "expert_offsets"})
    .Outputs({"outputs"})
    .SetKernelFn(PD_KERNEL(phi::W8A8GroupGemm))
    .SetInferShapeFn(PD_INFER_SHAPE(phi::W8A8GroupGemmShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(phi::W8A8GroupGemmDtype));
