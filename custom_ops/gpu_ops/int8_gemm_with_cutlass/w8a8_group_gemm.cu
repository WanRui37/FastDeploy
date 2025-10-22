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

#include "gemm_dequant.h"
#include "cutlass_helper.h"
#include "w8a8_group_gemm.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include <vector>
#include <algorithm>

template <typename Type, int CtaM, int CtaN, int Threads>
__global__ void int8_sq(int8_t const* act,
                          int8_t const* weight,
                          float const* scale,
                          Type* output,
                          int m,
                          int n,
                          int k) {
  using VecType = int4;
  static constexpr int kStepK = 128 / (8 * sizeof(int8_t));
  static constexpr int CtaK = kStepK * Threads;
  int tile_id_m = blockIdx.x * CtaM;
  int tile_id_n = blockIdx.y * CtaN;
  int tid = threadIdx.x;
  int8_t tile_a[kStepK], tile_w[CtaN * kStepK];
  int acc[CtaM * CtaN];
#pragma unroll
  for (int i = 0; i < CtaM * CtaN; ++i) {
    acc[i] = 0;
  }
  act += tile_id_m * k;
  weight += tile_id_n * k;
  scale += tile_id_n;
  output += tile_id_m * n + tile_id_n;
  for (int idx_k = tid * kStepK; idx_k < k; idx_k += CtaK) {
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      reinterpret_cast<VecType*>(tile_w)[i] =
          reinterpret_cast<VecType const*>(weight + i * k + idx_k)[0];
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      reinterpret_cast<VecType*>(tile_a)[0] =
          reinterpret_cast<VecType const*>(act + i * k + idx_k)[0];
#pragma unroll
      for (int j = 0; j < CtaN; ++j) {
#pragma unroll
        for (int l = 0; l < kStepK; l += 4) {
          acc[i * CtaN + j] =
              __dp4a(reinterpret_cast<int*>(tile_a + l)[0],
                     reinterpret_cast<int*>(tile_w + j * kStepK + l)[0],
                     acc[i * CtaN + j]);
        }
      }
    }
  }

  static constexpr int kWarpSize = 32;
  static constexpr int kWarpNum = Threads / kWarpSize;
  __shared__ int shmem[CtaM * CtaN * kWarpNum];
  int warp_id = tid / kWarpSize, lane_id = tid % kWarpSize;
#pragma unroll
  for (int i = 0; i < CtaM; ++i) {
#pragma unroll
    for (int j = 0; j < CtaN; ++j) {
      int val = acc[i * CtaN + j];
      val += __shfl_xor_sync(~0, val, 16);
      val += __shfl_xor_sync(~0, val, 8);
      val += __shfl_xor_sync(~0, val, 4);
      val += __shfl_xor_sync(~0, val, 2);
      val += __shfl_xor_sync(~0, val, 1);
      if (lane_id == 0) {
        shmem[i * CtaN + j + warp_id * CtaM * CtaN] = val;
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int ii = tid; ii < CtaM * CtaN; ii += Threads) {
    int mid = ii / CtaN, nid = ii % CtaN;
    int val = 0;
#pragma unroll
    for (int jj = 0; jj < kWarpNum; ++jj) {
      val += shmem[jj * CtaM * CtaN + ii];
    }
    output[mid * n + nid] = static_cast<Type>(static_cast<float>(val)*(float)*(scale+nid));
  }
}

template <typename InputType,
          typename OutputType,
          int32_t TILE_M,
          int32_t TILE_N,
          int32_t BLOCK_SIZE>
void int8_sq_kernel(GemmDequantParams const& params) {
  dim3 block(BLOCK_SIZE);
  dim3 grid(params.m / TILE_M, params.n / TILE_N);
  int8_sq<OutputType, TILE_M, TILE_N, BLOCK_SIZE>
      <<<grid, block, 0, params.stream>>>(
          reinterpret_cast<InputType const*>(params.act),
          reinterpret_cast<InputType const*>(params.weight),
          reinterpret_cast<float const*>(params.scale),
          reinterpret_cast<OutputType*>(params.output),
          params.m,
          params.n,
          params.k);
}

template <typename InputType,
          typename OutputType,
          int TILE_M,
          int TILE_N,
          int BLOCK_SIZE>
bool int8_sq_kernel_caller(GemmDequantParams const& params) {
  constexpr int cudaCoreGemmTemplateMaxM = 16;
  if (params.m == TILE_M) {
    int8_sq_kernel<InputType, OutputType, TILE_M, TILE_N, BLOCK_SIZE>(
        params);
    return true;
  }
  if constexpr (TILE_M < cudaCoreGemmTemplateMaxM) {
    return int8_sq_kernel_caller<InputType,
                                      OutputType,
                                      TILE_M + 1,
                                      TILE_N,
                                      BLOCK_SIZE>(params);
  }
  return false;
}

template <typename InputType, typename OutputType>
bool int8_sq_kernel_launcher(GemmDequantParams const& params) {
  return int8_sq_kernel_caller<InputType, OutputType, 1, 2, 256>(params);
}

template <paddle::DataType D, typename T>
void RunGemmDequant(const int8_t *a,
                    const int8_t *b,  // Transposed
                    const float *dequant_scale,
                    T *c,
                    int m,
                    int k,
                    int n,
                    cudaStream_t stream) {
    using ElementA = int8_t;
    using ElementB = int8_t;
    using ElementC = typename CutlassDtypeTraits<D>::DataType;
    using ElementCompute = int32_t;
    using ElementD = ElementC;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;

    using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 64>;
    using WarpShape = cutlass::gemm::GemmShape<32, 32, 64>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;

    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;

    static int const kStages = 5;

    /// Linear scaling operator
    using EpilogueFunctorOp = cutlass::epilogue::thread::LinearCombination<
        ElementC,
        128 / cutlass::sizeof_bits<ElementC>::value,
        ElementCompute,
        ElementCompute>;

    using GemmDequantT = cutlass::GemmDequant<ElementA,
                                              LayoutA,
                                              ElementB,
                                              LayoutB,
                                              ElementC,
                                              ElementCompute,
                                              OperatorClass,
                                              ArchTag,
                                              ThreadblockShape,
                                              WarpShape,
                                              InstructionShape,
                                              EpilogueFunctorOp,
                                              kStages>;

    using LayoutC = typename GemmDequantT::LayoutC;

    int64_t lda = LayoutA::packed({m, k}).stride(0);
    int64_t ldb = LayoutB::packed({k, n}).stride(0);
    int64_t ldc = LayoutC::packed({m, n}).stride(0);

    cutlass::gemm::GemmCoord problem_size(m, n, k);

    typename CutlassDtypeTraits<D>::DataType *c_tmp = nullptr;
    typename CutlassDtypeTraits<D>::DataType *d =
        reinterpret_cast<typename CutlassDtypeTraits<D>::DataType *>(c);

    typename GemmDequantT::TensorRefA ref_a(const_cast<int8_t *>(a), lda);
    typename GemmDequantT::TensorRefB ref_b(const_cast<int8_t *>(b), ldb);
    typename GemmDequantT::TensorRefC ref_c(c_tmp, ldc);
    typename GemmDequantT::TensorRefC ref_d(d, ldc);
    typename GemmDequantT::TensorRefScale ref_scale(
        const_cast<float *>(dequant_scale), 0);

    typename GemmDequantT::Arguments args(
        problem_size,
        ref_a,
        ref_b,
        ref_c,
        ref_d,
        ref_scale,
        {ElementCompute(1.0f), ElementCompute(0.0f)});

    GemmDequantT gemm;
    // Initialize
    auto status = gemm.initialize(args);
    PADDLE_ENFORCE_EQ(
        status,
        cutlass::Status::kSuccess,
        phi::errors::Fatal("cutlass GemmDequant initialize error"));

    // Run
    status = gemm(stream);
    PADDLE_ENFORCE_EQ(status,
                      cutlass::Status::kSuccess,
                      phi::errors::Fatal("cutlass GemmDequant runtime error"));
}

std::vector<paddle::Tensor> GemmDequant(const paddle::Tensor &x,
                                        const paddle::Tensor &y,
                                        const paddle::Tensor &scale,
                                        const std::string &out_dtype) {
    std::vector<int64_t> x_dims = x.shape(), y_dims = y.shape();
    PADDLE_ENFORCE_EQ(
        x_dims[x_dims.size() - 1],
        y_dims[y_dims.size() - 1],
        phi::errors::InvalidArgument(
            "The last dimension of x and y should be equal. But "
            "received x[%d] != y[%d].",
            "Ensure that x is not transposed and y is transposed.",
            x_dims[x_dims.size() - 1],
            y_dims[y_dims.size() - 1]));
    int64_t m = x_dims[x_dims.size() - 2];
    int64_t k = x_dims[x_dims.size() - 1];
    int64_t n = y_dims[y_dims.size() - 2];
    if(m <= 4)
    {
        if (out_dtype == "bfloat16") {
            paddle::Tensor out =
                    paddle::empty({m, n}, paddle::DataType::BFLOAT16, x.place());
            GemmDequantParams params = {
                reinterpret_cast<const void*>(x.data<int8_t>()),
                reinterpret_cast<const void*>(y.data<int8_t>()),
                reinterpret_cast<const void*>(scale.data<float>()),
                reinterpret_cast<void*>(out.data<paddle::bfloat16>()),
                m,
                n,
                k,
                x.stream()
            };
            if (!int8_sq_kernel_launcher<int8_t, __nv_bfloat16>(params)) {
                PADDLE_THROW(common::errors::Fatal("gemm dequamt kernel run error"));
            }
            return {out};
        } else if (out_dtype == "float16") {
            paddle::Tensor out =
                    paddle::empty({m, n}, paddle::DataType::FLOAT16, x.place());
            GemmDequantParams params = {
                reinterpret_cast<const void*>(x.data<int8_t>()),
                reinterpret_cast<const void*>(y.data<int8_t>()),
                reinterpret_cast<const void*>(scale.data<float>()),
                reinterpret_cast<void*>(out.data<paddle::float16>()),
                m,
                n,
                k,
                x.stream()
            };
            if (!int8_sq_kernel_launcher<int8_t, half>(params)) {
                PADDLE_THROW(common::errors::Fatal("gemm dequamt kernel run error"));
            }
            return {out};
        } else {
            PADDLE_THROW(phi::errors::InvalidArgument(
                "only support bfloat16 and float16, but got %s", out_dtype));
        }
    }
    if (out_dtype == "bfloat16") {
        paddle::Tensor out =
            paddle::empty({m, n}, paddle::DataType::BFLOAT16, x.place());
        RunGemmDequant<paddle::DataType::BFLOAT16, paddle::bfloat16>(
            x.data<int8_t>(),
            y.data<int8_t>(),
            scale.data<float>(),
            out.data<paddle::bfloat16>(),
            m,
            k,
            n,
            x.stream());
        return {out};
    } else if (out_dtype == "float16") {
        paddle::Tensor out =
            paddle::empty({m, n}, paddle::DataType::FLOAT16, x.place());
        RunGemmDequant<paddle::DataType::FLOAT16, paddle::float16>(
            x.data<int8_t>(),
            y.data<int8_t>(),
            scale.data<float>(),
            out.data<paddle::float16>(),
            m,
            k,
            n,
            x.stream());
        return {out};
    } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "only support bfloat16 and float16, but got %s", out_dtype));
    }
}

std::vector<std::vector<int64_t>> GemmDequantShape(
    const std::vector<int64_t> &x,
    const std::vector<int64_t> &y,
    const std::vector<int64_t> &scale,
    const std::string &out_dtype) {
    return {{x[x.size() - 2], y[y.size() - 2]}};
}

std::vector<paddle::DataType> GemmDequantDtype(const paddle::DataType &x,
                                               const paddle::DataType &y,
                                               const paddle::DataType &scale,
                                               const std::string &out_dtype) {
    if (out_dtype == "bfloat16") {
        return {paddle::DataType::BFLOAT16};
    } else if (out_dtype == "float16") {
        return {paddle::DataType::FLOAT16};
    } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "only support bfloat16 and float16, but got %s", out_dtype));
    }
}

PD_BUILD_STATIC_OP(gemm_dequant)
    .Inputs({"x", "y", "scale"})
    .Outputs({"out"})
    .Attrs({"out_dtype: std::string"})
    .SetKernelFn(PD_KERNEL(GemmDequant))
    .SetInferShapeFn(PD_INFER_SHAPE(GemmDequantShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GemmDequantDtype));


namespace phi {

bool W8A8GroupGemmLauncher::validate_scale_dims(const W8A8GroupGemmParams& params) {
    if (params.num_groups == 0) return true;

    // Validate activation scales dimensions
    if (params.scale_dims_a.size() != 2) {
        return false; // Expected 2D scales for activations [groups, features]
    }

    // Validate weight scales dimensions
    if (params.scale_dims_b.size() != 2) {
        return false; // Expected 2D scales for weights [groups, features]
    }

    // Check consistency with problem sizes
    for (int i = 0; i < params.num_groups; ++i) {
        const auto& problem_size = params.problem_sizes[i];

        // Activation scales should match M dimension
        if (params.scale_dims_a[0] != params.num_groups ||
            params.scale_dims_a[1] != problem_size.m()) {
            return false;
        }

        // Weight scales should match N dimension
        if (params.scale_dims_b[0] != params.num_groups ||
            params.scale_dims_b[1] != problem_size.n()) {
            return false;
        }
    }

    return true;
}

size_t W8A8GroupGemmLauncher::get_workspace_size(const W8A8GroupGemmParams& params) {
    if (params.num_groups == 0) {
        return 0;
    }

    if (!validate_scale_dims(params)) {
        throw std::invalid_argument("Invalid scale dimensions for W8A8 Grouped GEMM");
    }

    // Create temporary GEMM object to query workspace size
    using GemmGrouped = W8A8GroupGemmLauncher::GemmGrouped;
    GemmGrouped gemm;

    // Prepare arguments with dual-scale support
    typename GemmGrouped::Arguments args(
        params.problem_sizes.data(),
        params.num_groups,
        1,  // threadblock_count (will be computed by GEMM)
        typename W8A8GroupGemmLauncher::DualScaleEpilogueOp::Params(1.0f, 0.0f),
        params.activations,
        params.weights,
        reinterpret_cast<const cutlass::bfloat16_t*>(params.outputs),
        params.outputs,
        params.leading_dimensions_A.data(),
        params.leading_dimensions_B.data(),
        params.leading_dimensions_C.data(),
        params.leading_dimensions_C.data()
    );

    return gemm.get_workspace_size(args);
}

cutlass::Status W8A8GroupGemmLauncher::launch(
    const W8A8GroupGemmParams& params,
    int multi_processor_count) {

    if (params.num_groups == 0) {
        return cutlass::Status::kSuccess;
    }

    if (!validate_scale_dims(params)) {
        return cutlass::Status::kErrorInvalidProblem;
    }

    using GemmGrouped = W8A8GroupGemmLauncher::GemmGrouped;
    GemmGrouped gemm;

    // Compute optimal threadblock count based on problem sizes and GPU resources
    int threadblock_count = GemmGrouped::sufficient(
        params.problem_sizes.data(),
        params.num_groups
    );

    threadblock_count = std::min(threadblock_count, multi_processor_count * 2);

    if (threadblock_count == 0) {
        return cutlass::Status::kErrorNotSupported;
    }

    // Prepare epilogue operator with dual-scale dequantization
    typename W8A8GroupGemmLauncher::DualScaleEpilogueOp::Params epilogue_op(1.0f, 0.0f);

    // Prepare GEMM arguments with dual-scale support
    typename GemmGrouped::Arguments args(
        params.problem_sizes.data(),
        params.num_groups,
        threadblock_count,
        epilogue_op,
        params.activations,
        params.weights,
        reinterpret_cast<const cutlass::bfloat16_t*>(params.outputs), // C matrix
        params.outputs,                                              // D matrix
        params.leading_dimensions_A.data(),
        params.leading_dimensions_B.data(),
        params.leading_dimensions_C.data(),
        params.leading_dimensions_C.data()
    );

    // Initialize GEMM
    cutlass::Status status = gemm.initialize(args, params.workspace);
    if (status != cutlass::Status::kSuccess) {
        return status;
    }

    // Run GEMM
    return gemm.run(params.stream);
}

// Paddle operator interface with dual-scale support
template <paddle::DataType D>
void RunW8A8GroupGemm(const int8_t* activations,
                      const int8_t* weights,
                      const float* dequant_scales_a,  // Activation scales
                      const float* dequant_scales_b,  // Weight scales
                      cutlass::bfloat16_t* outputs,
                      const std::vector<cutlass::gemm::GemmCoord>& problem_sizes,
                      const std::vector<int64_t>& ldA,
                      const std::vector<int64_t>& ldB,
                      const std::vector<int64_t>& ldC,
                      const std::vector<int64_t>& scale_dims_a,
                      const std::vector<int64_t>& scale_dims_b,
                      int num_groups,
                      cudaStream_t stream) {

    static_assert(D == paddle::DataType::BFLOAT16, "Only BF16 output is supported");

    // Prepare parameters with dual scales
    W8A8GroupGemmParams params;
    params.activations = activations;
    params.weights = weights;
    params.dequant_scales_a = dequant_scales_a;
    params.dequant_scales_b = dequant_scales_b;
    params.outputs = outputs;
    params.problem_sizes = problem_sizes;
    params.leading_dimensions_A = ldA;
    params.leading_dimensions_B = ldB;
    params.leading_dimensions_C = ldC;
    params.scale_dims_a = scale_dims_a;
    params.scale_dims_b = scale_dims_b;
    params.num_groups = num_groups;
    params.stream = stream;

    // Get workspace size and allocate
    size_t workspace_size = W8A8GroupGemmLauncher::get_workspace_size(params);
    int32_t* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
        params.workspace = workspace;
    }

    // Get GPU properties for multi-processor count
    int multi_processor_count = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    multi_processor_count = prop.multiProcessorCount;

    // Launch grouped GEMM with dual scales
    cutlass::Status status = W8A8GroupGemmLauncher::launch(params, multi_processor_count);

    // Cleanup workspace
    if (workspace) {
        cudaFree(workspace);
    }

    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("W8A8 Grouped GEMM with dual scales launch failed");
    }
}

}  // namespace phi

// Paddle operator wrapper
std::vector<paddle::Tensor> W8A8GroupGemmOp(
    const paddle::Tensor& activations,
    const paddle::Tensor& weights,
    const paddle::Tensor& dequant_scales_a,
    const paddle::Tensor& dequant_scales_b,
    const paddle::Tensor& problem_sizes_tensor) {

    // Extract tensor data
    const int8_t* act_data = activations.data<int8_t>();
    const int8_t* weight_data = weights.data<int8_t>();
    const float* scale_a_data = dequant_scales_a.data<float>();
    const float* scale_b_data = dequant_scales_b.data<float>();

    // Extract problem sizes from tensor
    const int64_t* problem_sizes_data = problem_sizes_tensor.data<int64_t>();
    int num_groups = problem_sizes_tensor.shape()[0];

    // Prepare problem sizes vector
    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    for (int i = 0; i < num_groups; ++i) {
        int64_t m = problem_sizes_data[i * 3];
        int64_t n = problem_sizes_data[i * 3 + 1];
        int64_t k = problem_sizes_data[i * 3 + 2];
        problem_sizes.emplace_back(m, n, k);
    }

    // Prepare leading dimensions (assuming packed layout)
    std::vector<int64_t> ldA, ldB, ldC;
    for (const auto& problem_size : problem_sizes) {
        ldA.push_back(problem_size.k());
        ldB.push_back(problem_size.k());
        ldC.push_back(problem_size.n());
    }

    // Prepare scale dimensions
    std::vector<int64_t> scale_dims_a = {num_groups, 1};  // [groups, features]
    std::vector<int64_t> scale_dims_b = {num_groups, 1};  // [groups, features]

    // Create output tensor
    int64_t total_m = 0, total_n = problem_sizes[0].n();
    for (const auto& problem_size : problem_sizes) {
        total_m += problem_size.m();
    }

    paddle::Tensor outputs = paddle::empty(
        {total_m, total_n},
        paddle::DataType::BFLOAT16,
        activations.place()
    );

    // Run grouped GEMM with dual scales
    phi::RunW8A8GroupGemm<paddle::DataType::BFLOAT16>(
        act_data,
        weight_data,
        scale_a_data,
        scale_b_data,
        reinterpret_cast<cutlass::bfloat16_t*>(outputs.data<paddle::bfloat16>()),
        problem_sizes,
        ldA,
        ldB,
        ldC,
        scale_dims_a,
        scale_dims_b,
        num_groups,
        activations.stream()
    );

    return {outputs};
}

std::vector<std::vector<int64_t>> W8A8GroupGemmShape(
    const std::vector<int64_t>& activations_shape,
    const std::vector<int64_t>& weights_shape,
    const std::vector<int64_t>& dequant_scales_a_shape,
    const std::vector<int64_t>& dequant_scales_b_shape,
    const std::vector<int64_t>& problem_sizes_shape) {

    // Output shape is determined by problem_sizes tensor
    // The first dimension is total M, second dimension is N
    int64_t total_m = 0;
    if (problem_sizes_shape.size() > 0) {
        // For simplicity, assume problem_sizes tensor contains [num_groups, 3] shape
        // where each row is [m, n, k]
        total_m = problem_sizes_shape[0];  // Sum of all m dimensions
    }

    return {{total_m, 1}};  // Placeholder, actual N dimension should be determined
}

std::vector<paddle::DataType> W8A8GroupGemmDtype(
    const paddle::DataType& activations_dtype,
    const paddle::DataType& weights_dtype,
    const paddle::DataType& dequant_scales_a_dtype,
    const paddle::DataType& dequant_scales_b_dtype,
    const paddle::DataType& problem_sizes_dtype) {

    return {paddle::DataType::BFLOAT16};
}

PD_BUILD_STATIC_OP(w8a8_group_gemm)
    .Inputs({"activations", "weights", "dequant_scales_a", "dequant_scales_b", "problem_sizes"})
    .Outputs({"outputs"})
    .SetKernelFn(PD_KERNEL(W8A8GroupGemmOp))
    .SetInferShapeFn(PD_INFER_SHAPE(W8A8GroupGemmShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(W8A8GroupGemmDtype));
