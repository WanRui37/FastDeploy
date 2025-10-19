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

#include "helper.h"
#include "paddle/extension.h"
#include "w8a8_gemm_template.h"
#include "w8a8_gemm.h"

template <typename T> class NVTraits;

template <> class NVTraits<int8_t> {
public:
    typedef int8_t data_t;
};

template <> class NVTraits<__nv_bfloat16>{
public:
    typedef cutlass::bfloat16_t data_t;
};

template <typename OutputType>
void DisPatchW8A8Gemm(
        const int8_t* input,
        const int8_t* weight,
        const int64_t * tokens,
        const float * input_scale,
        const float * weight_scale,
        OutputType * out,
        const int64_t token_padding_size,
        const int64_t max_tokens,
        const int batch_size,
        const int64_t M,
        const int64_t K,
        cudaStream_t stream) {

    int kBlockN = 256;
    int TailN = 0;
    if constexpr (std::is_same_v<OutputType, cutlass::bfloat16_t>) {
        GEMM_SWITCH_BF16(
            M, K, batch_size, token_padding_size, kBlockN, TailN,
            weight,
            input,
            out,
            weight_scale,
            input_scale,
            tokens,
            max_tokens,
            stream)
    } else {
        PD_THROW("Only supported dtype in ['BFLOAT16'].");
    }
}

std::vector<paddle::Tensor> W8A8Gemm(
        const paddle::Tensor& input,
        const paddle::Tensor& weight,
        const paddle::Tensor& tokens,
        const paddle::Tensor& input_scale,
        const paddle::Tensor& weight_scale,
        const int64_t token_padding_size,
        const int64_t max_tokens,
        const bool is_bfloat16) {

    const int batch_size = weight.dims()[0];
    const int M = weight.dims()[1];
    const int K = weight.dims()[2];

    if (input.dtype() != paddle::DataType::INT8) {
        PD_THROW("Only supported dtype in ['INT8'].");
    }

    if (token_padding_size == 0) {
        const int all_tokens = input.dims()[0];
        if (is_bfloat16) {
            paddle::Tensor out = paddle::empty({all_tokens, M}, paddle::DataType::BFLOAT16, input.place());
            phi::dtype::bfloat16 *out_data = out.data<phi::dtype::bfloat16>();
            DisPatchW8A8Gemm(
                input.data<int8_t>(),
                weight.data<int8_t>(),
                tokens.data<int64_t>(),
                input_scale.data<float>(),
                weight_scale.data<float>(),
                reinterpret_cast<cutlass::bfloat16_t*>(out_data),
                token_padding_size,
                max_tokens,
                batch_size,
                M,
                K,
                input.stream());
            return {out};
        } else {
            PD_THROW("Only supported dtype in ['BFLOAT16'].");
        }
    } else {
        if (is_bfloat16) {
            paddle::Tensor out = paddle::empty({batch_size, token_padding_size, M}, paddle::DataType::BFLOAT16, input.place());
            phi::dtype::bfloat16 * out_data = out.data<phi::dtype::bfloat16>();
            DisPatchW8A8Gemm(
                input.data<int8_t>(),
                weight.data<int8_t>(),
                tokens.data<int64_t>(),
                input_scale.data<float>(),
                weight_scale.data<float>(),
                reinterpret_cast<cutlass::bfloat16_t*>(out_data),
                token_padding_size,
                max_tokens,
                batch_size,
                M,
                K,
                input.stream());
            return {out};
        } else {
            PD_THROW("Only supported dtype in ['BFLOAT16'].");
        }
    }
}

template <typename InputType, typename OutputType>
void DisPatchW8A8GemmWrapper(
        const InputType* input,
        const InputType* weight,
        const int64_t* total_rows_before_expert,
        const float* input_scale,
        const float* weight_scale,
        OutputType * out,
        const int64_t token_padding_size,
        const int64_t max_tokens,
        const int num_experts,
        const int64_t M,
        const int64_t K,
        cudaStream_t stream) {
    using InType = typename NVTraits<InputType>::data_t;
    using OutType = typename NVTraits<OutputType>::data_t;
    DisPatchW8A8Gemm(
        reinterpret_cast<const InType*>(input),
        reinterpret_cast<const InType*>(weight),
        total_rows_before_expert,
        input_scale,
        weight_scale,
        reinterpret_cast<OutType*>(out),
        token_padding_size,
        max_tokens,
        num_experts,
        M,
        K,
        stream);
}

PD_BUILD_STATIC_OP(w8a8_gemm)
    .Inputs({"input",
             "weight",
             "tokens",
             "input_scale",
             "weight_scale"})
    .Outputs({"out"})
    .Attrs({"token_padding_size: int64_t",
            "max_tokens: int64_t",
            "is_bfloat16: bool"})
    .SetKernelFn(PD_KERNEL(W8A8Gemm));
