# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import w8a8_gemm


class TestW8A8GEMM(unittest.TestCase):
    def setUp(self):
        paddle.seed(0)
        self.tokens_per_group = 256
        self.N = 256
        self.K = 256
        self.BATCH = 1
        self.TokenPadding = 0

        tokens = [self.tokens_per_group] * self.BATCH
        self.tokens_prefix_sum = np.cumsum(tokens)

        self.tokens = paddle.to_tensor(tokens, dtype="int64")
        self.tokens_prefix_sum = paddle.to_tensor(self.tokens_prefix_sum, dtype="int64")
        self.all_tokens = int(self.tokens.sum())

        self.input_bf16 = paddle.randn([self.all_tokens, self.K], dtype="bfloat16") / 10

        self.weight = paddle.randn([self.BATCH, self.N, self.K], dtype="bfloat16") / 10
        self.weight_scale = (self.weight.abs().max(axis=-1) / 127).reshape([self.BATCH, self.N, 1])
        self.weight_quant = paddle.clip(self.weight / self.weight_scale, -127, 127).astype("int8")
        self.weight_dequant_scale = self.weight_scale.astype("float32")

        self.input_scale = (self.input_bf16.abs().max(axis=1) / 127).reshape([self.all_tokens, 1])
        self.input_quant = paddle.clip(self.input_bf16 / self.input_scale, -127, 127).astype("int8")
        self.input_dequant_scale = self.input_scale.astype("float32")

        self.max_tokens = int(self.tokens.max())

    def w8a8_gemm_naive(self, input_quant, weight_quant, tokens, input_dequant_scale, weight_dequant_scale):
        all_tokens = int(tokens.sum())
        out = paddle.zeros([all_tokens, self.N], dtype="bfloat16")
        pre_fix_token = 0

        for i in range(self.BATCH):
            input_dequant = (
                input_quant[pre_fix_token : pre_fix_token + tokens[i], :].astype("bfloat16")
                * input_dequant_scale[pre_fix_token : pre_fix_token + tokens[i], :]
            )
            weight_dequant = weight_quant[i].astype("bfloat16") * weight_dequant_scale[i]

            out_i = paddle.matmul(input_dequant, weight_dequant.astype("bfloat16"), transpose_y=True)
            out[pre_fix_token : pre_fix_token + tokens[i], :] = out_i
            pre_fix_token += tokens[i]

        return out

    def test_w8a8_gemm(self):
        out_naive = self.w8a8_gemm_naive(
            self.input_quant, self.weight_quant, self.tokens, self.input_dequant_scale, self.weight_dequant_scale
        )

        weight_int8 = self.weight_quant.astype("int8")
        input_scale_for_kernel = self.input_scale.astype("float32")  # 用于GPU内部反量化

        # 调用W8A8 GEMM（输入和权重都是int8）
        if self.TokenPadding == 0:
            out_cuda = w8a8_gemm(
                self.input_quant.cuda(),  # 输入已经是int8量化后的数据
                weight_int8.cuda(),
                self.tokens_prefix_sum,
                input_scale_for_kernel.cuda(),  # 传递激活的scale
                self.weight_dequant_scale.astype("float32"),  # 权重的scale
                int(self.TokenPadding),
                self.max_tokens,
                True,
            )
        else:
            out_cuda = w8a8_gemm(
                self.input_quant.cuda(),
                weight_int8.cuda(),
                self.tokens,
                input_scale_for_kernel.cuda(),
                self.weight_dequant_scale.astype("float32"),
                int(self.TokenPadding),
                self.max_tokens,
                True,
            )

        # 验证误差
        gap = (out_cuda - out_naive).abs()
        self.assertLess(float(gap.mean()), 0.07)  # 可能需要调整容忍误差


if __name__ == "__main__":
    unittest.main()
