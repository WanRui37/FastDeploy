# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import nn


def scaled_int8_quant(x: paddle.Tensor, use_per_token_if_dynamic: bool = False):
    """
    Quantize input tensor to int8 with scaling.
    
    Args:
        x (paddle.Tensor): Input tensor to quantize.
        use_per_token_if_dynamic (bool): Whether to use per-token quantization for dynamic tensors.
    
    Returns:
        tuple: Quantized tensor (int8) and scale tensor (float32).
    """
    if use_per_token_if_dynamic and len(x.shape) == 2:
        # Per-token quantization for dynamic tensors
        scale = x.abs().max(axis=1, keepdim=True)  # [B, 1]
        max_bound = 127.0
        quant = x / scale * max_bound
        quant = paddle.round(quant).astype("int8")
        scale = scale / max_bound
    else:
        # Static quantization
        scale = x.abs().max()  # scalar
        max_bound = 127.0
        quant = x / scale * max_bound
        quant = paddle.round(quant).astype("int8")
        scale = scale / max_bound
    
    return quant, scale