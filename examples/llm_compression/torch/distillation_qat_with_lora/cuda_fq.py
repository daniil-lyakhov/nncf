# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
CUDA extension-based fused fake quantization for QAT with LoRA.

Uses NNCF's compiled CUDA extensions (Quantize_forward / Quantize_backward)
instead of Triton kernels. The LoRA correction (B @ A matmul) remains in PyTorch.

Compared to the default LoRA quantize path which uses the pure-Python reference
implementation (ReferenceQuantizedFunctions), this routes through the optimized
CUDA C++ kernels for both forward and backward.
"""

import types

import torch
from torch import Tensor, nn

from nncf.torch.quantization.extensions import QuantizedFunctionsCUDA
from nncf.torch.quantization.quantize_functions import TuneRange


class CudaFakeQuantizeAsymmetric(torch.autograd.Function):
    """
    Autograd wrapper using NNCF CUDA extensions for asymmetric FQ.

    Forward: TuneRange (Python) + CUDA Quantize_forward
    Backward: CUDA Quantize_backward (returns grad_input, grad_low, grad_range)
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        input_low: Tensor,
        input_range: Tensor,
        level_low: int,
        level_high: int,
        levels: int,
    ) -> Tensor:
        # TuneRange adjusts input_low/input_range for zero-point alignment
        input_low_tuned, input_range_tuned = TuneRange.apply(input_low, input_range, levels)

        # The CUDA kernel expects input_low/input_range to be 1D (flat) with
        # size matching input.size(0). Reshape x to 2D (num_groups, group_size).
        original_shape = x.shape
        num_groups = input_low_tuned.numel()
        group_size = x.numel() // num_groups
        x_2d = x.reshape(num_groups, group_size).contiguous()
        il_flat = input_low_tuned.reshape(-1).contiguous()
        ir_flat = input_range_tuned.reshape(-1).contiguous()

        # Cast per-group params to match input dtype (needed for fp16/bf16)
        if x_2d.dtype in [torch.bfloat16, torch.float16]:
            il_flat = il_flat.type(x_2d.dtype)
            ir_flat = ir_flat.type(x_2d.dtype)

        output = QuantizedFunctionsCUDA.get("Quantize_forward")(x_2d, il_flat, ir_flat, levels)

        ctx.save_for_backward(x_2d, il_flat, ir_flat)
        ctx.levels = levels
        ctx.level_low = level_low
        ctx.level_high = level_high
        ctx.original_shape = original_shape
        ctx.input_low_shape = input_low.shape
        ctx.input_range_shape = input_range.shape

        return output.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
        x_2d, input_low, input_range = ctx.saved_tensors
        levels = ctx.levels
        level_low = ctx.level_low
        level_high = ctx.level_high

        grad_2d = grad_output.reshape(x_2d.shape).contiguous()

        grad_input, grad_low, grad_range = QuantizedFunctionsCUDA.get("Quantize_backward")(
            grad_2d, x_2d, input_low, input_range, levels, level_low, level_high
        )

        return (
            grad_input.reshape(ctx.original_shape),
            grad_low.reshape(ctx.input_low_shape),
            grad_range.reshape(ctx.input_range_shape),
            None, None, None,
        )


class CudaFakeQuantizeSymmetric(torch.autograd.Function):
    """
    Autograd wrapper using NNCF CUDA extensions for symmetric FQ.

    Forward: Compute input_low/input_range from scale + TuneRange + CUDA Quantize_forward
    Backward: CUDA Quantize_backward (returns grad_input, grad_scale)
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        input_low: Tensor,
        input_range: Tensor,
        level_low: int,
        level_high: int,
        levels: int,
    ) -> Tensor:
        # TuneRange adjusts input_low/input_range for zero-point alignment
        input_low_tuned, input_range_tuned = TuneRange.apply(input_low, input_range, levels)

        # The CUDA kernel expects input_low/input_range to be 1D (flat) with
        # size matching input.size(0). Reshape x to 2D (num_groups, group_size).
        original_shape = x.shape
        num_groups = input_low_tuned.numel()
        group_size = x.numel() // num_groups
        x_2d = x.reshape(num_groups, group_size).contiguous()
        il_flat = input_low_tuned.reshape(-1).contiguous()
        ir_flat = input_range_tuned.reshape(-1).contiguous()

        if x_2d.dtype in [torch.bfloat16, torch.float16]:
            il_flat = il_flat.type(x_2d.dtype)
            ir_flat = ir_flat.type(x_2d.dtype)

        output = QuantizedFunctionsCUDA.get("Quantize_forward")(x_2d, il_flat, ir_flat, levels)

        ctx.save_for_backward(x_2d, il_flat, ir_flat)
        ctx.levels = levels
        ctx.level_low = level_low
        ctx.level_high = level_high
        ctx.original_shape = original_shape
        ctx.input_low_shape = input_low.shape
        ctx.input_range_shape = input_range.shape

        return output.reshape(original_shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
        x_2d, input_low, input_range = ctx.saved_tensors
        levels = ctx.levels
        level_low = ctx.level_low
        level_high = ctx.level_high

        grad_2d = grad_output.reshape(x_2d.shape).contiguous()

        grad_input, grad_low, grad_range = QuantizedFunctionsCUDA.get("Quantize_backward")(
            grad_2d, x_2d, input_low, input_range, levels, level_low, level_high
        )

        return (
            grad_input.reshape(ctx.original_shape),
            grad_low.reshape(ctx.input_low_shape),
            grad_range.reshape(ctx.input_range_shape),
            None, None, None,
        )


def _make_cuda_asym_quantize(self, x: Tensor, execute_traced_op_as_identity: bool = False) -> Tensor:
    """CUDA extension asymmetric quantize replacement for AsymmetricLoraQuantizer."""
    if execute_traced_op_as_identity:
        return x

    eps = self.eps
    input_range_safe = abs(self.input_range) + eps
    input_low = self.input_low

    # LoRA correction (matmul stays in PyTorch)
    x = (x + self.lora_B @ self.lora_A).to(x.dtype)

    # Reshape for group quantization
    input_shape = self._lspec.weight_shape
    original_shape = x.shape
    x = x.reshape(input_shape)

    # Run CUDA FQ (TuneRange + CUDA extension forward/backward)
    output = CudaFakeQuantizeAsymmetric.apply(
        x, input_low, input_range_safe, self.level_low, self.level_high, self.levels
    )

    return output.reshape(original_shape).to(x.dtype)


def _make_cuda_sym_quantize(self, x: Tensor, execute_traced_op_as_identity: bool = False) -> Tensor:
    """CUDA extension symmetric quantize replacement for SymmetricLoraQuantizer."""
    if execute_traced_op_as_identity:
        return x

    eps = self.eps
    scale = self.scale
    scale_safe = torch.where(torch.abs(scale) < eps, eps, scale)
    level_low = self.level_low
    level_high = self.level_high
    levels = self.levels

    # LoRA correction (matmul stays in PyTorch)
    x = (x + self.lora_B @ self.lora_A).to(x.dtype)

    # Reshape for group quantization
    input_shape = self._lspec.weight_shape
    original_shape = x.shape
    x = x.reshape(input_shape)

    # Compute input_low and input_range from scale (symmetric formula)
    input_low = torch.where(scale_safe > 0, -scale_safe, -scale_safe / level_low * level_high)
    input_range = torch.abs((2 + 1 / level_low) * scale_safe)

    # Run CUDA FQ (TuneRange + CUDA extension forward/backward)
    output = CudaFakeQuantizeSymmetric.apply(
        x, input_low, input_range, level_low, level_high, levels
    )

    return output.reshape(original_shape).to(x.dtype)


def replace_quantizers_with_cuda(model: nn.Module, nncf_modules: dict) -> None:
    """
    Replace all NNCF LoRA quantizer modules' quantize() method with the CUDA extension version.

    This routes through the compiled NNCF CUDA kernels instead of the pure-Python reference
    implementation that the LoRA quantize path normally uses.

    :param model: The model with NNCF hook storage containing quantizers.
    :param nncf_modules: Dictionary of NNCF imports (from get_nncf_imports).
    """
    get_hook_storage = nncf_modules["get_hook_storage"]
    AsymmetricLoraQuantizer = nncf_modules["AsymmetricLoraQuantizer"]
    SymmetricLoraQuantizer = nncf_modules["SymmetricLoraQuantizer"]

    hook_storage = get_hook_storage(model)
    replaced = 0
    for _, module in hook_storage.named_hooks():
        if isinstance(module, AsymmetricLoraQuantizer):
            module = module.to("cuda:0")
            module.quantize = types.MethodType(_make_cuda_asym_quantize, module)
            replaced += 1
        elif isinstance(module, SymmetricLoraQuantizer):
            module = module.to("cuda:0")
            module.quantize = types.MethodType(_make_cuda_sym_quantize, module)
            replaced += 1

    print(f"  Replaced {replaced} quantizers with CUDA extension FQ (NNCF compiled kernels)")
