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
Level 1 Triton kernel for fused fake quantization in QAT with LoRA.

Fuses: clamp + scale + round + dequant into a single GPU kernel,
eliminating 6-8 intermediate tensor allocations per quantizer call.

The LoRA correction (B @ A matmul) remains in PyTorch since it's not element-wise.
"""

import types

import torch
import triton
import triton.language as tl
from torch import Tensor, nn


@triton.jit
def _fused_fake_quantize_kernel(
    # Pointers
    x_ptr,
    input_low_ptr,
    input_range_ptr,
    output_ptr,
    # Scalar
    levels: tl.constexpr,
    # Strides
    group_size: tl.constexpr,
    # Total number of elements
    n_elements,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused fake quantize kernel operating per-group.

    For each element x[i]:
        group_idx = i // group_size
        il = input_low[group_idx]
        ir = input_range[group_idx]
        scale = (levels - 1) / ir
        clamped = clamp(x[i], il, il + ir)
        zp = round(-il * scale)
        q = round((clamped - il) * scale - zp)
        out = q / scale
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute group index for each element to look up per-group params
    group_idx = offsets // group_size

    # Load per-group input_low and input_range (broadcast within group)
    il = tl.load(input_low_ptr + group_idx, mask=mask)
    ir = tl.load(input_range_ptr + group_idx, mask=mask)

    # Fake quantize: clamp -> scale -> round -> dequant
    n = levels - 1
    scale = n / ir
    ih = il + ir

    # Clamp
    clamped = tl.where(x < il, il, x)
    clamped = tl.where(clamped > ih, ih, clamped)

    # Quantize
    zero_point = tl.extra.cuda.libdevice.round((-il) * scale)
    q = (clamped - il) * scale - zero_point
    q = tl.extra.cuda.libdevice.round(q)

    # Dequantize
    out = q / scale

    tl.store(output_ptr + offsets, out, mask=mask)


class FusedFakeQuantizeSTE(torch.autograd.Function):
    """
    Autograd wrapper: forward runs the Triton kernel, backward is STE (identity).
    """

    @staticmethod
    def forward(ctx, x: Tensor, input_low: Tensor, input_range: Tensor, levels: int, group_size: int) -> Tensor:
        # Flatten for kernel launch
        x_flat = x.reshape(-1)
        n_elements = x_flat.numel()
        output = torch.empty_like(x_flat)

        # input_low and input_range should be contiguous 1D with shape (num_groups,)
        input_low_flat = input_low.reshape(-1).contiguous()
        input_range_flat = input_range.reshape(-1).contiguous()

        BLOCK_SIZE = 1024
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _fused_fake_quantize_kernel[grid](
            x_flat,
            input_low_flat,
            input_range_flat,
            output,
            levels=levels,
            group_size=group_size,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x_flat, output)
        ctx.original_shape = x.shape
        return output.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None, None, None, None]:
        # STE: pass gradient through unchanged
        return grad_output, None, None, None, None


def triton_fake_quantize(x: Tensor, input_low: Tensor, input_range: Tensor, levels: int, group_size: int) -> Tensor:
    """
    Apply fused fake quantization using the Triton kernel with STE backward.

    :param x: Input tensor (reshaped to weight_shape for group quant).
    :param input_low: Per-group lower clamp bound, shape (num_groups, 1) or broadcastable.
    :param input_range: Per-group quantization range, shape (num_groups, 1) or broadcastable.
    :param levels: Number of quantization levels (e.g. 16 for 4-bit).
    :param group_size: Elements per quantization group.
    :return: Fake-quantized tensor with STE gradients.
    """
    return FusedFakeQuantizeSTE.apply(x, input_low, input_range, levels, group_size)


def _make_triton_asym_quantize(self, x: Tensor, execute_traced_op_as_identity: bool = False) -> Tensor:
    """Triton-fused asymmetric quantize replacement for AsymmetricLoraQuantizer."""
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

    # Determine group_size from weight_shape (last dim is the group size)
    group_size = input_shape[-1]

    # Run fused Triton FQ with STE
    output_dq = triton_fake_quantize(x, input_low, input_range_safe, self.levels, group_size)

    # STE: forward uses quantized values, backward passes through x
    output_dq = x + (output_dq - x).detach()

    return output_dq.reshape(original_shape).to(x.dtype)


def _make_triton_sym_quantize(self, x: Tensor, execute_traced_op_as_identity: bool = False) -> Tensor:
    """Triton-fused symmetric quantize replacement for SymmetricLoraQuantizer."""
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

    # Determine group_size from weight_shape (last dim is the group size)
    group_size = input_shape[-1]

    # Run fused Triton FQ with STE
    output_dq = triton_fake_quantize(x, input_low, input_range, levels, group_size)

    # STE: forward uses quantized values, backward passes through x
    output_dq = x + (output_dq - x).detach()

    return output_dq.reshape(original_shape).to(x.dtype)


def replace_quantizers_with_triton(model: nn.Module, nncf_modules: dict) -> None:
    """
    Replace all NNCF LoRA quantizer modules' quantize() method with the Triton-fused version.

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
            module.quantize = types.MethodType(_make_triton_asym_quantize, module)
            replaced += 1
        elif isinstance(module, SymmetricLoraQuantizer):
            module.quantize = types.MethodType(_make_triton_sym_quantize, module)
            replaced += 1

    print(f"  Replaced {replaced} quantizers with Triton-fused FQ (Level 1: clamp+scale+round+dequant)")
