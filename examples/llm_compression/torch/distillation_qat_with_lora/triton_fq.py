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

Fuses: TuneRange + clamp + scale + round + dequant into a single GPU kernel,
eliminating 6-8 intermediate tensor allocations per quantizer call and the
TuneRange autograd.Function overhead.

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
    Fused TuneRange + fake quantize kernel operating per-group.

    TuneRange ensures zero-point alignment (floating point zero maps exactly to
    a quantization level). Then FQ applies: clamp + scale + round + dequant.

    For each element x[i]:
        group_idx = i // group_size
        il, ir = tune_range(input_low[group_idx], input_range[group_idx], levels)
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
    il_raw = tl.load(input_low_ptr + group_idx, mask=mask)
    ir_raw = tl.load(input_range_ptr + group_idx, mask=mask)

    # --- TuneRange: adjust input_low/input_range for zero-point alignment ---
    ih_raw = ir_raw + il_raw
    # Clamp: input_low_copy = min(input_low, 0), input_high = max(input_high, 0)
    il_clamped = tl.where(il_raw > 0.0, 0.0, il_raw)
    ih_clamped = tl.where(ih_raw < 0.0, 0.0, ih_raw)

    n = levels - 1
    tr_scale = n / (ih_clamped - il_clamped)
    zp_tr = tl.extra.cuda.libdevice.round((-il_clamped) * tr_scale)

    # Compute candidate new bounds
    new_il = tl.where(zp_tr < n, zp_tr / (zp_tr - n) * ih_clamped, il_clamped)
    new_ih = tl.where(zp_tr > 0.0, (zp_tr - n) / zp_tr * il_clamped, ih_clamped)

    # Pick the wider range
    range_1 = ih_clamped - new_il
    range_2 = new_ih - il_clamped
    use_range_1 = range_1 > range_2

    il = tl.where(use_range_1, new_il, il_clamped)
    ir = tl.where(use_range_1, ih_clamped - new_il, new_ih - il_clamped)

    # --- Fake quantize: clamp -> scale -> round -> dequant ---
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


@triton.jit
def _fused_fq_backward_kernel(
    # Inputs
    grad_output_ptr,
    x_ptr,
    input_low_ptr,
    input_range_ptr,
    # Outputs
    grad_input_ptr,
    grad_low_ptr,
    grad_range_ptr,
    # Scalars
    levels: tl.constexpr,
    level_low: tl.constexpr,
    level_high: tl.constexpr,
    group_size: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Backward kernel for fused TuneRange + FQ.

    Computes:
      grad_input[i] = grad_output[i] * mask_in[i]  (zero for clamped elements)
      grad_low[g] = sum_over_group(grad_output * (mask_hi + mask_lo))
      grad_range[g] = sum_over_group(grad_output * (err*mask_in + sign(ir)*(level_low/level_high)*mask_lo + mask_hi))
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load inputs
    grad_out = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Load per-group params (already tuned)
    group_idx = offsets // group_size
    il = tl.load(input_low_ptr + group_idx, mask=mask, other=0.0)
    ir = tl.load(input_range_ptr + group_idx, mask=mask, other=0.0)

    # Compute masks
    ih = il + ir
    mask_hi = (x > ih).to(tl.float32)
    mask_lo = (x < il).to(tl.float32)
    mask_in = 1.0 - mask_hi - mask_lo

    # grad_input: STE with clamping mask
    grad_input = grad_out * mask_in
    tl.store(grad_input_ptr + offsets, grad_input, mask=mask)

    # Recompute FQ output for err calculation
    n = levels - 1
    scale = n / ir
    clamped = tl.where(x < il, il, x)
    clamped = tl.where(clamped > ih, ih, clamped)
    zero_point = tl.extra.cuda.libdevice.round((-il) * scale)
    q = (clamped - il) * scale - zero_point
    q = tl.extra.cuda.libdevice.round(q)
    out = q / scale

    # err = (output - input) / input_range
    err = (out - x) / ir

    # Per-element contributions to grad_low and grad_range
    alpha = level_low / level_high
    range_sign = tl.where(ir >= 0.0, 1.0, -1.0)

    grad_low_contrib = grad_out * (mask_hi + mask_lo)
    grad_range_contrib = grad_out * (err * mask_in + range_sign * alpha * mask_lo + mask_hi)

    # Atomic add to per-group accumulators
    tl.atomic_add(grad_low_ptr + group_idx, grad_low_contrib, mask=mask)
    tl.atomic_add(grad_range_ptr + group_idx, grad_range_contrib, mask=mask)


def _tune_range_forward(input_low: Tensor, input_range: Tensor, levels: int) -> tuple[Tensor, Tensor]:
    """
    Python reimplementation of TuneRange.forward for use in backward pass.
    Adjusts input_low/input_range so zero maps exactly to a quantization level.
    """
    input_high = input_range + input_low
    input_low_copy = input_low.clone()
    input_low_copy[input_low_copy > 0] = 0
    input_high[input_high < 0] = 0
    n = levels - 1
    scale = (n / (input_high - input_low_copy)).to(dtype=input_high.dtype)
    zp = torch.round(-input_low_copy * scale)

    new_input_low = torch.where(zp < n, zp / (zp - n) * input_high, input_low_copy)
    new_input_high = torch.where(zp > 0.0, (zp - n) / zp * input_low_copy, input_high)

    range_1 = input_high - new_input_low
    range_2 = new_input_high - input_low_copy

    mask = (range_1 > range_2).to(input_high.dtype)
    inv_mask = (1 - mask).abs()

    new_input_low = mask * new_input_low + inv_mask * input_low_copy
    new_input_range = inv_mask * new_input_high + mask * input_high - new_input_low

    return new_input_low, new_input_range


class FusedFakeQuantize(torch.autograd.Function):
    """
    Autograd wrapper: forward runs the Triton FQ kernel (with TuneRange fused),
    backward runs a Triton backward kernel matching QuantizeAsymmetric/SymmetricTorch.
    """

    @staticmethod
    def forward(
        ctx, x: Tensor, input_low: Tensor, input_range: Tensor, levels: int, level_low: int, level_high: int, group_size: int
    ) -> Tensor:
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

        # Save for backward
        ctx.save_for_backward(x_flat, input_low_flat, input_range_flat)
        ctx.levels = levels
        ctx.level_low = level_low
        ctx.level_high = level_high
        ctx.group_size = group_size
        ctx.original_shape = x.shape

        return output.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None, None, None]:
        x_flat, input_low_raw, input_range_raw = ctx.saved_tensors
        levels = ctx.levels
        level_low = ctx.level_low
        level_high = ctx.level_high
        group_size = ctx.group_size

        # Recompute TuneRange on per-group tensors (cheap, small tensors)
        tuned_low, tuned_range = _tune_range_forward(input_low_raw, input_range_raw, levels)

        grad_flat = grad_output.reshape(-1).contiguous()
        n_elements = x_flat.numel()
        num_groups = n_elements // group_size

        # Allocate outputs
        grad_input = torch.empty_like(x_flat)
        grad_low = torch.zeros(num_groups, dtype=torch.float32, device=x_flat.device)
        grad_range = torch.zeros(num_groups, dtype=torch.float32, device=x_flat.device)

        BLOCK_SIZE = 1024
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        _fused_fq_backward_kernel[grid](
            grad_flat,
            x_flat,
            tuned_low,
            tuned_range,
            grad_input,
            grad_low,
            grad_range,
            levels=levels,
            level_low=level_low,
            level_high=level_high,
            group_size=group_size,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Reshape grad_low and grad_range to match input_low/input_range shape
        grad_low = grad_low.to(grad_output.dtype)
        grad_range = grad_range.to(grad_output.dtype)

        return grad_input.reshape(ctx.original_shape), grad_low, grad_range, None, None, None, None


def triton_fake_quantize(
    x: Tensor, input_low: Tensor, input_range: Tensor, levels: int, level_low: int, level_high: int, group_size: int
) -> Tensor:
    """
    Apply fused fake quantization (TuneRange + FQ) using Triton with proper backward.

    :param x: Input tensor (reshaped to weight_shape for group quant).
    :param input_low: Per-group lower clamp bound, shape (num_groups, 1) or broadcastable.
    :param input_range: Per-group quantization range, shape (num_groups, 1) or broadcastable.
    :param levels: Number of quantization levels (e.g. 16 for 4-bit).
    :param level_low: Lowest quantization level (e.g. 0 for asymmetric, -8 for symmetric).
    :param level_high: Highest quantization level (e.g. 15 for asymmetric, 7 for symmetric).
    :param group_size: Elements per quantization group.
    :return: Fake-quantized tensor with proper gradients for input, input_low, and input_range.
    """
    return FusedFakeQuantize.apply(x, input_low, input_range, levels, level_low, level_high, group_size)


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

    # Run fused Triton FQ (TuneRange + FQ with proper backward)
    output_dq = triton_fake_quantize(
        x, input_low, input_range_safe, self.levels, self.level_low, self.level_high, group_size
    )

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

    # Run fused Triton FQ (TuneRange + FQ with proper backward)
    output_dq = triton_fake_quantize(x, input_low, input_range, levels, level_low, level_high, group_size)

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
