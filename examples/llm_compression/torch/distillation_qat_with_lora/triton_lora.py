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
Level 2 Triton kernel for fused LoRA correction + fake quantization.

Fuses: (x + B @ A) + clamp + scale + round + dequant into a single GPU kernel,
eliminating the full (M x N) intermediate tensor from the LoRA matmul correction
AND the FQ intermediates (6-8 tensors) per quantizer call.

Compared to Level 1 (triton_fq.py):
  - Level 1: PyTorch matmul B@A → add → [Triton FQ kernel]
  - Level 2: [Triton fused matmul+add+FQ kernel] — one kernel, zero intermediates

The backward pass uses STE for quantization (identity gradient) and standard matmul
gradients for LoRA parameters (kept in PyTorch for simplicity).
"""

import types

import torch
import triton
import triton.language as tl
from torch import Tensor, nn


@triton.jit
def _fused_lora_fq_kernel(
    # Data pointers
    x_ptr,
    lora_b_ptr,
    lora_a_ptr,
    input_low_ptr,
    input_range_ptr,
    output_ptr,
    # Matrix dimensions: x is (M, N), B is (M, K), A is (K, N)
    M,
    N,
    K,
    # Strides for x (row-major)
    stride_xm,
    stride_xn,
    # Strides for B
    stride_bm,
    stride_bk,
    # Strides for A
    stride_ak,
    stride_an,
    # Strides for output
    stride_om,
    stride_on,
    # Quantization params
    levels: tl.constexpr,
    group_size: tl.constexpr,
    # Number of groups per row: N // group_size
    groups_per_row,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused LoRA matmul + fake quantize kernel.

    For each output tile (bm, bn):
      1. Load x[bm, bn] tile
      2. Compute LoRA correction: accumulate B[bm, :] @ A[:, bn] over K dimension
      3. x_corrected = x + correction
      4. Apply per-group FQ: clamp + scale + round + dequant
      5. Store result
    """
    # Program ID maps to a 2D tile of the output
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Masks for boundary tiles
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_mn = mask_m[:, None] & mask_n[None, :]

    # Load x tile: shape (BLOCK_M, BLOCK_N)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x_tile = tl.load(x_ptrs, mask=mask_mn, other=0.0)

    # Compute LoRA correction via tiled matmul: B[bm, :] @ A[:, bn]
    # Accumulate in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_k = k_offs < K

        # Load B tile: (BLOCK_M, BLOCK_K)
        b_ptrs = lora_b_ptr + offs_m[:, None] * stride_bm + k_offs[None, :] * stride_bk
        b_tile = tl.load(b_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load A tile: (BLOCK_K, BLOCK_N)
        a_ptrs = lora_a_ptr + k_offs[:, None] * stride_ak + offs_n[None, :] * stride_an
        a_tile = tl.load(a_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate partial matmul
        acc += tl.dot(b_tile, a_tile)

    # Cast accumulator to match x dtype and add LoRA correction
    x_corrected = x_tile + acc.to(x_tile.dtype)

    # --- Fake Quantize (per-group) ---
    # Compute group index for each element in the tile
    # group_idx = row * groups_per_row + col // group_size
    group_indices = offs_m[:, None] * groups_per_row + offs_n[None, :] // group_size

    # Load per-group input_low and input_range
    il = tl.load(input_low_ptr + group_indices, mask=mask_mn, other=0.0)
    ir = tl.load(input_range_ptr + group_indices, mask=mask_mn, other=1.0)

    # FQ formula: clamp -> scale -> round -> dequant
    n_levels = levels - 1
    scale = n_levels / ir
    ih = il + ir

    # Clamp
    clamped = tl.where(x_corrected < il, il, x_corrected)
    clamped = tl.where(clamped > ih, ih, clamped)

    # Quantize
    zero_point = tl.extra.cuda.libdevice.round((-il) * scale)
    q = (clamped - il) * scale - zero_point
    q = tl.extra.cuda.libdevice.round(q)

    # Dequantize
    out = q / scale

    # Store output tile
    out_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask_mn)


class FusedLoRAFakeQuantizeSTE(torch.autograd.Function):
    """
    Autograd wrapper for the fused LoRA + FQ kernel.

    Forward: single Triton kernel computes (x + B@A) then FQ in one pass.
    Backward: STE for quantization (identity on x_corrected), standard matmul grads for B and A.
    """

    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        lora_b: Tensor,
        lora_a: Tensor,
        input_low: Tensor,
        input_range: Tensor,
        levels: int,
        group_size: int,
        weight_shape: tuple[int, ...],
    ) -> Tensor:
        # x comes in flat or 2D; reshape to weight_shape (M, N) for the kernel
        M, N = weight_shape[0], weight_shape[1]
        K = lora_b.shape[1]  # LoRA rank

        x_2d = x.reshape(M, N).contiguous()
        lora_b_c = lora_b.contiguous()
        lora_a_c = lora_a.contiguous()
        input_low_flat = input_low.reshape(-1).contiguous()
        input_range_flat = input_range.reshape(-1).contiguous()

        output = torch.empty_like(x_2d)

        groups_per_row = N // group_size

        # Determine block sizes
        BLOCK_M = 32
        BLOCK_N = min(64, triton.next_power_of_2(group_size))
        BLOCK_K = min(64, triton.next_power_of_2(K))

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        _fused_lora_fq_kernel[grid](
            x_2d,
            lora_b_c,
            lora_a_c,
            input_low_flat,
            input_range_flat,
            output,
            M=M,
            N=N,
            K=K,
            stride_xm=x_2d.stride(0),
            stride_xn=x_2d.stride(1),
            stride_bm=lora_b_c.stride(0),
            stride_bk=lora_b_c.stride(1),
            stride_ak=lora_a_c.stride(0),
            stride_an=lora_a_c.stride(1),
            stride_om=output.stride(0),
            stride_on=output.stride(1),
            levels=levels,
            group_size=group_size,
            groups_per_row=groups_per_row,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )

        # Save for backward: need x_corrected for STE grad, and B/A for LoRA grads
        # We don't save x_corrected explicitly — recompute in backward from saved tensors
        ctx.save_for_backward(lora_b, lora_a)
        ctx.weight_shape = weight_shape
        ctx.original_shape = x.shape

        return output.reshape(x.shape)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor, Tensor, None, None, None, None, None]:
        lora_b, lora_a = ctx.saved_tensors
        M, N = ctx.weight_shape

        # STE: gradient passes through the quantization unchanged
        grad_x = grad_output

        # LoRA parameter gradients (standard matmul backward)
        grad_2d = grad_output.reshape(M, N)
        # grad_B = grad_output @ A^T, shape (M, K)
        grad_b = grad_2d @ lora_a.T
        # grad_A = B^T @ grad_output, shape (K, N)
        grad_a = lora_b.T @ grad_2d

        return grad_x, grad_b, grad_a, None, None, None, None, None


def triton_lora_fake_quantize(
    x: Tensor,
    lora_b: Tensor,
    lora_a: Tensor,
    input_low: Tensor,
    input_range: Tensor,
    levels: int,
    group_size: int,
    weight_shape: tuple[int, ...],
) -> Tensor:
    """
    Fused LoRA correction + fake quantization using a single Triton kernel.

    :param x: Weight tensor (flat or 2D).
    :param lora_b: LoRA B matrix, shape (out_features, rank).
    :param lora_a: LoRA A matrix, shape (rank, in_features).
    :param input_low: Per-group lower bound, shape (num_groups, 1) or (num_groups,).
    :param input_range: Per-group quantization range, shape (num_groups, 1) or (num_groups,).
    :param levels: Number of quantization levels.
    :param group_size: Elements per quantization group.
    :param weight_shape: 2D shape (M, N) of the weight for group indexing.
    :return: Fake-quantized tensor with STE and LoRA gradients.
    """
    return FusedLoRAFakeQuantizeSTE.apply(x, lora_b, lora_a, input_low, input_range, levels, group_size, weight_shape)


def _make_triton_lora_asym_quantize(self, x: Tensor, execute_traced_op_as_identity: bool = False) -> Tensor:
    """Level 2 Triton-fused asymmetric LoRA + FQ replacement."""
    if execute_traced_op_as_identity:
        return x
    eps = self.eps
    input_range_safe = abs(self.input_range) + eps
    input_low = self.input_low

    input_shape = self._lspec.weight_shape
    original_shape = x.shape
    group_size = input_shape[-1]

    # weight_shape for the kernel: (num_groups_per_row * M, group_size) -> need 2D (M, N)
    # input_shape is (num_groups, group_size), we need the original 2D weight dims
    # Infer M and N: total elements = M * N, and N must be divisible by group_size
    total_elements = x.numel()
    # lora_B shape is (M, K), lora_A shape is (K, N)
    M = self.lora_B.shape[0]
    N = total_elements // M
    weight_shape_2d = (M, N)

    # Fused kernel: LoRA matmul + FQ in one pass
    # STE backward is handled inside FusedLoRAFakeQuantizeSTE (identity for x, matmul grads for B/A)
    output_dq = triton_lora_fake_quantize(
        x, self.lora_B, self.lora_A, input_low, input_range_safe, self.levels, group_size, weight_shape_2d
    )

    return output_dq.reshape(original_shape).to(x.dtype)


def _make_triton_lora_sym_quantize(self, x: Tensor, execute_traced_op_as_identity: bool = False) -> Tensor:
    """Level 2 Triton-fused symmetric LoRA + FQ replacement."""
    if execute_traced_op_as_identity:
        return x
    eps = self.eps
    scale = self.scale
    scale_safe = torch.where(torch.abs(scale) < eps, eps, scale)
    level_low = self.level_low
    level_high = self.level_high
    levels = self.levels

    # Compute input_low and input_range from scale (symmetric formula)
    input_low = torch.where(scale_safe > 0, -scale_safe, -scale_safe / level_low * level_high)
    input_range = torch.abs((2 + 1 / level_low) * scale_safe)

    input_shape = self._lspec.weight_shape
    original_shape = x.shape
    group_size = input_shape[-1]

    # Infer 2D weight shape from LoRA dims
    M = self.lora_B.shape[0]
    N = x.numel() // M
    weight_shape_2d = (M, N)

    # Fused kernel: LoRA matmul + FQ in one pass
    # STE backward is handled inside FusedLoRAFakeQuantizeSTE (identity for x, matmul grads for B/A)
    output_dq = triton_lora_fake_quantize(
        x, self.lora_B, self.lora_A, input_low, input_range, levels, group_size, weight_shape_2d
    )

    return output_dq.reshape(original_shape).to(x.dtype)


def replace_quantizers_with_triton_lora(model: nn.Module, nncf_modules: dict) -> None:
    """
    Replace all NNCF LoRA quantizers with the Level 2 Triton-fused version.

    Level 2 fuses the LoRA matmul (B@A) + addition + FQ into a single kernel,
    eliminating the full (M x N) intermediate correction tensor.

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
            module.quantize = types.MethodType(_make_triton_lora_asym_quantize, module)
            replaced += 1
        elif isinstance(module, SymmetricLoraQuantizer):
            module.quantize = types.MethodType(_make_triton_lora_sym_quantize, module)
            replaced += 1

    print(f"  Replaced {replaced} quantizers with Triton Level 2 (fused LoRA matmul + FQ)")
