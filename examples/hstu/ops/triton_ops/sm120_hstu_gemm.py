# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape-specialized SM120 GEMMs for the BS64 HSTU training workload.

This intentionally keeps SiLU and residual addition outside the GEMM so the
unfused benchmark retains the same operator boundaries.
"""

import os
from functools import lru_cache
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

_DISABLE_ENV = "HSTU_DISABLE_SM120_TRAINING_GEMM"
_SUPPORTED_MNK = {
    (64 * 2700, 4 * 128, 4 * 4 * 128),
    (64 * 2700, 4 * 128, 4 * 128),
}


@lru_cache(maxsize=None)
def _device_is_sm120(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (12, 0)


def _is_disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "0").lower() in ("1", "true", "yes", "on")


def should_use_sm120_hstu_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> bool:
    """Return whether ``linear(input, weight, bias)`` matches the tuned path."""
    if _is_disabled() or not input.is_cuda or not weight.is_cuda:
        return False
    if weight.device != input.device:
        return False
    if input.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        return False
    if input.ndim != 2 or weight.ndim != 2:
        return False
    if not input.is_contiguous() or not weight.is_contiguous():
        return False
    m, k = input.shape
    n, weight_k = weight.shape
    if k != weight_k or (m, k, n) not in _SUPPORTED_MNK:
        return False
    if bias is not None and (
        bias.dtype != torch.bfloat16
        or not bias.is_cuda
        or bias.device != input.device
        or not bias.is_contiguous()
        or bias.shape != (n,)
    ):
        return False
    device_index = input.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return _device_is_sm120(device_index)


@triton.jit
def _sm120_hstu_gemm_fwd(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    input_ptrs = input_ptr + offsets_m[:, None] * K + offsets_k[None, :]
    weight_ptrs = weight_ptr + offsets_n[:, None] * K + offsets_k[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        input_tile = tl.load(
            input_ptrs,
            mask=offsets_m[:, None] < M,
            other=0.0,
        )
        weight_tile = tl.load(weight_ptrs)
        accumulator += tl.dot(input_tile, tl.trans(weight_tile))
        input_ptrs += BLOCK_K
        weight_ptrs += BLOCK_K

    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + offsets_n)[None, :]

    output_ptrs = output_ptr + offsets_m[:, None] * N + offsets_n[None, :]
    tl.store(output_ptrs, accumulator, mask=offsets_m[:, None] < M)


def _sm120_hstu_gemm_forward(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    m, k = input.shape
    n = weight.shape[0]
    output = torch.empty((m, n), dtype=input.dtype, device=input.device)
    block_m = 128
    block_n = 128
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    _sm120_hstu_gemm_fwd[grid](
        input,
        weight,
        bias if bias is not None else weight,
        output,
        m,
        n,
        k,
        HAS_BIAS=bias is not None,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=32,
        GROUP_M=32,
        num_warps=8 if n == 2048 else 4,
        num_stages=3 if n == 2048 else 4,
    )
    return output


class _Sm120HSTUGemmFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        ctx.has_bias = bias is not None
        ctx.save_for_backward(input, weight)
        return _sm120_hstu_gemm_forward(input, weight, bias)

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.mm(grad_output, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.mm(grad_output.t(), input)
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, dim=0)
        return grad_input, grad_weight, grad_bias


def sm120_hstu_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute an HSTU linear layer with a tuned, unfused SM120 GEMM."""
    return _Sm120HSTUGemmFunction.apply(input, weight, bias)
