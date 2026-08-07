# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
import torch
import torch.nn.functional as F
from ops.triton_ops import sm120_hstu_gemm as sm120_gemm
from ops.triton_ops.sm120_hstu_gemm import sm120_hstu_linear


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda", 0)
        self.is_cuda = True
        self.ndim = len(shape)

    def is_contiguous(self):
        return True


def test_sm120_hstu_gemm_dispatch_can_be_disabled(monkeypatch):
    input: Any = _FakeTensor((64 * 2700, 512))
    weight: Any = _FakeTensor((2048, 512))
    bias: Any = _FakeTensor((2048,))
    monkeypatch.delenv("HSTU_DISABLE_SM120_TRAINING_GEMM", raising=False)
    monkeypatch.setattr(sm120_gemm, "_device_is_sm120", lambda _: True)

    assert sm120_gemm.should_use_sm120_hstu_gemm(input, weight, bias)
    monkeypatch.setenv("HSTU_DISABLE_SM120_TRAINING_GEMM", "1")
    assert not sm120_gemm.should_use_sm120_hstu_gemm(input, weight, bias)


@pytest.mark.parametrize("n,has_bias", [(2048, True), (512, False)])
def test_sm120_hstu_linear_forward_backward(n, has_bias):
    torch.manual_seed(0)
    m, k = 256, 512
    input = (
        torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    weight = (
        torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    bias = (
        torch.empty((n,), dtype=torch.bfloat16, device="cuda")
        .uniform_(-0.1, 0.1)
        .requires_grad_()
        if has_bias
        else None
    )
    input_ref = input.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    bias_ref = bias.detach().clone().requires_grad_() if bias is not None else None
    grad_output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda").uniform_(
        -0.1, 0.1
    )

    actual = sm120_hstu_linear(input, weight, bias)
    expected = F.linear(input_ref, weight_ref, bias_ref)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    actual.backward(grad_output)
    expected.backward(grad_output)
    torch.testing.assert_close(input.grad, input_ref.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=2e-2, atol=2e-2)
    if bias is not None:
        torch.testing.assert_close(bias.grad, bias_ref.grad, rtol=2e-2, atol=2e-2)
