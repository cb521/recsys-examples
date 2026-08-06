# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import torch
from ops.triton_ops.triton_addmm import should_use_triton_addmm_silu, triton_addmm
from ops.unfused import should_force_unfused_hstu


@pytest.mark.parametrize(
    "sm, expected",
    [
        (8, True),
        (9, False),
        (10, False),
        (12, True),
    ],
)
def test_should_use_triton_addmm_silu(monkeypatch, sm, expected):
    monkeypatch.delenv("HSTU_FORCE_UNFUSED", raising=False)
    monkeypatch.delenv("HSTU_FORCE_UNFUSED_ADDMM_SILU", raising=False)
    assert should_use_triton_addmm_silu(sm) is expected


@pytest.mark.parametrize("sm", [8, 9, 10, 12])
def test_force_unfused_addmm_silu(monkeypatch, sm):
    monkeypatch.delenv("HSTU_FORCE_UNFUSED", raising=False)
    monkeypatch.setenv("HSTU_FORCE_UNFUSED_ADDMM_SILU", "1")
    assert should_use_triton_addmm_silu(sm) is False


@pytest.mark.parametrize(
    "env_name", ["HSTU_FORCE_UNFUSED", "HSTU_FORCE_UNFUSED_ADDMM_SILU"]
)
def test_force_unfused_hstu_env_aliases(monkeypatch, env_name):
    monkeypatch.delenv("HSTU_FORCE_UNFUSED", raising=False)
    monkeypatch.delenv("HSTU_FORCE_UNFUSED_ADDMM_SILU", raising=False)
    monkeypatch.setenv(env_name, "1")
    assert should_force_unfused_hstu() is True
    assert should_use_triton_addmm_silu(12) is False


@pytest.mark.parametrize("m,n,k", [(62, 512, 128), [128, 128, 128]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("y_1d", [True])
@pytest.mark.parametrize("with_silu", [True, False])
def test_addmm(m, n, k, dtype, y_1d, with_silu):
    # this is must be set to False when dtype is float32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    x = (
        torch.empty((m, k), dtype=dtype, device="cuda")
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    weight = (
        torch.empty((k, n), dtype=dtype, device="cuda")
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    # y =  torch.empty((m), dtype=dtype, device="cuda").uniform_(-0.1, 0.1).requires_grad_()
    y = (
        torch.zeros((n,) if y_1d else (m, n), dtype=dtype, device="cuda")
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    x_ref = x.detach().clone().requires_grad_()
    weight_ref = weight.detach().clone().requires_grad_()
    y_ref = y.detach().clone().requires_grad_()

    dz = torch.empty((m, n), dtype=dtype, device="cuda").uniform_(-0.1, 0.1)

    triton_res = triton_addmm(y, x, weight, silu=with_silu)
    torch_res = torch.addmm(y_ref, x_ref, weight_ref)
    if with_silu:
        torch_res = torch.nn.functional.silu(torch_res)
    torch.testing.assert_close(triton_res, torch_res)

    triton_res.backward(dz)
    torch_res.backward(dz)

    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(weight.grad, weight_ref.grad)
    torch.testing.assert_close(y.grad, y_ref.grad)
