# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401

from test_dot_scaled_boundaries import _dot_boundary, _fp4_host

from dot_scaled_test_utils import require_dot_scaled_formats


@pytest.mark.parametrize("k", [32, 64, 128, 256])
@pytest.mark.parametrize("with_a_scale,with_b_scale", [(False, False), (False, True), (True, False), (True, True)])
def test_dot_scaled_native_fp4_boundary(k, with_a_scale, with_b_scale):
    # K=64 has storage K=32, the size at which the native MX decomposition
    # previously interpreted packed bytes as ordinary integer operands.
    require_dot_scaled_formats("e2m1", "e2m1")
    m, n = 16, 32
    a, ar = _fp4_host(m, k, 1)
    b, br = _fp4_host(k, n, 0, phase=3)
    ai = (torch.arange(m)[:, None] + torch.arange(k // 32)[None, :]) % 4
    bi = (torch.arange(n)[:, None] * 3 + torch.arange(k // 32)[None, :]) % 4
    sa = torch.tensor([125, 127, 128, 129], dtype=torch.uint8)[ai]
    sb = torch.tensor([126, 125, 128, 127], dtype=torch.uint8)[bi]
    sar = torch.exp2(sa.float() - 127).repeat_interleave(32, 1) if with_a_scale else 1.
    sbr = torch.exp2(sb.float() - 127).repeat_interleave(32, 1).T if with_b_scale else 1.
    expected = (ar * sar) @ (br * sbr) + .125
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _dot_boundary[(1, )](a.to("npu"), b.to("npu"), sa.to("npu") if with_a_scale else None,
                         sb.to("npu") if with_b_scale else None, actual, m, n, k, "e2m1", "e2m1", True, True, 32, .125)
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-5, rtol=1e-5)
