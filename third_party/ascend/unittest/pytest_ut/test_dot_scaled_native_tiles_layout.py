# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT

import pytest
import torch
import torch_npu  # noqa: F401

from test_dot_scaled_native_tiles import _batch_native_tile_dot, _native_tile_dot

from dot_scaled_test_utils import require_dot_scaled_formats


def _operand(rows, cols, format, pack_axis, phase):
    # A 13-position pattern exposes exchanges of 16-row/column subblocks.
    # Keep values binary-exact so the unchanged zero-tolerance comparison
    # diagnoses layout rather than reference rounding.
    indices = (torch.arange(rows)[:, None] * 3 + torch.arange(cols)[None, :] * 5 + phase) % 13
    if format == "e2m1":
        codes = torch.tensor([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 15])[indices]
        values = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])[codes]
        if pack_axis == 0:
            packed = codes[0::2, :] | (codes[1::2, :] << 4)
        else:
            packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        return packed.to(torch.uint8).contiguous(), values
    dtype = torch.float8_e4m3fn if format == "e4m3" else torch.float8_e5m2
    values = ((indices - 6).float() / 8).to(dtype)
    return values, values.float()


def _assert_layout_sensitive(product):
    for axis in [-2, -1]:
        size = product.shape[axis]
        if size < 32:
            continue
        first, second = (16, 32) if size >= 48 else (0, 16)
        assert not torch.equal(product.narrow(axis, first, 16), product.narrow(axis, second, 16))


@pytest.mark.parametrize("shape", [(128, 32, 128), (32, 128, 128), (128, 128, 128), (128, 128, 256)])
@pytest.mark.parametrize("a_format,b_format", [("e2m1", "e2m1"), ("e4m3", "e4m3"), ("e5m2", "e4m3")])
@pytest.mark.parametrize("with_lhs,with_rhs", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("explicit_acc", [False, True])
def test_dot_scaled_native_large_mn_nonperiodic(shape, a_format, b_format, with_lhs, with_rhs, explicit_acc):
    require_dot_scaled_formats(a_format, b_format)
    m, n, k = shape
    a, ar = _operand(m, 2 * k, a_format, 1, 0)
    b, br = _operand(2 * k, n, b_format, 0, 3)
    # Column-major B exercises indirect addresses or a full load followed by
    # a transpose. Both must fit UB with the two original K blocks intact.
    b = b.T.contiguous().T
    ai = (torch.arange(m)[:, None] + torch.arange(2 * k // 32)[None, :]) % 4
    bi = (torch.arange(n)[:, None] * 3 + torch.arange(2 * k // 32)[None, :]) % 4
    sa = torch.tensor([125, 127, 128, 129], dtype=torch.uint8)[ai]
    sb = torch.tensor([126, 125, 128, 127], dtype=torch.uint8)[bi]
    if with_lhs:
        ar = ar * torch.exp2(sa.float() - 127).repeat_interleave(32, 1)
    if with_rhs:
        br = br * torch.exp2(sb.float() - 127).repeat_interleave(32, 1).T
    expected = .125 + torch.arange(m)[:, None].float() / 128 + torch.arange(n)[None, :].float() / 64
    product = torch.zeros_like(expected)
    for block in range(2):
        partial = ar[:, block * k:(block + 1) * k] @ br[block * k:(block + 1) * k]
        expected += partial
        product += partial
    _assert_layout_sensitive(product)
    actual = torch.empty((m, n), dtype=torch.float32, device="npu")
    _native_tile_dot[(1, )](a.to("npu"), b.to("npu"), sa.to("npu") if with_lhs else None,
                            sb.to("npu") if with_rhs else None, actual, *b.stride(), m, n, k, a_format, b_format,
                            explicit_acc)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(64, 16, 128), (16, 64, 128)])
@pytest.mark.parametrize("format", ["e2m1", "e4m3"])
@pytest.mark.parametrize("with_lhs,with_rhs", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("nonzero_acc", [False, True])
def test_dot_scaled_native_batched_mn_nonperiodic(shape, format, with_lhs, with_rhs, nonzero_acc):
    require_dot_scaled_formats(format, format)
    m, n, k = shape
    a, ar = zip(*[_operand(m, k, format, 1, batch * 3) for batch in range(2)])
    b, br = zip(*[_operand(k, n, format, 0, batch * 5 + 3) for batch in range(2)])
    a, ar, b, br = (torch.stack(values) for values in (a, ar, b, br))
    ai = (torch.arange(2)[:, None, None] * 2 + torch.arange(m)[None, :, None] +
          torch.arange(k // 32)[None, None, :]) % 4
    bi = (torch.arange(2)[:, None, None] * 3 + torch.arange(n)[None, :, None] * 3 +
          torch.arange(k // 32)[None, None, :]) % 4
    sa = torch.tensor([125, 127, 128, 129], dtype=torch.uint8)[ai]
    sb = torch.tensor([126, 125, 128, 127], dtype=torch.uint8)[bi]
    if with_lhs:
        ar = ar * torch.exp2(sa.float() - 127).repeat_interleave(32, 2)
    if with_rhs:
        br = br * torch.exp2(sb.float() - 127).repeat_interleave(32, 2).transpose(1, 2)
    expected = ar @ br
    _assert_layout_sensitive(expected)
    if nonzero_acc:
        expected += (.125 + torch.arange(2)[:, None, None].float() / 4 + torch.arange(m)[None, :, None].float() / 128 +
                     torch.arange(n)[None, None, :].float() / 64)
    actual = torch.empty((2, m, n), dtype=torch.float32, device="npu")
    _batch_native_tile_dot[(1, )](a.to("npu"), b.to("npu"), sa.to("npu") if with_lhs else None,
                                  sb.to("npu") if with_rhs else None, actual, m, n, k, format, nonzero_acc)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
