# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pytest

import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import test_common

import torch
import torch_npu


def torch_fmod(x0, x1):
    """Reference fmod. Computed on CPU via std::fmod, which is exact."""
    return torch.fmod(x0.cpu(), x1.cpu()).to(x0.device)


@triton.jit
def triton_fmod(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        x_index = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        tmp0 = tl.load(in_ptr0 + x_index)
        tmp1 = tl.load(in_ptr1 + x_index)
        tmp2 = libdevice.fmod(tmp0, tmp1)
        tl.store(out_ptr0 + x_index, tmp2, None)


@pytest.mark.parametrize('param_list',
                         [
                             ['float32', (2, 4096, 8), 2, 32768, 1024],
                         ])
def test_fmod(param_list):
    """Test fmod precision for float32 and float16 with various input ranges."""
    # Generate data
    dtype, shape, ncore, xblock, xblock_sub = param_list
    x0 = test_common.generate_tensor(shape, dtype).npu()
    y_tmp = test_common.generate_tensor(shape, dtype)
    # Avoid division by zero
    y0 = y_tmp.masked_fill(y_tmp == 0, 1)
    y0 = y0.npu()

    # torch result
    y_ref = torch_fmod(x0, y0).to(eval('torch.' + dtype))
    # triton result
    y_cal = torch.zeros(shape, dtype=eval('torch.' + dtype)).npu()
    triton_fmod[ncore, 1, 1](x0, y0, y_cal, xblock, xblock_sub)
    # Compare results
    test_common.validate_cmp(dtype, y_cal, y_ref)


def test_fmod_known_values():
    """Test fmod with known expected values for regression testing."""
    dtype = 'float32'
    shape = (8,)
    ncore = 1
    xblock = 8
    xblock_sub = 8

    # Known test cases with expected results
    # fmod(7.5, 2.0) should be 1.5
    # fmod(-7.5, 2.0) should be -1.5
    # fmod(7.5, -2.0) should be 1.5
    # fmod(-7.5, -2.0) should be -1.5
    x0_np = test_common.generate_numpy(shape, dtype)
    x0_np[0] = 7.5
    x0_np[1] = -7.5
    x0_np[2] = 7.5
    x0_np[3] = -7.5
    x0_np[4] = 10.0
    x0_np[5] = -10.0
    x0_np[6] = 10.0
    x0_np[7] = -10.0

    y0_np = test_common.generate_numpy(shape, dtype)
    y0_np[0] = 2.0
    y0_np[1] = 2.0
    y0_np[2] = -2.0
    y0_np[3] = -2.0
    y0_np[4] = 3.0
    y0_np[5] = 3.0
    y0_np[6] = -3.0
    y0_np[7] = -3.0

    x0 = torch.from_numpy(x0_np).to(torch.float32).npu()
    y0 = torch.from_numpy(y0_np).to(torch.float32).npu()

    # torch result
    y_ref = torch_fmod(x0, y0)
    # triton result
    y_cal = torch.zeros(shape, dtype=torch.float32).npu()
    triton_fmod[ncore, 1, 1](x0, y0, y_cal, xblock, xblock_sub)
    # Compare results
    test_common.validate_cmp(dtype, y_cal, y_ref)


def test_fmod_edge_cases():
    """Test fmod edge cases: negative values, very small/large values."""
    dtype = 'float32'
    shape = (5,)
    ncore = 1
    xblock = 5
    xblock_sub = 5

    # Test cases: negative and positive values
    x0_np = test_common.generate_numpy(shape, dtype)
    x0_np[0] = -10.5
    x0_np[1] = -5.5
    x0_np[2] = 0.0
    x0_np[3] = 5.5
    x0_np[4] = 10.5

    y0_np = test_common.generate_numpy(shape, dtype)
    y0_np[0] = 3.0
    y0_np[1] = 3.0
    y0_np[2] = 3.0
    y0_np[3] = 3.0
    y0_np[4] = 3.0

    x0 = torch.from_numpy(x0_np).to(torch.float32).npu()
    y0 = torch.from_numpy(y0_np).to(torch.float32).npu()

    # torch result
    y_ref = torch_fmod(x0, y0)
    # triton result
    y_cal = torch.zeros(shape, dtype=torch.float32).npu()
    triton_fmod[ncore, 1, 1](x0, y0, y_cal, xblock, xblock_sub)
    # Compare results
    test_common.validate_cmp(dtype, y_cal, y_ref)


def test_fmod_large_values():
    """Test fmod with large values."""
    dtype = 'float32'
    shape = (5,)
    ncore = 1
    xblock = 5
    xblock_sub = 5

    # Test cases: large values
    x0_np = test_common.generate_numpy(shape, dtype)
    x0_np[0] = 1e3
    x0_np[1] = 1e4
    x0_np[2] = 1e5
    x0_np[3] = 1e6
    x0_np[4] = 1e7

    y0_np = test_common.generate_numpy(shape, dtype)
    y0_np[0] = 1e2
    y0_np[1] = 1e2
    y0_np[2] = 1e2
    y0_np[3] = 1e2
    y0_np[4] = 1e2

    x0 = torch.from_numpy(x0_np).to(torch.float32).npu()
    y0 = torch.from_numpy(y0_np).to(torch.float32).npu()

    # torch result
    y_ref = torch_fmod(x0, y0)
    # triton result
    y_cal = torch.zeros(shape, dtype=torch.float32).npu()
    triton_fmod[ncore, 1, 1](x0, y0, y_cal, xblock, xblock_sub)
    # Compare results
    test_common.validate_cmp(dtype, y_cal, y_ref)


def test_fmod_small_values():
    """Test fmod with very small values."""
    dtype = 'float32'
    shape = (5,)
    ncore = 1
    xblock = 5
    xblock_sub = 5

    # Test cases: very small values
    x0_np = test_common.generate_numpy(shape, dtype)
    x0_np[0] = 1e-6
    x0_np[1] = 1e-5
    x0_np[2] = 1e-4
    x0_np[3] = 1e-3
    x0_np[4] = 1e-2

    y0_np = test_common.generate_numpy(shape, dtype)
    y0_np[0] = 1e-3
    y0_np[1] = 1e-3
    y0_np[2] = 1e-3
    y0_np[3] = 1e-3
    y0_np[4] = 1e-3

    x0 = torch.from_numpy(x0_np).to(torch.float32).npu()
    y0 = torch.from_numpy(y0_np).to(torch.float32).npu()

    # torch result
    y_ref = torch_fmod(x0, y0)
    # triton result
    y_cal = torch.zeros(shape, dtype=torch.float32).npu()
    triton_fmod[ncore, 1, 1](x0, y0, y_cal, xblock, xblock_sub)
    # Compare results
    test_common.validate_cmp(dtype, y_cal, y_ref)


def test_fmod_pi_multiples():
    """Test fmod with pi multiples.
    
    Note: fmod(k*pi, pi) should give 0 for integer k, but due to floating-point
    representation, some cases may have small non-zero remainders.
    """
    import math
    dtype = 'float32'
    shape = (5,)
    ncore = 1
    xblock = 5
    xblock_sub = 5

    # Test cases: pi multiples
    x0_np = test_common.generate_numpy(shape, dtype)
    x0_np[0] = math.pi
    x0_np[1] = 2 * math.pi
    x0_np[2] = 3 * math.pi
    x0_np[3] = 4 * math.pi
    x0_np[4] = 5 * math.pi

    y0_np = test_common.generate_numpy(shape, dtype)
    y0_np[0] = math.pi
    y0_np[1] = math.pi
    y0_np[2] = math.pi
    y0_np[3] = math.pi
    y0_np[4] = math.pi

    x0 = torch.from_numpy(x0_np).to(torch.float32).npu()
    y0 = torch.from_numpy(y0_np).to(torch.float32).npu()

    # torch result
    y_ref = torch_fmod(x0, y0)
    # triton result
    y_cal = torch.zeros(shape, dtype=torch.float32).npu()
    triton_fmod[ncore, 1, 1](x0, y0, y_cal, xblock, xblock_sub)
    test_common.validate_cmp(dtype, y_cal, y_ref)


def test_fmod_sign_behavior():
    """Test that fmod sign follows the dividend (x) sign."""
    dtype = 'float32'
    shape = (4,)
    ncore = 1
    xblock = 4
    xblock_sub = 4

    # Test that result has same sign as dividend
    x0_np = test_common.generate_numpy(shape, dtype)
    x0_np[0] = -5.0
    x0_np[1] = -5.0
    x0_np[2] = 5.0
    x0_np[3] = 5.0

    y0_np = test_common.generate_numpy(shape, dtype)
    y0_np[0] = 2.0
    y0_np[1] = -2.0
    y0_np[2] = 2.0
    y0_np[3] = -2.0

    x0 = torch.from_numpy(x0_np).to(torch.float32).npu()
    y0 = torch.from_numpy(y0_np).to(torch.float32).npu()

    # triton result
    y_cal = torch.zeros(shape, dtype=torch.float32).npu()
    triton_fmod[ncore, 1, 1](x0, y0, y_cal, xblock, xblock_sub)

    # Result should have same sign as dividend (x)
    result = y_cal.cpu().numpy()
    x0_cpu = x0.cpu().numpy()
    assert result[0] < 0, f"Expected negative result for x={x0_cpu[0]}, got {result[0]}"
    assert result[1] < 0, f"Expected negative result for x={x0_cpu[1]}, got {result[1]}"
    assert result[2] > 0, f"Expected positive result for x={x0_cpu[2]}, got {result[2]}"
    assert result[3] > 0, f"Expected positive result for x={x0_cpu[3]}, got {result[3]}"