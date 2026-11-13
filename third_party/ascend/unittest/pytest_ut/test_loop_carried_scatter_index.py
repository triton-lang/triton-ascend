# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl


@triton.jit
def scan_scatter(X, Y, Counts, steps, B: tl.constexpr, R: tl.constexpr, SCATTER: tl.constexpr):
    # Regression adapted from PR #2607. The counts-to-address dependency crosses
    # a scalar reduction; it must not force counts into legacy BlockData.
    core = tl.program_id(0)
    if core < R:
        counts = tl.zeros((R, ), tl.int32)
        offsets = tl.arange(0, B)
        for block in range(steps):
            labels = tl.load(X + block * B + offsets)
            positions = tl.zeros((B, ), tl.int32)
            for route in tl.static_range(R):
                hit = (labels == route).to(tl.int32)
                prefix = tl.cumsum(hit, 0) - hit
                hot = (tl.arange(0, R) == route).to(tl.int32)
                previous = tl.sum(counts * hot, 0)
                positions = tl.where(labels == route, previous + prefix, positions)
                counts += hot * tl.sum(hit, 0)
            if SCATTER:
                index = core * steps * B * R + labels * (steps * B) + positions
            else:
                index = core * steps * B + block * B + offsets
            tl.store(Y + index, block * B + offsets)
        # Keep the carried state observable even in the non-scatter control.
        tl.store(Counts + core * R + tl.arange(0, R), counts)


def reference(x, steps, B, R, scatter):
    y = torch.full((R * R * B * steps, ), -1, dtype=torch.int32)
    totals = torch.empty((R, R), dtype=torch.int32)
    for core in range(R):
        counts = [0] * R
        for block in range(steps):
            labels = x[block * B:(block + 1) * B].tolist()
            positions = [0] * B
            for route in range(R):
                hit = [1 if l == route else 0 for l in labels]
                run = 0
                prefix = []
                for h in hit:
                    prefix.append(run)
                    run += h
                previous = counts[route]
                for i in range(B):
                    if labels[i] == route:
                        positions[i] = previous + prefix[i]
                counts[route] += sum(hit)
            for i in range(B):
                if scatter:
                    index = core * steps * B * R + labels[i] * (steps * B) + positions[i]
                else:
                    index = core * steps * B + block * B + i
                y[index] = block * B + i
        totals[core] = torch.tensor(counts, dtype=torch.int32)
    return y, totals


@pytest.mark.parametrize("pattern", ["balanced", "skewed"])
@pytest.mark.parametrize("scatter", [True, False])
@pytest.mark.parametrize("B,R,steps", [
    pytest.param(8, 4, 2, id="repro"),
    pytest.param(16, 4, 3, id="wider"),
    pytest.param(4, 2, 2, id="minimal"),
])
def test_loop_carried_counts_scatter(pattern, scatter, B, R, steps):
    lanes = torch.arange(B * steps, dtype=torch.int32)
    if pattern == "balanced":
        x = lanes % R
    else:
        # Uneven counts across blocks, with empty routes when R > 2.
        x = torch.where(lanes % 3 == 0, R - 1, 0).to(torch.int32)
    x = x.npu()
    y = torch.full((R * R * B * steps, ), -1, dtype=torch.int32).npu()
    counts = torch.full((R, R), -1, dtype=torch.int32).npu()
    scan_scatter[(R, )](x, y, counts, steps, B=B, R=R, SCATTER=scatter)
    torch.npu.synchronize()
    expected, expected_counts = reference(x.cpu(), steps, B, R, scatter)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(counts.cpu(), expected_counts, rtol=0, atol=0)
