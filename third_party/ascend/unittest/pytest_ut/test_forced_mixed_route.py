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

import torch

from test_simd_simt_costmodel_cases import (
    _launch_options,
    _vector_core_count,
    gather_dot_min,
    simd_simt_910_95_only,
)


@simd_simt_910_95_only
def test_costmodel_forced_mixed_factors(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", "1")
    logical_programs, block = _vector_core_count(), 16
    a = torch.randn((logical_programs * block, 256), dtype=torch.float16, device="npu")
    b = torch.randn((256, block), dtype=torch.float16, device="npu")
    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255, 1, 2, 3, 4, 6, 7, 8, 9], dtype=torch.int32, device="npu")
    output = torch.empty((logical_programs * block, block), dtype=torch.float32, device="npu")
    index_cpu = indices.cpu().long()
    expected = a.cpu()[:, index_cpu].float() @ b.cpu()[index_cpu, :].float()
    args = (a, b, indices, output, a.shape[0], b.shape[1], *a.stride(), *b.stride(), *output.stride())
    # Reuse the report path while toggling route forcing through the environment
    # and supplying F through the ordinary superblock_factor launch option.
    options = dict(_launch_options(tmp_path / "forced_mixed_route.json", logical_programs), BLOCK_M=block,
                   BLOCK_N=block, BLOCK_K=block)
    kernels = {}
    binaries = {}
    for factor in (None, 1, 2, 1, None):
        if factor is None:
            monkeypatch.delenv("TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE", raising=False)
        else:
            monkeypatch.setenv("TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE", "mixed_simd_simt")
        launch_options = dict(options, superblock_factor=factor if factor is not None else 1)
        compiled = gather_dot_min.warmup(*args, grid=(logical_programs, 1), **launch_options)
        if factor is not None:
            assert compiled.metadata.auto_simt_effective_kind == "mixed_simd_simt"
            assert compiled.metadata.auto_simt_forced_factor == factor
            assert compiled.metadata.auto_simt_scope_superblock_factor == factor
        else:
            assert not compiled.metadata.auto_simt_force_route
        if factor in kernels:
            assert compiled is kernels[factor]
        kernels[factor] = compiled
        binaries[factor] = compiled.asm["npubin"]
        gather_dot_min[(logical_programs, 1)](*args, **launch_options)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)
    assert binaries[1] != binaries[2]
    explicit_options = dict(options, auto_simt_force_route="mixed_simd_simt", superblock_factor=2)
    explicit = gather_dot_min.warmup(*args, grid=(logical_programs, 1), **explicit_options)
    assert explicit is kernels[2]
    gather_dot_min[(logical_programs, 1)](*args, **explicit_options)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)
