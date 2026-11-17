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

from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def _forced_mixed_indirect_load_kernel(source, indices, output, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    order = tl.load(indices + tl.arange(0, BLOCK_SIZE))
    values = tl.load(source + tl.program_id(0) * BLOCK_SIZE + order)
    tl.store(output + offsets, values + 1.0)


def _check_forced_mixed_factors(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_ALL_BLOCKS_PARALLEL", "1")
    properties = driver.active.utils.get_device_properties(torch.npu.current_device())
    vector_cores = int(properties["num_vectorcore"])
    # Local F2 batching needs multiple iterations of the V1 program loop.
    logical_programs, block = 4 * vector_cores, 128
    source = torch.arange(logical_programs * block, dtype=torch.float32, device="npu").reshape(logical_programs, block)
    indices = torch.arange(block - 1, -1, -1, dtype=torch.int32, device="npu")
    output = torch.empty_like(source)
    expected = source.cpu().index_select(1, indices.cpu().long()) + 1.0
    args = (source, indices, output)
    options = {
        "num_warps": 4,
        "compile_mode": "simd_simt",
        "auto_simt_scope_mode": "auto",
        "auto_simt_scope_dump": str(tmp_path / "forced_mixed_route.json"),
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": vector_cores,
        "physical_aicore_count_hint": int(properties["num_aicore"]),
        "BLOCK_SIZE": block,
    }
    kernels = {}
    binaries = {}
    for factor in (None, 1, 2, 1, None):
        if factor is None:
            monkeypatch.delenv("TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE", raising=False)
        else:
            monkeypatch.setenv("TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE", "mixed_simd_simt")
        launch_options = dict(options, superblock_factor=factor if factor is not None else 1)
        compiled = _forced_mixed_indirect_load_kernel.warmup(*args, grid=(logical_programs, ), **launch_options)
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
        _forced_mixed_indirect_load_kernel[(logical_programs, )](*args, **launch_options)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
        print(f"route={compiled.metadata.auto_simt_effective_kind} factor={factor} precision=passed", flush=True)
    assert binaries[1] != binaries[2]
    explicit_options = dict(options, auto_simt_force_route="mixed_simd_simt", superblock_factor=2)
    explicit = _forced_mixed_indirect_load_kernel.warmup(*args, grid=(logical_programs, ), **explicit_options)
    assert explicit is kernels[2]
    _forced_mixed_indirect_load_kernel[(logical_programs, )](*args, **explicit_options)
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)


def test_costmodel_forced_mixed_factors(tmp_path):
    arch = str(driver.active.get_current_target().arch)
    if not arch.startswith(("Ascend910_95", "Ascend950")):
        pytest.skip("forced Mixed runtime regression requires Ascend 950")
    # Keep device-context errors within this regression's child process.
    command = [sys.executable, str(Path(__file__).resolve()), str(tmp_path), str(torch.npu.current_device())]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    log = result.stdout + result.stderr
    (tmp_path / "forced_mixed_runtime.log").write_text(log, encoding="utf-8")
    assert result.returncode == 0, log[-20000:]
    print(result.stdout, end="")


if __name__ == "__main__":
    torch.npu.set_device(int(sys.argv[2]))
    with pytest.MonkeyPatch.context() as monkeypatch:
        _check_forced_mixed_factors(Path(sys.argv[1]), monkeypatch)
