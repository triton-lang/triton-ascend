# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Independent rows: automatic axis, packed inputs, both tails and padding."""
import pytest
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl
from block_batching_test_utils import batching_target  # noqa: F401


@triton.jit(do_not_specialize=['N', 'SX', 'SO'])
def _independent_rows(X, Y, O, N, SX, SO, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * B + tl.arange(0, B)
    x = tl.load(X + row * SX + col, col < N, other=0).to(tl.float32)
    y = tl.load(Y + row * SX + col, col < N, other=0).to(tl.float32)
    tl.store(O + row * SO + col, x * tl.sigmoid(x) + y, col < N)


@pytest.mark.parametrize('rows,columns,padding', [(128, 512, 0), (129, 511, 7)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_independent_rows_preserve_packed_views_and_both_tails(rows, columns, padding, dtype):
    generator = torch.Generator().manual_seed(881)
    cpu = torch.randn((rows, 2 * columns + padding), generator=generator).to(dtype)
    packed = cpu.npu()
    x, y = packed[:, :columns], packed[:, columns:2 * columns]
    stride = columns + 11
    storage = [torch.full((rows * stride + 32, ), -123., dtype=dtype, device='npu') for _ in range(2)]
    outputs = [allocation[16:-16].view(rows, stride) for allocation in storage]
    grid = (rows, triton.cdiv(columns, 128))
    kernels = [
        _independent_rows[grid](x, y, out, columns, x.stride(0), stride, 128, block_batching_factor=factor, num_warps=4)
        for out, factor in zip(outputs, (1, 8))
    ]
    torch.npu.synchronize()
    expected = (cpu[:, :columns].float() * torch.sigmoid(cpu[:, :columns].float()) +
                cpu[:, columns:2 * columns].float()).to(dtype)
    tolerance = 1e-2 if dtype == torch.bfloat16 else 2e-4
    for out, allocation in zip(outputs, storage):
        torch.testing.assert_close(out[:, :columns].cpu(), expected, rtol=tolerance, atol=tolerance)
        assert (out[:, columns:].cpu() == -123.).all()
        assert (allocation[:16].cpu() == -123.).all()
        assert (allocation[-16:].cpu() == -123.).all()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(packed.cpu(), cpu, rtol=0, atol=0)
    bundle = kernels[1]
    assert bundle.last_launch['applied_factor'] == 8, bundle.last_launch
    assert bundle.last_launch['selected_axis'] == 0
    assert kernels[0].kernel == bundle.baseline.kernel
    assert all(access['reason'] == 'ALREADY_STRUCTURED' and access['action'] == 'Keep' and not access['rewritten']
               for access in bundle.block_batching['layout']['accesses'])
    if triton.runtime.driver.active.get_current_target().arch.startswith('Ascend950'):
        assert bundle.block_batching['memory_lowering']['structured_row_copy'] is True
    assert 'triton_indirect' not in bundle.candidate.asm['ttadapter']
