import pytest
import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def _conditional_indirect_store(indices, values, output, enabled, active_count, N: tl.constexpr,
                                MASK_INDEX: tl.constexpr):
    for i in range(N):
        if MASK_INDEX:
            index = tl.load(indices + i + tl.arange(0, 1), mask=enabled)
        else:
            index = tl.load(indices + i + tl.arange(0, 1))
        ptr = output + index
        if enabled:
            value = tl.load(values + i + tl.arange(0, 1))
            tl.store(ptr, value + 1, mask=i < active_count)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('active_count', [0, 9, 17])
@pytest.mark.parametrize('mask_index', [False, True])
def test_conditional_indirect_store_operands(enabled, active_count, mask_index):
    n = 17
    indices_cpu = (torch.arange(n, dtype=torch.int32) * 5) % n
    values_cpu = torch.arange(n, dtype=torch.int32) * 3
    output = torch.full((n, ), -1, dtype=torch.int32, device='npu')

    _conditional_indirect_store[(1, )](
        indices_cpu.npu(),
        values_cpu.npu(),
        output,
        enabled,
        active_count,
        n,
        mask_index,
    )

    expected = torch.full((n, ), -1, dtype=torch.int32)
    if enabled:
        expected[indices_cpu[:active_count].long()] = values_cpu[:active_count] + 1
    assert torch.equal(output.cpu(), expected)


@triton.jit
def _indirect_load_after_store(indices, data, output, value, N: tl.constexpr):
    index = tl.load(indices + tl.arange(0, N))
    ptr = data + index
    tl.store(data, value)
    loaded = tl.load(ptr)
    tl.store(output + tl.arange(0, N), loaded)


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('num_indices', [1, 2])
def test_indirect_load_observes_preceding_store(index, num_indices):
    indices_cpu = torch.tensor([index, 1 - index][:num_indices], dtype=torch.int32)
    indices = indices_cpu.npu()
    data = torch.tensor([3, 7], dtype=torch.int32, device='npu')
    output = torch.empty((num_indices, ), dtype=torch.int32, device='npu')

    _indirect_load_after_store[(1, )](indices, data, output, 42, num_indices)

    expected = torch.tensor([42, 7], dtype=torch.int32)[indices_cpu.long()]
    assert torch.equal(output.cpu(), expected)
    assert torch.equal(data.cpu(), torch.tensor([42, 7], dtype=torch.int32))
