import pytest
import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def _conditional_indirect_store(indices, values, output, enabled, active_count, N: tl.constexpr):
    for i in range(N):
        index = tl.load(indices + i + tl.arange(0, 1))
        ptr = output + index
        if enabled:
            value = tl.load(values + i + tl.arange(0, 1))
            tl.store(ptr, value + 1, mask=i < active_count)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('active_count', [0, 9, 17])
def test_conditional_indirect_store_operands(enabled, active_count):
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
    )

    expected = torch.full((n, ), -1, dtype=torch.int32)
    if enabled:
        expected[indices_cpu[:active_count].long()] = values_cpu[:active_count] + 1
    assert torch.equal(output.cpu(), expected)


@triton.jit
def _indirect_load_after_store(indices, data, output, value):
    index = tl.load(indices + tl.arange(0, 1))
    ptr = data + index
    tl.store(data, value)
    loaded = tl.load(ptr)
    tl.store(output + tl.arange(0, 1), loaded)


@pytest.mark.parametrize('index', [0, 1])
def test_indirect_load_observes_preceding_store(index):
    indices = torch.tensor([index], dtype=torch.int32, device='npu')
    data = torch.tensor([3, 7], dtype=torch.int32, device='npu')
    output = torch.empty((1, ), dtype=torch.int32, device='npu')

    _indirect_load_after_store[(1, )](indices, data, output, 42)

    expected = torch.tensor([42 if index == 0 else 7], dtype=torch.int32)
    assert torch.equal(output.cpu(), expected)
    assert torch.equal(data.cpu(), torch.tensor([42, 7], dtype=torch.int32))
