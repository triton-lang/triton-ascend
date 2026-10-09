import pytest
import torch
import triton
import triton.language as tl


@triton.jit(noinline=True)
def _advance(ptr, shift):
    return ptr + shift


@triton.jit(noinline=True)
def _nested_advance(ptr, shift):
    return _advance(ptr, shift) + 3


@triton.jit(noinline=True)
def _two_pointers(x, y, shift):
    return x + shift, y + 2 * shift


@triton.jit
def _read_returned_pointer(x, out, shift, NESTED: tl.constexpr):
    if NESTED:
        ptr = _nested_advance(x + 8, shift)
    else:
        ptr = _advance(x + 8, shift)
    offsets = tl.arange(0, 16)
    tl.store(out + offsets, tl.load(ptr + offsets))


@pytest.mark.parametrize('shift,nested', [(-3, False), (0, False), (5, False), (2, True)])
def test_returned_pointer_offset(shift, nested):
    x = torch.arange(64, dtype=torch.int32, device='npu') * 3 - 17
    out = torch.empty(16, dtype=torch.int32, device='npu')
    _read_returned_pointer[(1, )](x, out, shift, nested)
    torch.npu.synchronize()
    start = 8 + shift + (3 if nested else 0)
    torch.testing.assert_close(out.cpu(), x[start:start + 16].cpu(), rtol=0, atol=0)


@triton.jit
def _read_two_returned_pointers(x, y, out, shift):
    x_ret, y_ret = _two_pointers(x, y, shift)
    offsets = tl.arange(0, 16)
    value = 2 * tl.load(x_ret + offsets) + tl.load(y_ret + offsets)
    tl.store(out + offsets, value)


def test_multiple_returned_pointers():
    x = torch.arange(64, dtype=torch.int32, device='npu') * 3 - 17
    y = torch.arange(64, dtype=torch.int32, device='npu') * 7 + 51
    out = torch.empty(16, dtype=torch.int32, device='npu')
    _read_two_returned_pointers[(1, )](x, y, out, 5)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), 2 * x[5:21].cpu() + y[10:26].cpu(), rtol=0, atol=0)


@triton.jit(noinline=True)
def _offsets_with_side_effect(shift, counter):
    tl.store(counter, tl.load(counter) + 1)
    return shift, 2 * shift


@triton.jit
def _read_with_returned_offsets(x, y, out, shift, counter):
    sx, sy = _offsets_with_side_effect(shift, counter)
    offsets = tl.arange(0, 16)
    value = 2 * tl.load(x + sx + offsets) + tl.load(y + sy + offsets)
    tl.store(out + offsets, value)


def test_metadata_call_runs_once():
    # Both results are address metadata; their producer must keep its store.
    x = torch.arange(64, dtype=torch.int32, device='npu') * 3 - 17
    y = torch.arange(64, dtype=torch.int32, device='npu') * 7 + 51
    out = torch.empty(16, dtype=torch.int32, device='npu')
    counter = torch.zeros(1, dtype=torch.int32, device='npu')
    _read_with_returned_offsets[(1, )](x, y, out, 5, counter)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), 2 * x[5:21].cpu() + y[10:26].cpu(), rtol=0, atol=0)
    assert counter.item() == 1
