import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _pointer_select_loop(X, Y, Out, N, USE_WHILE: tl.constexpr):
    pointer = X
    total = tl.full((), 0, tl.int32)
    if USE_WHILE:
        i = 0
        while i < N:
            pointer = tl.where(i % 2 == 0, pointer + 1, Y + i)
            total += tl.load(pointer)
            i += 1
    else:
        for i in range(N):
            pointer = tl.where(i % 2 == 0, pointer + 1, Y + i)
            total += tl.load(pointer)
    tl.store(Out, total)
    tl.store(Out + 1, tl.load(pointer))


@pytest.mark.parametrize("use_while", [False, True])
@pytest.mark.parametrize("count", [0, 1, 5])
def test_scalar_pointer_select_loop(use_while, count):
    x = torch.arange(10, 18, device="npu", dtype=torch.int32)
    y = torch.arange(100, 108, device="npu", dtype=torch.int32)
    output = torch.empty(2, device="npu", dtype=torch.int32)
    _pointer_select_loop[(1, )](x, y, output, count, use_while)
    base, offset, total = 10, 0, 0
    for i in range(count):
        if i % 2 == 0:
            offset += 1
        else:
            base, offset = 100, i
        total += base + offset
    expected = torch.tensor([total, base + offset], dtype=torch.int32)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
