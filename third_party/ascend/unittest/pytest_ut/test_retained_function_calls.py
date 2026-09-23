import pytest
import torch
import triton
import triton.language as tl


@triton.jit(noinline=True)
def _increment(x):
    return x + 1


@triton.jit(noinline=True)
def _nested(x):
    return _increment(x) * 2


@triton.jit(noinline=True)
def _pair(x):
    return x + 1, x - 1


@triton.jit(noinline=True)
def _store_increment(out, x):
    tl.store(out, x + 1)


@triton.jit
def _call_kernel(x_ptr, out, MODE: tl.constexpr):
    x = tl.load(x_ptr)
    if MODE == 'return':
        tl.store(out, _increment(x))
    elif MODE == 'nested':
        tl.store(out, _nested(x))
    elif MODE == 'pair':
        a, b = _pair(x)
        tl.store(out, a + b)
    else:
        _store_increment(out, x)


@pytest.mark.parametrize('mode,expected', [('return', -4), ('nested', -8), ('pair', -10), ('store', -4)])
def test_retained_scalar_call(mode, expected):
    x = torch.tensor([-5], dtype=torch.int32, device='npu')
    out = torch.empty_like(x)
    compiled = _call_kernel[(1, )](x, out, mode)
    torch.npu.synchronize()
    assert out.item() == expected
    assert 'func.func private @' in compiled.asm['ttadapter']
    assert 'no_inline' in compiled.asm['ttadapter']


@triton.jit(noinline=True)
def _program_info():
    return (100000 * tl.num_programs(0) + 10000 * tl.num_programs(1) + 1000 * tl.num_programs(2) +
            100 * tl.program_id(0) + 10 * tl.program_id(1) + tl.program_id(2))


@triton.jit
def _program_info_kernel(out):
    offset = tl.program_id(0) + 2 * (tl.program_id(1) + 2 * tl.program_id(2))
    tl.store(out + offset, _program_info())


def test_retained_call_program_info():
    out = torch.empty(8, dtype=torch.int32, device='npu')
    _program_info_kernel[(2, 2, 2)](out)
    torch.npu.synchronize()
    expected = torch.tensor([222000 + 100 * x + 10 * y + z for z in range(2) for y in range(2) for x in range(2)],
                            dtype=torch.int32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
