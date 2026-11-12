"""Concatenate two runtime row windows into one NZ tile."""
from pathlib import Path
import sys

_CASE_DIR = Path(__file__).resolve().parent
sys.path[:] = [p for p in sys.path if Path(p or '.').resolve() != _CASE_DIR]
sys.path.insert(0, str(_CASE_DIR.parent))

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import compile_offline, lower_offline, get_stages, check_kernel_binary


@tla.jit
def concatenate(A, B, COUNT, OUT, DTYPE: tl.constexpr):
    result = tla.alloc((112, 64), DTYPE, space='ub')
    with tla.scope.cube():
        tile = tla.alloc((4, 7, 16, 16), DTYPE, space='l1')
        first = tl.load(COUNT)
        tla.load_gm_to_l1(tla.view(A, (128, 64)), tile, tla.LoadGM2L1Params(row_count=first, source_row=128 - first))
        if first < 112:
            tla.load_gm_to_l1(tla.view(B, (128, 64)), tile,
                              tla.LoadGM2L1Params(row_count=112 - first, destination_row=first))
        tla.copy_l1_to_ub(tile, result)
        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE1, 0)
    with tla.scope.vector():
        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 0)
        if tla.sub_block_id() == 0:
            tla.store_ub_to_gm(result, tla.view(OUT, (112, 64)))


@pytest.mark.parametrize('dtype', ('fp16', 'bf16'))
def test_ir(dtype):
    element = tl.float16 if dtype == 'fp16' else tl.bfloat16
    signature = dict(A='*' + dtype, B='*' + dtype, COUNT='*i32', OUT='*' + dtype)
    module, context = compile_offline(concatenate, signature, dict(DTYPE=element))
    text = str(lower_offline(module, context))
    assert text.count('hivm.hir.nd2nz ') == 2
    assert 'memref.subview' in text and 'arith.divui' in text
    assert 'hivm.hir.l12ub' in text


@pytest.mark.parametrize('dtype', ('fp16', 'bf16'))
def test_run(dtype):
    if not get_stages()[2]:
        pytest.skip('Run stage requires an Ascend NPU')
    import torch
    import torch_npu  # noqa: F401

    element = tl.float16 if dtype == 'fp16' else tl.bfloat16
    torch_dtype = torch.float16 if dtype == 'fp16' else torch.bfloat16
    generator = torch.Generator().manual_seed(37)
    a = torch.randn((128, 64), generator=generator).to(torch_dtype)
    b = torch.randn((128, 64), generator=generator).to(torch_dtype)
    a_device, b_device = a.npu(), b.npu()
    count = torch.empty((), dtype=torch.int32, device='npu')
    out = torch.empty((112, 64), dtype=torch_dtype, device='npu')
    for first in (16, 64, 112):
        count.fill_(first)
        out.fill_(float('nan'))
        binary = concatenate[(1, )](a_device, b_device, count, out, element)
        torch.npu.synchronize()
        check_kernel_binary(binary)
        expected = torch.cat((a[128 - first:], b[:112 - first]))
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)
