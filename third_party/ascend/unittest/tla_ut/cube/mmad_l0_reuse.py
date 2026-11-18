"""Check FlashMLA-shaped L0A reuse against independent FP32 matmul."""

import pytest
import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import Options, check_kernel_binary, compile_offline, get_stages, lower_offline


class A5Options(Options):
    arch = 'Ascend950PR_9589'


@tla.jit
def resident_q(Q, KEYS, OUT, DTYPE: tl.constexpr):
    M: tl.constexpr = 96
    N: tl.constexpr = 128
    K: tl.constexpr = 96
    with tla.scope.cube():
        query_l1 = tla.alloc((K // 16, M // 16, 16, 16), DTYPE, space='l1')
        key_l1 = tla.alloc((K // 16, N // 16, 16, 16), DTYPE, space='l1')
        query_l0a = tla.alloc((K // 16, M // 16, 16, 16), DTYPE, space='l0a')
        key_l0b = tla.alloc((K // 16, N // 16, 16, 16), DTYPE, space='l0b')
        score = tla.alloc((N // 16, M // 16, 16, 16), tl.float32, space='l0c')
        tla.load_gm_to_l1(tla.view(Q, (M, K)), query_l1)
        tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
        tla.copy_l1_to_l0(query_l1, query_l0a, tla.CopyL1ToL0Params(
            k_part_idx=0, k_part=K, k_part_ceil=K, k_part_loop=1, k_part_actual=K, m=M))

        # Two different K tiles consume the same resident Q tile.
        for page in tl.range(2):
            tla.load_gm_to_l1(tla.view(KEYS + page * N * K, (N, K)), key_l1)
            tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
            tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
            tla.copy_l1_to_l0(key_l1, key_l0b, tla.CopyL1ToL0Params(
                is_transpose=True, k_part_idx=0, k_part=K, k_part_ceil=K,
                k_part_loop=1, k_part_actual=K, m=N))
            tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
            tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
            tla.cube.mmad_l0(query_l0a, key_l0b, score, m=M, n=N, k=K, clear_accum=True)
            tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
            tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
            tla.store_l0c_to_gm(score, tla.view(OUT + page * M * N, (M, N)),
                                tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND))
            tla.sync.barrier(tla.sync.PIPE.ALL)


DTYPES = ('fp16', 'bf16')


@pytest.mark.parametrize('dtype', DTYPES)
def test_ir(dtype):
    element = tl.bfloat16 if dtype == 'bf16' else tl.float16
    sig = dict(Q=f'*{dtype}', KEYS=f'*{dtype}', OUT='*fp32')
    module, ctx = compile_offline(resident_q, sig, dict(DTYPE=element), options=A5Options())
    ir = str(lower_offline(module, ctx))
    assert ir.count('hivm.hir.mmadL0') == 1
    assert ir.count('hivm.hir.l12l0') == 2
    assert 'hivm.hir.mmadL1' not in ir
    assert ir.index('hivm.hir.l12l0') < ir.index('scf.for') < ir.rindex('hivm.hir.l12l0')


@pytest.mark.parametrize('dtype', DTYPES)
def test_run(dtype):
    import torch
    import torch_npu  # noqa: F401

    if not get_stages()[2]:
        pytest.skip('Run stage requires an Ascend NPU')
    is_bf16 = dtype == 'bf16'
    element = tl.bfloat16 if is_bf16 else tl.float16
    torch_dtype = torch.bfloat16 if is_bf16 else torch.float16
    generator = torch.Generator().manual_seed(11)
    q = torch.randn((96, 96), generator=generator).to(torch_dtype)
    keys = torch.randn((2, 128, 96), generator=generator).to(torch_dtype)
    out = torch.full((2, 96, 128), float('nan'), dtype=torch.float32, device='npu')
    kernel = resident_q[(1,)](q.npu(), keys.npu(), out, element, unit_flag=False, sync_solver=False)
    torch.npu.synchronize()
    check_kernel_binary(kernel)
    expected = q.float() @ keys.float().transpose(-2, -1)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-4, rtol=1e-4)
    print('L0 REUSE PASS', dtype, 'max_abs_error', (out.cpu() - expected).abs().max().item())
