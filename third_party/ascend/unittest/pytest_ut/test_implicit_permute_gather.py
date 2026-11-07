import pytest
import torch
import torch_npu
import triton
import triton.language as tl

import test_common


@triton.jit
def implicit_permute_gather_kernel(in_ptr0, in_ptr1, out_ptr0, N, M, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """Implicit transpose with data-dependent gather offsets along the innermost axis."""
    pid = tl.program_id(0)
    row = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    col = tl.arange(0, BLOCK_M)[None, :]
    row_mask = row < N
    col_mask = col < M
    offsets_x = tl.load(in_ptr0 + col, mask=col_mask, other=0)
    addr = offsets_x * N + row
    mask = row_mask & col_mask
    val = tl.load(in_ptr1 + addr, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (row * M + col), val, mask=mask)


@triton.jit
def row_gather_load_kernel(in_ptr0, in_ptr1, out_ptr0, N, M, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """Row gather with a contiguous innermost axis, without implicit transpose."""
    pid = tl.program_id(0)
    row = pid * BLOCK_N + tl.arange(0, BLOCK_N)[:, None]
    col = tl.arange(0, BLOCK_M)[None, :]
    row_mask = row < N
    col_mask = col < M
    offsets_y = tl.load(in_ptr0 + row, mask=row_mask, other=0)
    addr = offsets_y * M + col
    mask = row_mask & col_mask
    val = tl.load(in_ptr1 + addr, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + (row * M + col), val, mask=mask)


@pytest.mark.parametrize("N,M", [
    (128, 64),
    (100, 50),
    (257, 64),
    (128, 8),
])
@pytest.mark.parametrize("variant", ["implicit_permute_gather", "row_gather"])
def test_implicit_permute_gather(variant, N, M):
    torch.manual_seed(0)
    n_rows = 23
    block_n = 128
    block_m = triton.next_power_of_2(M)
    implicit_permute_gather = variant == "implicit_permute_gather"

    indices = torch.randint(0, n_rows, (M if implicit_permute_gather else N, ), dtype=torch.int32)
    data = test_common.generate_tensor((n_rows, N if implicit_permute_gather else M), "float16")
    expected = data[indices.long()].float()
    if implicit_permute_gather:
        expected = expected.t()

    output = torch.empty((N, M), dtype=torch.float32, device="npu")
    kernel = implicit_permute_gather_kernel if implicit_permute_gather else row_gather_load_kernel
    grid = (triton.cdiv(N, block_n), )
    kernel[grid](indices.npu(), data.npu(), output, N, M, BLOCK_N=block_n, BLOCK_M=block_m, compile_mode="simd")
    test_common.validate_cmp("float32", output, expected)


@triton.jit
def data_mask_gather_kernel(indices, data, output, D: tl.constexpr, N: tl.constexpr, BLOCK_D: tl.constexpr,
                            BLOCK_N: tl.constexpr, BROADCAST_MASK: tl.constexpr):
    """Discrete broadcast mask, with a non-broadcast fallback variant."""
    d = tl.arange(0, BLOCK_D)
    n = tl.arange(0, BLOCK_N)
    token = tl.load(indices + n, mask=n < N, other=-1)
    mask = (token >= 0)[None, :]
    if not BROADCAST_MASK:
        # This data-dependent conjunction is not itself a broadcast and falls
        # back to the original load until mask analysis supports it.
        mask = mask & (d < D)[:, None]
    value = tl.load(data + token[None, :] * D + d[:, None], mask=mask, other=0.0)
    tl.store(output + d[:, None] * N + n[None, :], value, mask=(d < D)[:, None] & (n < N)[None, :])


@pytest.mark.parametrize("D,N,broadcast_mask", [(64, 64, True), (128, 64, True), (100, 50, False)])
def test_implicit_permute_data_mask(D, N, broadcast_mask):
    torch.manual_seed(0)
    n_rows = 23
    block_n = triton.next_power_of_2(N)
    indices = torch.arange(N, dtype=torch.int32) % n_rows
    # Interleaved invalid lanes cannot be represented by one interval bound.
    indices[1::3] = -1
    data = torch.randn((n_rows, D), dtype=torch.float32)
    gathered = data[indices.clamp(min=0).long()].t()
    expected = torch.where((indices >= 0)[None, :], gathered, 0.0)
    output = torch.empty((D, N), dtype=torch.float32, device="npu")
    data_mask_gather_kernel[(1, )](indices.npu(), data.npu(), output, D, N, BLOCK_D=triton.next_power_of_2(D),
                                   BLOCK_N=block_n, BROADCAST_MASK=broadcast_mask, compile_mode="simd")
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)


@triton.jit
def tensor_other_gather_kernel(indices, data, output, D: tl.constexpr, N: tl.constexpr, VALID_N: tl.constexpr,
                               BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr):
    """Non-uniform tensor other with an analyzable mask must retain its lanes."""
    d = tl.arange(0, BLOCK_D)
    n = tl.arange(0, BLOCK_N)
    token = tl.load(indices + n, mask=n < N, other=0)
    mask = (d < D)[:, None] & (n < VALID_N)[None, :]
    other = (d[:, None] * BLOCK_N + n[None, :] + 1).to(tl.float32)
    value = tl.load(data + token[None, :] * D + d[:, None], mask=mask, other=other)
    tl.store(output + d[:, None] * N + n[None, :], value, mask=(d < D)[:, None] & (n < N)[None, :])


@pytest.mark.parametrize("D,N", [(64, 64), (128, 64), (100, 50)])
def test_implicit_permute_tensor_other(D, N):
    torch.manual_seed(0)
    n_rows = 23
    block_n = triton.next_power_of_2(N)
    # Store all N columns so the masked load's tensor other is observable.
    valid_n = N // 2
    indices = torch.arange(N, dtype=torch.int32) % n_rows
    data = torch.randn((n_rows, D), dtype=torch.float32)
    gathered = data[indices.long()].t()
    other = (torch.arange(D)[:, None] * block_n + torch.arange(N)[None, :] + 1).float()
    expected = torch.where((torch.arange(N) < valid_n)[None, :], gathered, other)
    output = torch.empty((D, N), dtype=torch.float32, device="npu")
    tensor_other_gather_kernel[(1, )](indices.npu(), data.npu(), output, D, N, VALID_N=valid_n,
                                      BLOCK_D=triton.next_power_of_2(D), BLOCK_N=block_n, compile_mode="simd")
    torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
