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
    kernel[grid](indices.npu(), data.npu(), output, N, M, BLOCK_N=block_n, BLOCK_M=block_m,
                 compile_mode="simd")
    test_common.validate_cmp("float32", output, expected)
