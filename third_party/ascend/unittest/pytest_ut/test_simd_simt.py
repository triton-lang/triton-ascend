import torch
import pytest
import triton
import triton.language as tl
from triton.tools.get_ascend_devices import is_compile_on_910_95

simd_simt_910_95_only = pytest.mark.xfail(not is_compile_on_910_95,
                                          reason="simd_simt compile mode only supports 910_95", run=False)


@triton.jit
def simple_indirect_load_add_kernel(src_ptr, indices_ptr, out_ptr, add_value, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    index = tl.load(indices_ptr + idx)
    value = tl.load(src_ptr + index)

    # 增加一个 add 计算
    value = value + add_value

    tl.store(out_ptr + idx, value)


@simd_simt_910_95_only
def test_simple_indirect_load_add():
    src = torch.arange(256, dtype=torch.float32, device='npu')
    print(f"Source data: [{src[0]}, {src[1]}, ..., {src[255]}]")

    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255], dtype=torch.int64, device='npu')
    print(f"Indices: {indices.tolist()}")

    output = torch.zeros(8, dtype=torch.float32, device='npu')

    add_value = 3.0
    grid = (1, )
    simple_indirect_load_add_kernel[grid](src, indices, output, add_value, BLOCK=8, compile_mode='simd_simt')

    expected = src[indices] + add_value

    print(f"Expected: {expected.tolist()}")
    print(f"Got:      {output.tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    return True


@triton.jit
def indirect_index_load_div_kernel(src_ptr, add_ptr, indices_ptr, out_ptr, offset1_ptr, offset2_ptr,
                                   BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    index = tl.load(indices_ptr + idx)
    off1 = tl.load(offset1_ptr + idx)
    off2 = tl.load(offset2_ptr + idx)
    tmp = index + (off2 / off1).to(tl.int64) * 2
    value = tl.load(src_ptr + tmp)

    # 增加一个 add 计算
    add_value = tl.load(add_ptr + idx)
    value = value + add_value

    tl.store(out_ptr + idx, value)


@simd_simt_910_95_only
def test_indirect_index_load_div_kernel():
    src = torch.arange(256, dtype=torch.float32, device='npu')
    add_src = torch.arange(8, dtype=torch.float32, device='npu')
    print(f"Source data: [{src[0]}, {src[1]}, ..., {src[255]}]")

    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255], dtype=torch.int64, device='npu')
    offset1 = torch.arange(1, 9, dtype=torch.int64, device='npu')
    offset2 = torch.full((8, ), 3, dtype=torch.int64, device='npu')
    print(f"Indices: {indices.tolist()}")

    output = torch.zeros(8, dtype=torch.float32, device='npu')

    # add_value = 3.0
    grid = (1, )
    indirect_index_load_div_kernel[grid](src, add_src, indices, output, offset1, offset2, BLOCK=8,
                                         compile_mode='simd_simt')

    expected = src[indices + (offset2 / offset1).to(torch.int64) * 2] + add_src

    print(f"Expected: {expected.tolist()}")
    print(f"Got:      {output.tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    return True


@triton.jit
def indirect_index_load_add_kernel(src_ptr, add_ptr, indices_ptr, out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    index = tl.load(indices_ptr + idx)
    tmp = index + 3
    value = tl.load(src_ptr + tmp)

    # 增加一个 add 计算
    add_value = tl.load(add_ptr + idx)
    value = value + add_value

    tl.store(out_ptr + idx, value)


@simd_simt_910_95_only
def test_indirect_index_load_add_kernel():
    src = torch.arange(256, dtype=torch.float32, device='npu')
    add_src = torch.arange(8, dtype=torch.float32, device='npu')
    print(f"Source data: [{src[0]}, {src[1]}, ..., {src[255]}]")

    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255], dtype=torch.int64, device='npu')
    print(f"Indices: {indices.tolist()}")

    output = torch.zeros(8, dtype=torch.float32, device='npu')

    # add_value = 3.0
    grid = (1, )
    indirect_index_load_add_kernel[grid](src, add_src, indices, output, BLOCK=8, compile_mode='simd_simt')

    expected = src[indices + 3] + add_src

    print(f"Expected: {expected.tolist()}")
    print(f"Got:      {output.tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    return True


@triton.jit
def indirect_index_load_and_kernel(src_ptr, add_ptr, indices_ptr, out_ptr, offset1_ptr, offset2_ptr,
                                   BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    index = tl.load(indices_ptr + idx)
    off1 = tl.load(offset1_ptr + idx)
    off2 = tl.load(offset2_ptr + idx)
    tmp = index & 63
    value = tl.load(src_ptr + tmp)

    # 增加一个 add 计算
    add_value = tl.load(add_ptr + idx)
    value = value + add_value

    tl.store(out_ptr + idx, value)


@simd_simt_910_95_only
def test_indirect_index_load_and_kernel():
    src = torch.arange(256, dtype=torch.float32, device='npu')
    add_src = torch.arange(8, dtype=torch.float32, device='npu')
    print(f"Source data: [{src[0]}, {src[1]}, ..., {src[255]}]")

    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255], dtype=torch.int64, device='npu')
    offset1 = torch.arange(8, dtype=torch.int64, device='npu')
    offset2 = torch.full((8, ), 3, dtype=torch.int64, device='npu')
    print(f"Indices: {indices.tolist()}")

    output = torch.zeros(8, dtype=torch.float32, device='npu')

    # add_value = 3.0
    grid = (1, )
    indirect_index_load_and_kernel[grid](src, add_src, indices, output, offset1, offset2, BLOCK=8,
                                         compile_mode='simd_simt')

    expected = src[indices & 63] + add_src

    print(f"Expected: {expected.tolist()}")
    print(f"Got:      {output.tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    return True


@triton.jit
def indirect_index_load_max_min_kernel(src_ptr, add_ptr, indices_ptr, out_ptr, offset1_ptr, offset2_ptr,
                                       BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    index = tl.load(indices_ptr + idx)
    off1 = tl.load(offset1_ptr + idx)
    off2 = tl.load(offset2_ptr + idx)
    tmp = tl.minimum(tl.maximum(index, 3), 64)
    value = tl.load(src_ptr + tmp)

    # 增加一个 add 计算
    add_value = tl.load(add_ptr + idx)
    value = value + add_value

    tl.store(out_ptr + idx, value)


@simd_simt_910_95_only
def test_indirect_index_load_max_min_kernel():
    src = torch.arange(256, dtype=torch.float32, device='npu')
    add_src = torch.arange(8, dtype=torch.float32, device='npu')
    print(f"Source data: [{src[0]}, {src[1]}, ..., {src[255]}]")

    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255], dtype=torch.int64, device='npu')
    offset1 = torch.arange(8, dtype=torch.int64, device='npu')
    offset2 = torch.full((8, ), 3, dtype=torch.int64, device='npu')
    print(f"Indices: {indices.tolist()}")

    output = torch.zeros(8, dtype=torch.float32, device='npu')

    # add_value = 3.0
    grid = (1, )
    indirect_index_load_max_min_kernel[grid](src, add_src, indices, output, offset1, offset2, BLOCK=8,
                                             compile_mode='simd_simt')
    down_limit = torch.full((8, ), 0, dtype=torch.int64, device='npu')
    up_limit = torch.full((8, ), 64, dtype=torch.int64, device='npu')
    expected = src[torch.minimum(torch.maximum(indices, down_limit), up_limit)] + add_src

    print(f"Expected: {expected.tolist()}")
    print(f"Got:      {output.tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    return True


@triton.jit
def indirect_index_load_mul_kernel(src_ptr, add_ptr, indices_ptr, out_ptr, offset1_ptr, offset2_ptr,
                                   BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    index = tl.load(indices_ptr + idx)
    off1 = tl.load(offset1_ptr + idx)
    off2 = tl.load(offset2_ptr + idx)
    tmp = index + off2 * off1
    value = tl.load(src_ptr + tmp)

    # 增加一个 add 计算
    add_value = tl.load(add_ptr + idx)
    value = value + add_value

    tl.store(out_ptr + idx, value)


@simd_simt_910_95_only
def test_indirect_index_load_mul_kernel():
    src = torch.arange(256, dtype=torch.float32, device='npu')
    add_src = torch.arange(8, dtype=torch.float32, device='npu')
    print(f"Source data: [{src[0]}, {src[1]}, ..., {src[255]}]")

    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255], dtype=torch.int64, device='npu')
    offset1 = torch.arange(8, dtype=torch.int64, device='npu')
    offset2 = torch.full((8, ), 3, dtype=torch.int64, device='npu')
    print(f"Indices: {indices.tolist()}")

    output = torch.zeros(8, dtype=torch.float32, device='npu')

    # add_value = 3.0
    grid = (1, )
    indirect_index_load_mul_kernel[grid](src, add_src, indices, output, offset1, offset2, BLOCK=8,
                                         compile_mode='simd_simt')

    expected = src[indices + offset2 * offset1] + add_src

    print(f"Expected: {expected.tolist()}")
    print(f"Got:      {output.tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    return True


@triton.jit
def indirect_index_load_or_kernel(src_ptr, add_ptr, indices_ptr, out_ptr, offset1_ptr, offset2_ptr,
                                  BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    index = tl.load(indices_ptr + idx)
    off1 = tl.load(offset1_ptr + idx)
    off2 = tl.load(offset2_ptr + idx)
    tmp = index | 3
    value = tl.load(src_ptr + tmp)

    # 增加一个 add 计算
    add_value = tl.load(add_ptr + idx)
    value = value + add_value

    tl.store(out_ptr + idx, value)


@simd_simt_910_95_only
def test_indirect_index_load_or_kernel():
    src = torch.arange(256, dtype=torch.float32, device='npu')
    add_src = torch.arange(8, dtype=torch.float32, device='npu')
    print(f"Source data: [{src[0]}, {src[1]}, ..., {src[255]}]")

    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255], dtype=torch.int64, device='npu')
    offset1 = torch.arange(8, dtype=torch.int64, device='npu')
    offset2 = torch.full((8, ), 3, dtype=torch.int64, device='npu')
    print(f"Indices: {indices.tolist()}")

    output = torch.zeros(8, dtype=torch.float32, device='npu')

    # add_value = 3.0
    grid = (1, )
    indirect_index_load_or_kernel[grid](src, add_src, indices, output, offset1, offset2, BLOCK=8,
                                        compile_mode='simd_simt')

    expected = src[indices | 3] + add_src

    print(f"Expected: {expected.tolist()}")
    print(f"Got:      {output.tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    return True


@triton.jit
def indirect_index_load_sub_kernel(src_ptr, add_ptr, indices_ptr, out_ptr, offset1_ptr, offset2_ptr,
                                   BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    index = tl.load(indices_ptr + idx)
    off1 = tl.load(offset1_ptr + idx)
    off2 = tl.load(offset2_ptr + idx)
    tmp = index + off2 - off1
    value = tl.load(src_ptr + tmp)

    # 增加一个 add 计算
    add_value = tl.load(add_ptr + idx)
    value = value + add_value

    tl.store(out_ptr + idx, value)


@simd_simt_910_95_only
def test_indirect_index_load_sub_kernel():
    src = torch.arange(256, dtype=torch.float32, device='npu')
    add_src = torch.arange(8, dtype=torch.float32, device='npu')
    print(f"Source data: [{src[0]}, {src[1]}, ..., {src[255]}]")

    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255], dtype=torch.int64, device='npu')
    offset1 = torch.arange(8, dtype=torch.int64, device='npu')
    offset2 = torch.full((8, ), 3, dtype=torch.int64, device='npu')
    print(f"Indices: {indices.tolist()}")

    output = torch.zeros(8, dtype=torch.float32, device='npu')

    # add_value = 3.0
    grid = (1, )
    indirect_index_load_sub_kernel[grid](src, add_src, indices, output, offset1, offset2, BLOCK=8,
                                         compile_mode='simd_simt')

    expected = src[indices + offset2 - offset1] + add_src

    print(f"Expected: {expected.tolist()}")
    print(f"Got:      {output.tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    return True


@triton.jit
def simple_indirect_load_2d_min_kernel(
    src_ptr,  # [M, N]
    indices_ptr,  # [M, K]
    out_ptr,  # [M, K]
    N,
    M: tl.constexpr,
    K: tl.constexpr,
):
    offs_m = tl.arange(0, M)[:, None]  # [M, 1]
    offs_k = tl.arange(0, K)[None, :]  # [1, K]

    gathered_idx = tl.load(indices_ptr + offs_m * K + offs_k)  # [M, K]
    values = tl.load(src_ptr + offs_m * N + gathered_idx)  # [M, K]
    tl.store(out_ptr + offs_m * K + offs_k, values)


@simd_simt_910_95_only
def test_simple_indirect_load_2d_min():
    M, N = 4, 16
    K = 8

    src = torch.arange(M * N, dtype=torch.float32, device='npu').reshape(M, N).contiguous()
    print("Source data:")
    print(src)

    indices = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [15, 14, 13, 12, 11, 10, 9, 8],
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 0],
    ], dtype=torch.int64, device='npu').contiguous()
    print("Indices:")
    print(indices)

    output = torch.empty((M, K), dtype=torch.float32, device='npu').contiguous()

    simple_indirect_load_2d_min_kernel[(1, )](src, indices, output, N, M=M, K=K, compile_mode='simd_simt')

    expected = torch.gather(src, dim=1, index=indices)

    print("Expected:")
    print(expected)

    print("Got:")
    print(output)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    print("PASS")
    return True


@triton.jit
def simple_indirect_load_2d_kernel(
    src_ptr,  # [M, N]
    indices_ptr,  # [M, K]
    out_ptr,  # [M, K]
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    offs_m_2d = offs_m[:, None]  # [BLOCK_M, 1]
    offs_k_2d = offs_k[None, :]  # [1, BLOCK_K]

    # indices[m, k]
    idx_ptrs = indices_ptr + offs_m_2d * K + offs_k_2d
    gathered_idx = tl.load(idx_ptrs)

    # src[m, indices[m, k]]
    src_ptrs = src_ptr + offs_m_2d * N + gathered_idx
    values = tl.load(src_ptrs)

    # out[m, k]
    out_ptrs = out_ptr + offs_m_2d * K + offs_k_2d
    tl.store(out_ptrs, values)


@simd_simt_910_95_only
def test_simple_indirect_load_2d():
    M, N = 4, 16
    K = 8

    src = torch.arange(M * N, dtype=torch.float32, device='npu').reshape(M, N).contiguous()
    print("Source data:")
    print(src)

    indices = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [15, 14, 13, 12, 11, 10, 9, 8],
        [1, 3, 5, 7, 9, 11, 13, 15],
        [2, 4, 6, 8, 10, 12, 14, 0],
    ], dtype=torch.int64, device='npu').contiguous()
    print("Indices:")
    print(indices)

    output = torch.empty((M, K), dtype=torch.float32, device='npu').contiguous()

    BLOCK_M = 2
    BLOCK_K = 4

    # 要求 M % BLOCK_M == 0 且 K % BLOCK_K == 0
    grid = (M // BLOCK_M, K // BLOCK_K)

    simple_indirect_load_2d_kernel[grid](src, indices, output, N, K, BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
                                         compile_mode='simd_simt')

    expected = torch.gather(src, dim=1, index=indices)

    print("Expected:")
    print(expected)

    print("Got:")
    print(output)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    return True


@triton.jit
def simple_indirect_load_kernel(src_ptr, indices_ptr, out_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    index = tl.load(indices_ptr + idx)
    value = tl.load(src_ptr + index)

    # Store result
    tl.store(out_ptr + idx, value)


@simd_simt_910_95_only
def test_simple_indirect_load():
    src = torch.arange(256, dtype=torch.float32, device='npu')
    print(f"Source data: [{src[0]}, {src[1]}, ..., {src[255]}]")

    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255], dtype=torch.int64, device='npu')
    print(f"Indices: {indices.tolist()}")

    output = torch.zeros(8, dtype=torch.float32, device='npu')

    grid = (1, )
    simple_indirect_load_kernel[grid](src, indices, output, BLOCK=8, compile_mode='simd_simt')

    expected = src[indices]

    print(f"Expected: {expected.tolist()}")
    print(f"Got:      {output.tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)

    return True


@triton.jit
def index_put_kernel(
    value_ptr,  # [M, D] values to scatter
    indices_ptr,  # [M] index tensor
    dst_ptr,  # [N, D] destination tensor
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    index_put on dim=0:
        dst[indices[i], j] = value[i, j]

    Pointer arithmetic:
        ptr = dst_ptr + indices[i][:, None] * D + arange(0, D)[None, :]
    dim 0 is unstructured (indirect), dim 1 is structured (contiguous).
    Expected: unstructuredDims=[0], burstlen=D.
    """
    row_idx = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    col_idx = tl.arange(0, D)  # [D]

    # Load indirect indices
    indices = tl.load(indices_ptr + row_idx)  # [BLOCK_M], int64

    # Load values to scatter
    val_offsets = row_idx[:, None] * D + col_idx[None, :]
    values = tl.load(value_ptr + val_offsets)  # [BLOCK_M, D]

    # Build 2D pointer for scatter: dst_ptr + indices[:, None] * D + col_idx[None, :]
    ptr = dst_ptr + indices[:, None] * D + col_idx[None, :]  # [BLOCK_M, D]

    # This store should be semi-structured:
    #   dim 0 -> unstructured (from indices)
    #   dim 1 -> structured (from arange)
    tl.store(ptr, values)


@simd_simt_910_95_only
def test_index_put():
    N, D, M = 64, 32, 8

    values = torch.arange(M * D, dtype=torch.float32, device='npu').reshape(M, D)
    indices = torch.tensor([3, 10, 0, 55, 7, 20, 63, 1], dtype=torch.int64, device='npu')
    assert indices.shape[0] == M

    output = torch.zeros(N, D, dtype=torch.float32, device='npu')

    grid = (1, )
    index_put_kernel[grid](
        values,
        indices,
        output,
        D=D,
        BLOCK_M=M,
        compile_mode='simd_simt',
    )

    # Build expected result
    expected = torch.zeros(N, D, dtype=torch.float32, device='npu')
    expected[indices] = values

    print(f"values shape: {values.shape}")
    print(f"indices: {indices.tolist()}")
    print(f"expected[3]:  {expected[3].tolist()}")
    print(f"got[3]:       {output[3].tolist()}")
    print(f"expected[10]: {expected[10].tolist()}")
    print(f"got[10]:      {output[10].tolist()}")
    # Check a row that was NOT written to — should be zeros
    print(f"expected[2]:  {expected[2].tolist()}")
    print(f"got[2]:       {output[2].tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    print("index_put test PASSED")
    return True


@triton.jit
def index_select_kernel(
    src_ptr,  # [N, D] source tensor, row-major
    indices_ptr,  # [M] index tensor
    out_ptr,  # [M, D] output tensor
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    index_select on dim=0:
        result[i, j] = src[indices[i], j]

    Pointer arithmetic:
        ptr = src_ptr + indices[i][:, None] * D + arange(0, D)[None, :]
    dim 0 is unstructured (indirect), dim 1 is structured (contiguous).
    Expected: unstructuredDims=[0], burstlen=D.
    """
    row_idx = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    col_idx = tl.arange(0, D)  # [D]

    # Load indirect indices
    indices = tl.load(indices_ptr + row_idx)  # [BLOCK_M], int64

    # Build 2D pointer: src_ptr + indices[:, None] * D + col_idx[None, :]
    ptr = src_ptr + indices[:, None] * D + col_idx[None, :]  # [BLOCK_M, D]

    # This load should be semi-structured:
    #   dim 0 -> unstructured (from indices)
    #   dim 1 -> structured (from arange)
    values = tl.load(ptr)  # [BLOCK_M, D]

    # Store result contiguously
    out_offsets = row_idx[:, None] * D + col_idx[None, :]
    tl.store(out_ptr + out_offsets, values)


@simd_simt_910_95_only
def test_index_select():
    N, D, M = 64, 32, 8

    src = torch.arange(N * D, dtype=torch.float32, device='npu').reshape(N, D)
    indices = torch.tensor([3, 10, 0, 55, 7, 20, 63, 1], dtype=torch.int64, device='npu')
    assert indices.shape[0] == M

    output = torch.zeros(M, D, dtype=torch.float32, device='npu')

    grid = (1, )
    index_select_kernel[grid](
        src,
        indices,
        output,
        D=D,
        BLOCK_M=M,
        compile_mode='simd_simt',
    )

    expected = src[indices]  # [M, D]

    print(f"src shape: {src.shape}")
    print(f"indices: {indices.tolist()}")
    print(f"expected[0]: {expected[0].tolist()}")
    print(f"got[0]:      {output[0].tolist()}")
    print(f"expected[7]: {expected[7].tolist()}")
    print(f"got[7]:      {output[7].tolist()}")

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    print("index_select test PASSED")
    return True


@triton.jit
def simple_indirect_store_kernel(dst_ptr, indices_ptr, values_ptr, BLOCK: tl.constexpr):
    idx = tl.arange(0, BLOCK)
    index = tl.load(indices_ptr + idx)
    value = tl.load(values_ptr + idx)
    tl.store(dst_ptr + index, value)


@simd_simt_910_95_only
def test_simple_indirect_store():
    dst = torch.zeros(256, dtype=torch.float32, device='npu')

    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255], dtype=torch.int64, device='npu')
    print(f"Indices to write at: {indices.tolist()}")

    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float32, device='npu')
    print(f"Values to write: {values.tolist()}")

    grid = (1, )
    simple_indirect_store_kernel[grid](dst, indices, values, BLOCK=8, compile_mode='simd_simt')

    for i in range(len(indices)):
        idx = indices[i].item()
        expected = values[i].item()
        actual = dst[idx].item()
        assert abs(actual - expected) < 1e-5, \
            f"Mismatch at dst[{idx}]: expected {expected}, got {actual}"

    written_indices = set(indices.tolist())
    for i in range(256):
        if i not in written_indices:
            assert abs(dst[i].item()) < 1e-5, \
                f"dst[{i}] should be 0, but got {dst[i].item()}"
    for i in range(len(indices)):
        p = indices[i]
        torch.testing.assert_close(dst[p], values[i], rtol=1e-5, atol=1e-5)

    return True
