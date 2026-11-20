import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _maximum_with_index(left_value, left_index, right_value, right_index):
    choose_left = (left_value > right_value) | ((left_value == right_value) & (left_index < right_index))
    return tl.where(choose_left, left_value, right_value), tl.where(choose_left, left_index, right_index)


@triton.jit
def _canonical_index_reduce_kernel(
    input_ptr,
    value_output_ptr,
    index_output_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    INDEX_DTYPE: tl.constexpr,
    USE_RESHAPE: tl.constexpr,
):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    values = tl.load(input_ptr + rows[:, None] * BLOCK_N + columns[None, :])

    indices = columns.to(INDEX_DTYPE)
    if USE_RESHAPE:
        indices = indices.reshape(1, BLOCK_N)
    else:
        indices = indices[None, :]
    indices = tl.broadcast_to(indices, (BLOCK_M, BLOCK_N))

    reduced_values, reduced_indices = tl.reduce((values, indices), 1, _maximum_with_index)
    tl.store(value_output_ptr + rows, reduced_values)
    tl.store(index_output_ptr + rows, reduced_indices)


@pytest.mark.parametrize("use_reshape", [False, True], ids=["expand_dims", "reshape"])
def test_reduce_with_index_canonical_cast_reshape(use_reshape):
    input_cpu = torch.tensor(
        [
            [8, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 9, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 10],
            [1, 2, 3, 4, 12, 5, 6, 7],
        ],
        dtype=torch.float32,
    )
    expected_values, expected_indices = torch.max(input_cpu, dim=1)

    input_npu = input_cpu.npu()
    actual_values = torch.empty(4, dtype=torch.float32, device="npu")
    actual_indices = torch.empty(4, dtype=torch.int64, device="npu")
    _canonical_index_reduce_kernel[(1, )](
        input_npu,
        actual_values,
        actual_indices,
        BLOCK_M=4,
        BLOCK_N=8,
        INDEX_DTYPE=tl.int64,
        USE_RESHAPE=use_reshape,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(actual_values.cpu(), expected_values)
    assert torch.equal(actual_indices.cpu(), expected_indices.to(torch.int64))
