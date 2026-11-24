import pytest
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95

import test_common
from test_reduce_min_4_keepdim_True_with_index_op import triton_min_5d_dim4_keepdim


@triton.jit
def _maximum_with_index(left_value, left_index, right_value, right_index):
    choose_left = (left_value > right_value) | ((left_value == right_value) & (left_index < right_index))
    return tl.where(choose_left, left_value, right_value), tl.where(choose_left, left_index, right_index)


@triton.jit
def _maximum_with_index_right(left_value, left_index, right_value, right_index):
    choose_left = (left_value > right_value) | ((left_value == right_value) & (left_index > right_index))
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
    RIGHT_TIE_BREAK: tl.constexpr,
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

    if RIGHT_TIE_BREAK:
        reduced_values, reduced_indices = tl.reduce((values, indices), 1, _maximum_with_index_right)
    else:
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
        RIGHT_TIE_BREAK=False,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(actual_values.cpu(), expected_values)
    assert torch.equal(actual_indices.cpu(), expected_indices.to(torch.int64))


# Reduce over dim 4 with an index operand that carries each element's global
# linear index instead of its position along dim 4. The winning element is the
# same as for the position index, but the returned index must be the global
# value, not the local position. Only dims larger than 1 are used: for a unit
# reduction dim AscendNPU-IR folds the reduce to a squeeze and currently fills
# the index result with zeros (see HFusion FoldUnitDims rewriteNoOpReduce).
_global_index_shapes = [
    (2, 2, 2, 2, 8),
    (9, 3, 2, 4, 17),
    (129, 1, 5, 1, 4),
    (3, 11, 1, 3, 42),
    (2, 91, 4, 2, 4),
    (5, 2, 4, 1, 26),
    (1, 1, 3, 2, 600),
]

_global_index_value_dtypes = [
    'int8',
    'int16',
    'int32',
    'int64',
    'float16',
    'float32',
    'bfloat16',
    'bool',
]
_global_index_dtype_shape = (2, 2, 2, 2, 8)

_global_index_shape_ids = [
    "{}-{}".format(triton_min_5d_dim4_keepdim.__name__, "-".join(map(str, shape))) for shape in _global_index_shapes
]


def _run_global_index_case(sigtype, shape):
    dtype = eval('torch.' + sigtype)
    x0 = torch.randn(shape).to(dtype).npu()

    x1 = torch.arange(x0.numel()).view(x0.shape).npu().to(torch.int32)
    if 'int' in sigtype:
        ans, ans1 = torch.min(x0.to(torch.int64), 4)
        ans = ans.to(dtype)
    else:
        ans, ans1 = torch.min(x0, 4)
    # x1 = global linear index, so the winner's index is the base of its row
    # (x1[..., 0]) plus its local position along dim 4.
    base = torch.arange(x0.numel(), device=x0.device).view(shape)[..., 0]
    ans1 = (ans1 + base).to(torch.int32)
    output = torch.zeros(shape[0:4], dtype=dtype).npu()
    output1 = torch.zeros(shape[0:4], dtype=torch.int32).npu()
    triton_min_5d_dim4_keepdim[(1, )](x0, x1, output, output1, *shape)
    test_common.validate_cmp(sigtype, output, ans)
    test_common.validate_cmp('int32', output1, ans1)


@pytest.mark.skipif(is_compile_on_910_95(), reason="requires A3")
@pytest.mark.parametrize('shape', _global_index_shapes, ids=_global_index_shape_ids)
def test_min_dim4_global_index_shapes(shape):
    _run_global_index_case('float32', shape)


@pytest.mark.skipif(is_compile_on_910_95(), reason="requires A3")
@pytest.mark.parametrize('sigtype', _global_index_value_dtypes)
def test_min_dim4_global_index_dtypes(sigtype):
    _run_global_index_case(sigtype, _global_index_dtype_shape)


@triton.jit
def _split_reduce_kernel(input_ptr, value_ptr, index_ptr, ROWS: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.arange(0, ROWS)
    acc_value = tl.full((ROWS, ), float("-inf"), tl.float32)
    acc_index = tl.zeros((ROWS, ), tl.int32)
    for chunk in tl.static_range(0, N // BLOCK):
        columns = tl.arange(0, BLOCK)
        values = tl.load(input_ptr + rows[:, None] * N + chunk * BLOCK + columns[None, :])
        indices = columns + chunk * BLOCK
        indices = tl.broadcast_to(indices[None, :], (ROWS, BLOCK))
        chunk_value, chunk_index = tl.reduce((values, indices), 1, _maximum_with_index)
        take_chunk = chunk_value > acc_value
        acc_value = tl.where(take_chunk, chunk_value, acc_value)
        acc_index = tl.where(take_chunk, chunk_index, acc_index)
    tl.store(value_ptr + rows, acc_value)
    tl.store(index_ptr + rows, acc_index)


@pytest.mark.skipif(is_compile_on_910_95(), reason="explicit non-canonical index reduction is supported on A3 only")
@pytest.mark.parametrize("block", [4, 2])
def test_reduce_with_index_split_loop_keeps_chunk_offset(block):
    input_cpu = torch.tensor(
        [
            [9, 1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 10],
            [1, 2, 3, 4, 5, 6, 8, 7],
            [5, 1, 2, 3, 4, 9, 6, 7],
        ],
        dtype=torch.float32,
    )
    expected_values, expected_indices = torch.max(input_cpu, dim=1)

    input_npu = input_cpu.npu()
    actual_values = torch.empty(4, dtype=torch.float32, device="npu")
    actual_indices = torch.empty(4, dtype=torch.int32, device="npu")
    _split_reduce_kernel[(1, )](input_npu, actual_values, actual_indices, ROWS=4, N=8, BLOCK=block)
    torch.npu.synchronize()

    torch.testing.assert_close(actual_values.cpu(), expected_values)
    assert torch.equal(actual_indices.cpu(), expected_indices.to(torch.int32))


@triton.jit
def _reversed_index_reduce_kernel(input_ptr, value_output_ptr, index_output_ptr, ROWS: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, ROWS)
    columns = tl.arange(0, N)
    values = tl.load(input_ptr + rows[:, None] * N + columns[None, :])
    indices = (N - 1 - columns).to(tl.int32)
    indices = tl.broadcast_to(indices[None, :], (ROWS, N))
    reduced_values, reduced_indices = tl.reduce((values, indices), 1, _maximum_with_index)
    tl.store(value_output_ptr + rows, reduced_values)
    tl.store(index_output_ptr + rows, reduced_indices)


def _run_reversed_index(input_cpu, expected_indices):
    input_npu = input_cpu.npu()
    actual_values = torch.empty(input_cpu.shape[0], dtype=torch.float32, device="npu")
    actual_indices = torch.empty(input_cpu.shape[0], dtype=torch.int32, device="npu")
    _reversed_index_reduce_kernel[(1, )](
        input_npu,
        actual_values,
        actual_indices,
        ROWS=input_cpu.shape[0],
        N=input_cpu.shape[1],
    )
    torch.npu.synchronize()

    expected_values, _ = torch.max(input_cpu, dim=1)
    torch.testing.assert_close(actual_values.cpu(), expected_values)
    assert torch.equal(actual_indices.cpu(), torch.tensor(expected_indices, dtype=torch.int32))


@pytest.mark.skipif(is_compile_on_910_95(), reason="explicit non-canonical index reduction is supported on A3 only")
def test_reduce_with_index_nonmonotonic():
    # The index decreases along the reduce axis, so the returned index can not be
    # recovered by adding a per-row base to the winner's position. Values are
    # unique on purpose: ties are lowered to "prefer the left/right element",
    # not to a comparison of index values.
    input_cpu = torch.tensor(
        [
            [1, 2, 3, 4, 5, 9, 7, 8],
            [1, 8, 3, 4, 5, 6, 7, 2],
            [9, 1, 1, 1, 1, 1, 1, 8],
            [1, 2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=torch.float32,
    )
    # Winner positions are 5, 1, 0, 7; reversed indices are 2, 6, 7, 0.
    _run_reversed_index(input_cpu, [2, 6, 7, 0])


def test_reduce_with_index_tie_break_prefers_left():
    # Deterministic ties: with a canonical iota, "prefer left" and "prefer the
    # smaller index" are the same thing, so torch.max gives the expected index.
    input_cpu = torch.tensor(
        [
            [1, 2, 9, 4, 5, 6, 9, 8],
            [1, 9, 1, 1, 1, 1, 1, 9],
            [5, 2, 2, 2, 2, 2, 2, 5],
            [5, 1, 2, 3, 4, 9, 6, 7],
        ],
        dtype=torch.float32,
    )
    expected_values, expected_indices = torch.max(input_cpu, dim=1)

    input_npu = input_cpu.npu()
    actual_values = torch.empty(4, dtype=torch.float32, device="npu")
    actual_indices = torch.empty(4, dtype=torch.int32, device="npu")
    _canonical_index_reduce_kernel[(1, )](
        input_npu,
        actual_values,
        actual_indices,
        BLOCK_M=4,
        BLOCK_N=8,
        INDEX_DTYPE=tl.int32,
        USE_RESHAPE=False,
        RIGHT_TIE_BREAK=False,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(actual_values.cpu(), expected_values)
    assert torch.equal(actual_indices.cpu(), expected_indices.to(torch.int32))


def test_reduce_with_index_tie_break_prefers_right():
    # Mirror of the left-tie test with the opposite tie-break: canonical index
    # means "prefer right" is the last occurrence of the maximum.
    input_cpu = torch.tensor(
        [
            [1, 2, 9, 4, 5, 6, 9, 8],
            [1, 9, 1, 1, 1, 1, 1, 9],
            [5, 2, 2, 2, 2, 2, 2, 5],
            [5, 1, 2, 3, 4, 9, 6, 7],
        ],
        dtype=torch.float32,
    )
    expected_values, _ = torch.max(input_cpu, dim=1)
    expected_indices = torch.tensor([6, 7, 7, 5], dtype=torch.int32)

    input_npu = input_cpu.npu()
    actual_values = torch.empty(4, dtype=torch.float32, device="npu")
    actual_indices = torch.empty(4, dtype=torch.int32, device="npu")
    _canonical_index_reduce_kernel[(1, )](
        input_npu,
        actual_values,
        actual_indices,
        BLOCK_M=4,
        BLOCK_N=8,
        INDEX_DTYPE=tl.int32,
        USE_RESHAPE=False,
        RIGHT_TIE_BREAK=True,
    )
    torch.npu.synchronize()

    torch.testing.assert_close(actual_values.cpu(), expected_values)
    assert torch.equal(actual_indices.cpu(), expected_indices)


@triton.jit
def _minimum_with_index_nan(left_value, left_index, right_value, right_index):
    left_nan = left_value != left_value
    right_nan = right_value != right_value
    equal = (left_value == right_value) | (left_nan & right_nan)
    choose_left = (left_value < right_value) | (left_nan & (~right_nan)) | (equal & (left_index < right_index))
    return tl.where(choose_left, left_value, right_value), tl.where(choose_left, left_index, right_index)


@triton.jit
def _nan_index_reduce_kernel(input_ptr, value_output_ptr, index_output_ptr, BLOCK_M: tl.constexpr,
                             BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    columns = tl.arange(0, BLOCK_N)
    values = tl.load(input_ptr + rows[:, None] * BLOCK_N + columns[None, :])
    indices = tl.broadcast_to(columns[None, :], (BLOCK_M, BLOCK_N))
    reduced_values, reduced_indices = tl.reduce((values, indices), 1, _minimum_with_index_nan)
    tl.store(value_output_ptr + rows, reduced_values)
    tl.store(index_output_ptr + rows, reduced_indices)


@pytest.mark.skipif(is_compile_on_910_95(), reason="NaN reduction semantics are verified on A3 only")
def test_reduce_with_index_propagates_nan():
    input_cpu = torch.tensor(
        [
            [1, float("nan"), 2, 3, float("nan"), 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [float("nan"), 2, float("nan"), 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7, float("nan")],
        ],
        dtype=torch.float32,
    )
    expected_values, expected_indices = torch.min(input_cpu, dim=1)

    input_npu = input_cpu.npu()
    actual_values = torch.empty(4, dtype=torch.float32, device="npu")
    actual_indices = torch.empty(4, dtype=torch.int32, device="npu")
    _nan_index_reduce_kernel[(1, )](input_npu, actual_values, actual_indices, BLOCK_M=4, BLOCK_N=8)
    torch.npu.synchronize()

    torch.testing.assert_close(actual_values.cpu(), expected_values, equal_nan=True)
    assert torch.equal(actual_indices.cpu(), expected_indices.to(torch.int32))
