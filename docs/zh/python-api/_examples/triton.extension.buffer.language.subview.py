import torch
import torch_npu
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
import pytest


@triton.jit
def test_subview_kernel_2d(in_ptr0, out_ptr0, shape_0: tl.constexpr, shape_1: tl.constexpr, offsets: tl.constexpr,
                           sizes_0: tl.constexpr, sizes_1: tl.constexpr, strides: tl.constexpr):
    lblk_idx = tl.arange(0, shape_0)
    mblk_idx = tl.arange(0, shape_1)
    idx = lblk_idx[:, None] * shape_1 + mblk_idx[None, :]
    input = tl.load(in_ptr0 + idx)
    lblk_odx = tl.arange(0, sizes_0)
    mblk_odx = tl.arange(0, sizes_1)
    odx = lblk_odx[:, None] * sizes_1 + mblk_odx[None, :]
    src_buffer = bl.to_buffer(input, al.ascend_address_space.UB)
    result_buffer = bl.subview(src_buffer, offsets, [sizes_0, sizes_1], strides)
    result_tensor = bl.to_tensor(result_buffer)
    tl.store(out_ptr0 + odx, result_tensor)


test_list = [
    # (src_shape, offsets, sizes, strides)
    ([100, 32], [1, 0], [10, 10], [1, 1]),
]


@pytest.mark.parametrize('src_shape, offsets, sizes, strides', test_list)
def test_subview(
    src_shape,
    offsets,
    sizes,
    strides,
):
    input_tensor = torch.rand(size=src_shape, dtype=torch.float32).npu()
    output_tensor = torch.zeros(sizes, dtype=torch.float32).npu()
    grid = (1, )
    test_subview_kernel_2d[grid](input_tensor, output_tensor, shape_0=src_shape[0], shape_1=src_shape[1],
                                 offsets=tuple(offsets), sizes_0=sizes[0], sizes_1=sizes[1], strides=tuple(strides),
                                 enable_auto_bind_sub_block=False)
