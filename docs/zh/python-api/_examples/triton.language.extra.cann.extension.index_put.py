import torch
import triton
import triton.language as tl
from triton.language.extra.cann.extension import index_put
from triton.tools.get_ascend_devices import is_compile_on_910_95


@triton.jit
def kernel(value_ptr, index_ptr, dst_ptr):
    """
    Test index_put with a 2D value and 1D index:
    1. Load index tensor from GM (shape [2])
    2. Load value tensor from GM (shape [2, 2])
    3. Scatter value rows into dst at positions given by index along dim=0
    4. The operation: dst[index[i], :] = value[i, :]
    """
    # index tile shape: [2]
    index_local = tl.arange(0, 2)
    x1_local = tl.arange(0, 2)[None, :]  # shape=(1,2)

    index_tile = tl.load(index_ptr + index_local)
    value_tile = tl.load(value_ptr + index_local[:, None] * 2 + x1_local)

    index_put(ptr=dst_ptr, index=index_tile, value=value_tile, dim=0, index_boundary=4, end_offset=(2, 2),
              start_offset=(0, 0), dst_stride=(2, 1))


def test_index_put():
    # dst: (4,2) of zeros
    dst = torch.zeros((4, 2), device='npu', dtype=torch.float32)
    # value = [[1., 2.],
    #          [3., 4.]]
    value = torch.tensor([[1., 2.], [3., 4.]], device='npu')
    # index = [2, 0]
    index = torch.tensor([2, 0], device='npu')

    kernel[(1, )](value, index, dst)

    # Reference: dst[index[i], :] = value[i, :]
    # dst[2, :] = value[0, :] = [1., 2.]
    # dst[0, :] = value[1, :] = [3., 4.]
    # expected = [[3., 4.],
    #             [0., 0.],
    #             [1., 2.],
    #             [0., 0.]]
    expected = torch.tensor([[3., 4.], [0., 0.], [1., 2.], [0., 0.]], device='cpu')
    assert torch.allclose(dst.cpu(), expected), \
        f"Mismatch: expected {expected}, got {dst.cpu()}"


if __name__ == "__main__":
    if not is_compile_on_910_95:
        print("index_put is only supported on Ascend 950, skipping test.")
    else:
        test_index_put()
