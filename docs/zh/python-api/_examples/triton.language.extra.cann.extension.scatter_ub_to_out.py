import torch
import triton
import triton.language as tl
from triton.language.extra.cann.extension import scatter_ub_to_out
from triton.tools.get_ascend_devices import is_compile_on_910_95


@triton.jit
def kernel(value_ptr, index_ptr, dst_ptr):
    """
    Test scatter_ub_to_out with a 2D value tile:
    1. Load a [2,2] value tile from GM into UB
    2. Load a [2,2] index tile from GM into UB
    3. Scatter value elements from UB into dst (in GM) at positions given by index along dim=0
    4. The operation: dst[index[i,j], j] = value[i,j]
    """
    # index tile shape: [2,2]
    y0_local = tl.arange(0, 2)[:, None]  # [0,1] rows
    x1_local = tl.arange(0, 2)[None, :]  # [0,1] cols
    mask = (y0_local < 2) & (x1_local < 2)

    value = tl.load(value_ptr + y0_local * 2 + x1_local, mask)
    index = tl.load(index_ptr + y0_local * 2 + x1_local, mask)

    scatter_ub_to_out(ptr=dst_ptr, value=value, index=index, index_boundary=4, dim=0, dst_stride=(2, 1),
                      end_offset=(2, 2), start_offset=(0, 0))


def test_scatter_ub_to_out():
    # dst: (4,2) of zeros
    dst = torch.zeros((4, 2), device='npu', dtype=torch.float32)
    # value(2,2) = [[1.,2.],[3.,4.]]
    value = torch.tensor([[1., 2.], [3., 4.]], device='npu')
    # index(2,2) = [[1,2],[3,0]]
    index = torch.tensor([[1, 2], [3, 0]], device='npu')

    kernel[(1, )](value, index, dst)

    # Reference: scatter along dim=0 with dst(4,2) and value/index(2,2)
    # dst[index[i,j], j] = value[i,j]
    # dst[index[0,0]=1, 0] = value[0,0] = 1.0
    # dst[index[0,1]=2, 1] = value[0,1] = 2.0
    # dst[index[1,0]=3, 0] = value[1,0] = 3.0
    # dst[index[1,1]=0, 1] = value[1,1] = 4.0
    # expected = [[0., 4.],
    #             [1., 0.],
    #             [0., 2.],
    #             [3., 0.]]
    expected = torch.tensor([[0., 4.], [1., 0.], [0., 2.], [3., 0.]], device='cpu')
    assert torch.allclose(dst.cpu(), expected), \
        f"Mismatch: expected {expected}, got {dst.cpu()}"


if __name__ == "__main__":
    if not is_compile_on_910_95:
        print("scatter_ub_to_out is only supported on Ascend 950, skipping test.")
    else:
        test_scatter_ub_to_out()
