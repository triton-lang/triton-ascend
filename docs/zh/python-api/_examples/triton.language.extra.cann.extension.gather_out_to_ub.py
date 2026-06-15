import torch
import triton
import triton.language as tl
from triton.language.extra.cann.extension import gather_out_to_ub
from triton.tools.get_ascend_devices import is_compile_on_910_95


@triton.jit
def kernel(src_ptr, index_ptr, out_ptr):
    """
    Test gather_out_to_ub with a 2D index tile:
    1. Load a [2,2] index tile from GM into UB
    2. Gather values from src (in GM) along dim=0 using the index tile
    3. Store the gathered result (in UB) to output
    """
    # index tile shape: [2,2]
    y0_local = tl.arange(0, 2)[:, None]  # [0,1] rows
    x1_local = tl.arange(0, 2)[None, :]  # [0,1] cols
    mask = (y0_local < 2) & (x1_local < 2)

    # Load index tile to UB
    index = tl.load(index_ptr + y0_local * 2 + x1_local, mask)

    # Call gather_out_to_ub: gather values from src along dim=0
    gathered = gather_out_to_ub(src=src_ptr, index=index, index_boundary=4, dim=0, src_stride=(2, 1), end_offset=(2, 2),
                                start_offset=(0, 0))

    tl.store(out_ptr + y0_local * 2 + x1_local, gathered, mask)


def test_gather_out_to_ub():
    # src(4,2) = [[1.,2.],[3.,4.],[5.,6.],[7.,8.]]
    src = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], device='npu')
    # index(2,2) = [[0,1],[2,3]]
    index = torch.tensor([[0, 1], [2, 3]], device='npu')
    out = torch.empty((2, 2), device='npu', dtype=torch.float32)

    kernel[(1, )](src, index, out)

    # Reference: gather along dim=0 with src(4,2) and index(2,2)
    # out[i,j] = src[index[i,j], j]
    # out[0,0] = src[index[0,0]=0, 0] = 1.0
    # out[0,1] = src[index[0,1]=1, 1] = 4.0
    # out[1,0] = src[index[1,0]=2, 0] = 5.0
    # out[1,1] = src[index[1,1]=3, 1] = 8.0
    expected = torch.tensor([[1., 4.], [5., 8.]], device='cpu')
    assert torch.allclose(out.cpu(), expected), \
        f"Mismatch: expected {expected}, got {out.cpu()}"


if __name__ == "__main__":
    if not is_compile_on_910_95:
        print("gather_out_to_ub is only supported on Ascend 950, skipping test.")
    else:
        test_gather_out_to_ub()
