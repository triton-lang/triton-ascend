import torch
import triton
import triton.language as tl


@triton.jit
def kernel(src_ptr, idx_ptr, out_ptr, axis: tl.constexpr, src_dim0: tl.constexpr, src_dim1: tl.constexpr,
           src_stride0: tl.constexpr, src_stride1: tl.constexpr, idx_dim0: tl.constexpr, idx_dim1: tl.constexpr,
           idx_stride0: tl.constexpr, idx_stride1: tl.constexpr, out_dim0: tl.constexpr, out_dim1: tl.constexpr,
           out_stride0: tl.constexpr, out_stride1: tl.constexpr):
    """
    Test tl.gather with a 2D tensor:
    1. Load src tensor from GM
    2. Load index tensor from GM
    3. Gather elements from src along the given axis using the index
    4. Store the gathered result to output
    """
    src_offs = (tl.arange(0, src_dim0)[:, None] * src_stride0 + tl.arange(0, src_dim1)[None, :] * src_stride1)
    src = tl.load(src_ptr + src_offs)

    idx_offs = (tl.arange(0, idx_dim0)[:, None] * idx_stride0 + tl.arange(0, idx_dim1)[None, :] * idx_stride1)
    idx = tl.load(idx_ptr + idx_offs)

    out = tl.gather(src, idx, axis)

    out_offs = (tl.arange(0, out_dim0)[:, None] * out_stride0 + tl.arange(0, out_dim1)[None, :] * out_stride1)
    tl.store(out_ptr + out_offs, out)


def test_gather():
    # src(4,2) = [[1.,2.],[3.,4.],[5.,6.],[7.,8.]]
    src = torch.tensor([[1., 2.], [3., 4.], [5., 6.], [7., 8.]], device='npu')
    # idx(2,2) = [[0,1],[2,0]]
    idx = torch.tensor([[0, 1], [2, 0]], device='npu', dtype=torch.int32)
    out = torch.empty((2, 2), device='npu', dtype=torch.float32)

    kernel[(1, )](src, idx, out, axis=0, src_dim0=4, src_dim1=2, src_stride0=2, src_stride1=1, idx_dim0=2, idx_dim1=2,
                  idx_stride0=2, idx_stride1=1, out_dim0=2, out_dim1=2, out_stride0=2, out_stride1=1)
    # Reference: gather along axis=0 with src(4,2) and idx(2,2)
    # out[i,j] = src[idx[i,j], j]
    # out[0,0] = src[idx[0,0]=0, 0] = 1.0
    # out[0,1] = src[idx[0,1]=1, 1] = 4.0
    # out[1,0] = src[idx[1,0]=2, 0] = 5.0
    # out[1,1] = src[idx[1,1]=0, 1] = 2.0
    expected = torch.tensor([[1., 4.], [5., 2.]], device='cpu')
    assert torch.allclose(out.cpu(), expected), \
        f"Mismatch: expected {expected}, got {out.cpu()}"


if __name__ == "__main__":
    test_gather()
