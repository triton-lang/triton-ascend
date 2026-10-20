import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _compare(x, y):
    if x < y:
        return -1
    elif x == y:
        return 0
    else:
        return 1


@triton.jit
def kernel(X, Y, Z, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.load(Y + tl.arange(0, BLOCK))
    z = tl.map_elementwise(_compare, x, y)
    tl.store(Z + tl.arange(0, BLOCK), z)


def test_map_elementwise():
    shape = (128, )
    x = torch.randint(-100, 100, shape, dtype=torch.int32, device='npu')
    y = torch.randint(-100, 100, shape, dtype=torch.int32, device='npu')
    z = torch.zeros(shape, dtype=torch.int32, device='npu')
    kernel[(1, )](x, y, z, BLOCK=shape[0])
    expected = (x > y).int() - (y > x).int()
    assert torch.equal(z.cpu(), expected.cpu())


if __name__ == "__main__":
    test_map_elementwise()
