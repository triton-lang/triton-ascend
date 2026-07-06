import torch
import triton
import triton.language as tl


@triton.jit
def matrix_add_bias_kernel(x_ptr, bias_ptr, output_ptr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr,
                           BLOCK_N: tl.constexpr):
    # broadcast bias to (BLOCK_M, BLOCK_N)
    offsets_m = tl.arange(0, BLOCK_M)[:, None]
    offsets_n = tl.arange(0, BLOCK_N)[None, :]
    x = tl.load(x_ptr + offsets_m * N + offsets_n)
    bias = tl.load(bias_ptr + offsets_n)
    bias_broadcast = bias.broadcast_to([BLOCK_M, BLOCK_N])
    output = x + bias_broadcast
    tl.store(output_ptr + offsets_m * N + offsets_n, output)


def test_broadcast_to():
    M, N = 8, 16
    BLOCK_M, BLOCK_N = M, N

    torch.manual_seed(0)
    x = torch.randn((M, N), dtype=torch.float32).npu()
    bias = torch.randn((1, N), dtype=torch.float32).npu()

    output = torch.empty((M, N), dtype=torch.float32, device="npu")

    matrix_add_bias_kernel[(1, )](x, bias, output, M=M, N=N, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

    ref = x.cpu() + bias.cpu()

    torch.testing.assert_close(output.cpu(), ref)
    print("Test passed.")


if __name__ == "__main__":
    test_broadcast_to()
