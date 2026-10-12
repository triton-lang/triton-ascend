import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def conv1d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    C_in: tl.constexpr,
    L_in: tl.constexpr,
    C_out: tl.constexpr,
    L_out: tl.constexpr,
    K: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    groups: tl.constexpr,
):
    # Load input: (N, C_in, L_in)
    n_offs = tl.arange(0, N)[:, None, None]
    c_offs = tl.arange(0, C_in)[None, :, None]
    l_offs = tl.arange(0, L_in)[None, None, :]
    input_tile = tl.load(input_ptr + n_offs * (C_in * L_in) + c_offs * L_in + l_offs)

    # Load weight: (C_out, C_in // groups, K)
    co_offs = tl.arange(0, C_out)[:, None, None]
    ci_offs = tl.arange(0, C_in // groups)[None, :, None]
    k_offs = tl.arange(0, K)[None, None, :]
    weight_tile = tl.load(weight_ptr + co_offs * ((C_in // groups) * K) + ci_offs * K + k_offs)

    # Load bias: (C_out,)
    bias_tile = tl.load(bias_ptr + tl.arange(0, C_out))

    output = al.conv1d(
        input_tile,
        weight_tile,
        bias_tile,
        groups=groups,
        padding=padding,
        stride=stride,
        dilation=1,
    )

    # Store output: (N, C_out, L_out)
    no_offs = tl.arange(0, N)[:, None, None]
    co_offs = tl.arange(0, C_out)[None, :, None]
    lo_offs = tl.arange(0, L_out)[None, None, :]
    tl.store(output_ptr + no_offs * (C_out * L_out) + co_offs * L_out + lo_offs, output)


def test_conv1d():
    N, C_in, L_in = 2, 16, 32
    C_out, K = 32, 3
    stride, padding, groups = 1, 1, 1
    L_out = (L_in + 2 * padding - (K - 1) - 1) // stride + 1

    x = torch.randn(N, C_in, L_in, dtype=torch.float16)
    w = torch.randn(C_out, C_in // groups, K, dtype=torch.float16)
    b = torch.randn(C_out, dtype=torch.float16)

    x_npu = x.npu()
    w_npu = w.npu()
    b_npu = b.npu()
    out_npu = torch.empty(N, C_out, L_out, dtype=torch.float16).npu()

    conv1d_kernel[(1, )](x_npu, w_npu, b_npu, out_npu, N=N, C_in=C_in, L_in=L_in, C_out=C_out, L_out=L_out, K=K,
                         stride=stride, padding=padding, groups=groups)

    gold = torch.nn.functional.conv1d(x, w, b, stride=stride, padding=padding, groups=groups)
    torch.testing.assert_close(out_npu.cpu(), gold, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_conv1d()
