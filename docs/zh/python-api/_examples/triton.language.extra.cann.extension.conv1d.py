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
    padding: tl.constexpr,
):
    n_offsets = tl.arange(0, N)[:, None, None]
    ci_offsets = tl.arange(0, C_in)[None, :, None]
    input_offsets = n_offsets * (C_in * L_in) + ci_offsets * L_in + tl.arange(0, L_in)[None, None, :]
    input_tile = tl.load(input_ptr + input_offsets)

    co_offsets = tl.arange(0, C_out)[:, None, None]
    weight_offsets = co_offsets * (C_in * K) + ci_offsets * K + tl.arange(0, K)[None, None, :]
    weight_tile = tl.load(weight_ptr + weight_offsets)
    bias_tile = tl.load(bias_ptr + tl.arange(0, C_out))

    output = al.conv1d(input_tile, weight_tile, bias_tile, stride=1, padding=padding, dilation=1, groups=1)

    output_co_offsets = tl.arange(0, C_out)[None, :, None]
    output_offsets = (n_offsets * (C_out * L_out) + output_co_offsets * L_out + tl.arange(0, L_out)[None, None, :])
    tl.store(output_ptr + output_offsets, output)


def test_conv1d_asymmetric_padding():
    N, C_in, L_in = 2, 16, 32
    C_out, K = 32, 4
    pad_left, pad_right = 1, 2
    L_out = L_in + pad_left + pad_right - K + 1

    x = torch.randn(N, C_in, L_in, dtype=torch.float16)
    weight = torch.randn(C_out, C_in, K, dtype=torch.float16)
    bias = torch.randn(C_out, dtype=torch.float16)
    output = torch.empty(N, C_out, L_out, dtype=torch.float16).npu()

    conv1d_kernel[(1, )](x.npu(), weight.npu(), bias.npu(), output, N=N, C_in=C_in, L_in=L_in, C_out=C_out, L_out=L_out,
                         K=K, padding=(pad_left, pad_right))

    padded_x = torch.nn.functional.pad(x, (pad_left, pad_right))
    expected = torch.nn.functional.conv1d(padded_x, weight, bias)
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_conv1d_asymmetric_padding()
