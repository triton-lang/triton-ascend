import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N: tl.constexpr,
    C_in: tl.constexpr,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    C_out: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    K_h: tl.constexpr,
    K_w: tl.constexpr,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    padding_h: tl.constexpr,
    padding_w: tl.constexpr,
    groups: tl.constexpr,
):
    # Load input: (N, C_in, H_in, W_in)
    n_offs = tl.arange(0, N)[:, None, None, None]
    c_offs = tl.arange(0, C_in)[None, :, None, None]
    h_offs = tl.arange(0, H_in)[None, None, :, None]
    w_offs = tl.arange(0, W_in)[None, None, None, :]
    input_tile = tl.load(input_ptr + n_offs * (C_in * H_in * W_in) + c_offs * (H_in * W_in) + h_offs * W_in + w_offs)

    # Load weight: (C_out, C_in // groups, K_h, K_w)
    co_offs = tl.arange(0, C_out)[:, None, None, None]
    ci_offs = tl.arange(0, C_in // groups)[None, :, None, None]
    kh_offs = tl.arange(0, K_h)[None, None, :, None]
    kw_offs = tl.arange(0, K_w)[None, None, None, :]
    weight_tile = tl.load(weight_ptr + co_offs * ((C_in // groups) * K_h * K_w) + ci_offs * (K_h * K_w) +
                          kh_offs * K_w + kw_offs)

    # Load bias: (C_out,)
    bias_tile = tl.load(bias_ptr + tl.arange(0, C_out))

    output = al.conv2d(
        input_tile,
        weight_tile,
        bias_tile,
        groups=groups,
        padding=(padding_h, padding_w),
        stride=(stride_h, stride_w),
        dilation=1,
    )

    # Store output: (N, C_out, H_out, W_out)
    no_offs = tl.arange(0, N)[:, None, None, None]
    co_offs = tl.arange(0, C_out)[None, :, None, None]
    ho_offs = tl.arange(0, H_out)[None, None, :, None]
    wo_offs = tl.arange(0, W_out)[None, None, None, :]
    tl.store(output_ptr + no_offs * (C_out * H_out * W_out) + co_offs * (H_out * W_out) + ho_offs * W_out + wo_offs,
             output)


def test_conv2d():
    N, C_in, H_in, W_in = 2, 16, 32, 32
    C_out, K_h, K_w = 32, 3, 3
    stride = (1, 1)
    padding = (1, 1)
    groups = 1
    H_out = (H_in + 2 * padding[0] - (K_h - 1) - 1) // stride[0] + 1
    W_out = (W_in + 2 * padding[1] - (K_w - 1) - 1) // stride[1] + 1

    x = torch.randn(N, C_in, H_in, W_in, dtype=torch.float16)
    w = torch.randn(C_out, C_in // groups, K_h, K_w, dtype=torch.float16)
    b = torch.randn(C_out, dtype=torch.float16)

    x_npu = x.npu()
    w_npu = w.npu()
    b_npu = b.npu()
    out_npu = torch.empty(N, C_out, H_out, W_out, dtype=torch.float16).npu()

    conv2d_kernel[(1, )](x_npu, w_npu, b_npu, out_npu, N=N, C_in=C_in, H_in=H_in, W_in=W_in, C_out=C_out, H_out=H_out,
                         W_out=W_out, K_h=K_h, K_w=K_w, stride_h=stride[0], stride_w=stride[1], padding_h=padding[0],
                         padding_w=padding[1], groups=groups)

    gold = torch.nn.functional.conv2d(x, w, b, stride=stride, padding=padding, groups=groups)
    torch.testing.assert_close(out_npu.cpu(), gold, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    test_conv2d()
