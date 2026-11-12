# Source: triton_kernel_opt/rmsnorm_situ_base.py
# Source SHA256: a19434dbfaad9eea644c7b8df37c88642c8ad4b8f602f456b50362e78e8bd0d4
# Original kernel bodies; intentionally no wrapper or smoke invocation.
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fwd_fused(input_ptr,  # pointer to the input
                       output_ptr,  # pointer to the output
                       weight_ptr,  # pointer to the weights
                       rrms_ptr,  # pointer to the rrms
                       stride,  # how much to increase the pointer when moving by 1 row
                       d,  # number of columns in input
                       eps,  # epsilon to avoid division by zero
                       D_ROUNDED: tl.constexpr):
    # Map the program id to the row of input and output it should compute.
    row_idx = tl.program_id(0).to(tl.int64)
    output_ptr += row_idx * stride
    input_ptr += row_idx * stride

    # Compute rms
    col_offsets = tl.arange(0, D_ROUNDED)
    mask = col_offsets < d
    x = tl.load(input_ptr + col_offsets, mask=mask, other=0.).to(tl.float32)
    x_square = x * x
    ms = tl.sum(x_square, axis=0) / d + eps
    rrms = 1 / tl.sqrt(ms)

    # Write rms
    tl.store(rrms_ptr + row_idx, rrms)

    # Calc y
    w = tl.load(weight_ptr + col_offsets, mask=mask, other=0.).to(tl.float32)
    x_norm = x * rrms
    y = x_norm * w

    # store data
    tl.store(output_ptr + col_offsets, y, mask=mask)


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _rmsnorm_bwd_dxdw_fused(dy_ptr,  # pointer to the output grads
                            dx_ptr,  # pointer to the input grads
                            dw_ptr,  # pointer to the partial weights grads
                            input_ptr,  # pointer to the input
                            weight_ptr,  # pointer to the weights
                            rrms_ptr,  # pointer to the rrms
                            stride,  # how much to increase the pointer when moving by 1 row
                            s,  # number of rows in input_ptr
                            d,  # number of columns in input_ptr, hidden dimension
                            BLOCK_S: tl.constexpr, D_ROUNDED: tl.constexpr):
    # Map the program id to the row of all the data should compute
    row_block_idx = tl.program_id(0).to(tl.int64)  # group_size_s * d, group_id
    col_offsets = tl.arange(0, D_ROUNDED)  # full d for each row
    col_mask = col_offsets < d
    w = tl.load(weight_ptr + col_offsets, mask=col_mask, other=0.).to(tl.float32)

    partial_dw = tl.zeros((D_ROUNDED, ), dtype=tl.float32)
    start = row_block_idx * BLOCK_S
    end = min(start + BLOCK_S, s)
    for row in range(start, end):
        # Calc the dx and store
        rrms = tl.load(rrms_ptr + row)
        dy = tl.load(dy_ptr + row * stride + col_offsets, mask=col_mask, other=0.).to(tl.float32)
        x = tl.load(input_ptr + row * stride + col_offsets, mask=col_mask, other=0.).to(tl.float32)

        # impl following torch titan
        x_norm = x * rrms
        partial_dw += dy * x_norm

        dxnorm = w * dy
        c1 = tl.sum(x_norm * dxnorm, axis=0) / d
        dx = (dxnorm - x_norm * c1) * rrms
        tl.store(dx_ptr + row * stride + col_offsets, dx, mask=col_mask)

    tl.store(dw_ptr + row_block_idx * d + col_offsets, partial_dw, mask=col_mask)


@triton.jit
def _fused_situgluv2_fwd_triton_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    stride_om,
    stride_on,
    beta,
    inv_beta,
    linear_beta,
    inv_linear_beta,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    nb = tl.program_id(1).to(tl.int64)

    offs_n = nb * BLOCK + tl.arange(0, BLOCK)
    mask = offs_n < N

    a_row = a_ptr + m * stride_am + offs_n * stride_an
    b_row = b_ptr + m * stride_bm + offs_n * stride_bn
    o_row = out_ptr + m * stride_om + offs_n * stride_on

    a = tl.load(a_row, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_row, mask=mask, other=0).to(tl.float32)

    sig_a = tl.sigmoid(a)
    z_a = a * inv_beta
    tanh_a = _tanh(z_a)
    f = beta * tanh_a * sig_a

    z_b = b * inv_linear_beta
    tanh_b = _tanh(z_b)
    r = linear_beta * tanh_b

    out = f * r
    tl.store(o_row, out, mask=mask)


@triton.jit
def _fused_situgluv2_bwd_triton_kernel(
    a_ptr,
    b_ptr,
    dout_ptr,
    da_ptr,
    db_ptr,
    N,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    stride_dom,
    stride_don,
    stride_dam,
    stride_dan,
    stride_dbm,
    stride_dbn,
    beta,
    inv_beta,
    linear_beta,
    inv_linear_beta,
    BLOCK: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64)
    nb = tl.program_id(1).to(tl.int64)
    offs_n = nb * BLOCK + tl.arange(0, BLOCK)
    mask = offs_n < N

    a_row = a_ptr + m * stride_am + offs_n * stride_an
    b_row = b_ptr + m * stride_bm + offs_n * stride_bn
    do_row = dout_ptr + m * stride_dom + offs_n * stride_don
    da_row = da_ptr + m * stride_dam + offs_n * stride_dan
    db_row = db_ptr + m * stride_dbm + offs_n * stride_dbn

    a = tl.load(a_row, mask=mask, other=0).to(tl.float32)
    b = tl.load(b_row, mask=mask, other=0).to(tl.float32)
    dout = tl.load(do_row, mask=mask, other=0).to(tl.float32)

    s = tl.sigmoid(a)
    s_neg = tl.sigmoid(-a)
    z_a = a * inv_beta
    p_a = tl.sigmoid(2.0 * z_a)
    t_a = 2.0 * p_a - 1.0
    tanh_prime_a = 4.0 * p_a * (1.0 - p_a)
    f = beta * t_a * s
    dfdx = beta * t_a * s * s_neg + s * tanh_prime_a

    z_b = b * inv_linear_beta
    p_b = tl.sigmoid(2.0 * z_b)
    t_b = 2.0 * p_b - 1.0
    tanh_prime_b = 4.0 * p_b * (1.0 - p_b)
    r = linear_beta * t_b

    da = dout * r * dfdx
    db = dout * f * tanh_prime_b

    tl.store(da_row, da, mask=mask)
    tl.store(db_row, db, mask=mask)
