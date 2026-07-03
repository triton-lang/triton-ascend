import torch
import triton
import triton.language as tl


@triton.jit
def kernel_minimal(
    idx_ptr, table_ptr, out_ptr,
    y0_numel, x1_numel,
    Y0BLOCK: tl.constexpr, Y0BLOCK_SUB: tl.constexpr, X1BLOCK_SUB: tl.constexpr,
):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0 = tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    base_x1 = tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (x1_numel + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:, None]
        y0_mask = y0 < min(Y0BLOCK + y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = (loop_x1 * X1BLOCK_SUB) + base_x1[None, :]
            x1_mask = x1 < x1_numel

            m_lo = (y0 >= 64) & (y0 < 128) & y0_mask
            idx_lo = tl.load(idx_ptr + ((-64) + y0), m_lo, other=0)

            m_hi = (y0 >= 128) & y0_mask
            idx_hi = tl.load(idx_ptr + ((-128) + y0), m_hi, other=0)

            val_lo = tl.load(table_ptr + (x1 + 2048 * idx_lo), x1_mask, other=0.0)
            val_hi = tl.load(table_ptr + (x1 + 2048 * idx_hi), x1_mask, other=0.0)

            result = tl.where(m_hi, val_hi, tl.where(m_lo, val_lo, 0.0))
            tl.store(out_ptr + (x1 + 2048 * y0), result, x1_mask & y0_mask)


def _reference(idx, table):
    ref = torch.zeros((192, 2048), device=table.device, dtype=table.dtype)
    flat = idx.flatten()
    ref[64:128] = table[flat[0:64]]
    ref[128:192] = table[flat[0:64]]
    return ref.unsqueeze(0)


def test():
    device = 'npu'
    idx = torch.randint(0, 95, (1, 64), device=device, dtype=torch.int64)
    table = torch.randn((96, 2048), device=device, dtype=torch.float32)
    out = torch.empty((1, 192, 2048), device=device, dtype=torch.float32)
    kernel_minimal[(36,)](
        idx, table, out,
        192, 2048, 6, 4, 2048,
        multibuffer=False,
    )
    torch.npu.synchronize()
    torch.testing.assert_close(out, _reference(idx, table))


if __name__ == "__main__":
    test()
