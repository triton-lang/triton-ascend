"""LSB-first packed masks for tiled Triton kernels on Ascend."""

import torch
import triton
import triton.language as tl


def generate_bitmask(rows, cols, row_capacity, column_capacity, *, mask):
    """Pack a (rows, cols) NPU bool mask into a padded uint8 tensor.

    The capacities cover complete consumer tiles. The column capacity
    must be a multiple of 8, and the packed tensor stays on mask.device.
    """
    device = mask.device

    # Zero-fill the full allocation so padded rows and columns encode as 0.
    dense = torch.zeros((row_capacity, column_capacity), dtype=torch.bool, device=device)

    # Pass your own NPU bool mask as mask. Copy it into the valid region.
    dense[:rows, :cols] = mask

    # Pack 8 consecutive mask columns into one byte. The first column goes
    # into bit 0 (weight 1), and the eighth goes into bit 7 (weight 128).
    bits_per_byte = 8
    row_bytes = column_capacity // bits_per_byte
    groups = dense.view(row_capacity, row_bytes, bits_per_byte)
    bit_weights = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                               dtype=torch.int32, device=device)

    # Weight each 0/1 value by its bit position, then sum each group.
    group_bits = groups.to(torch.int32)
    weighted_bits = group_bits * bit_weights
    byte_values = weighted_bits.sum(dim=-1, dtype=torch.int32)
    packed = byte_values.to(torch.uint8)
    return packed


@triton.jit
def unpack_mask(Packed, row_start, col_start, ROW_BYTES: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Decode one tile. col_start is byte-aligned and the full tile is backed by GM."""
    row = tl.arange(0, BLOCK_M)[:, None]
    col = tl.arange(0, BLOCK_N)[None, :]

    # Find the packed byte and the bit position for each logical column.
    start_byte = col_start // 8
    byte_col = start_byte + col // 8
    bit = (col % 8).to(tl.uint8)

    # Eight adjacent logical columns share one address, as in FA.
    row_offset = (row_start + row) * ROW_BYTES
    ptrs = Packed + row_offset + byte_col
    packed = tl.load(ptrs)

    # Shift the selected bit into the lowest position, then keep only that bit.
    shifted = packed >> bit
    bits = shifted & 1
    decoded = bits.to(tl.uint8)
    return decoded
