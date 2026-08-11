import torch
import torch_npu  # noqa: F401

import triton
import triton.language as tl


@triton.jit
def _inttoptr_static_offset_kernel(
    block_table_ptrs,
    out,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)

    # Load a raw i64 address and cast to pointer — creates tt.int_to_ptr.
    block_table_ptr = tl.load(block_table_ptrs).to(tl.pointer_type(tl.int32))

    # Scalar addptr with a constant offset — creates a ReinterpretCastOp
    # whose result type has a non-zero static offset in the strided layout.
    block_table_ptr = block_table_ptr + 1

    block_numbers = tl.load(block_table_ptr + offsets)
    slot_ids = block_numbers.to(tl.int64)
    tl.store(out + offsets, slot_ids)


@triton.jit
def _inttoptr_dyn_offset_kernel(
    block_table_ptrs,
    dyn_offset,
    out,
    BLOCK_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)

    block_table_ptr = tl.load(block_table_ptrs).to(tl.pointer_type(tl.int32))

    # Scalar addptr with a runtime offset — creates a ReinterpretCastOp
    # whose result type has a dynamic offset (?) in the strided layout.
    block_table_ptr = block_table_ptr + tl.load(dyn_offset)

    block_numbers = tl.load(block_table_ptr + offsets)
    slot_ids = block_numbers.to(tl.int64)
    tl.store(out + offsets, slot_ids)


def test_inttoptr_static_offset_reset():
    """Static offset (constant 1) baked into base address."""
    block = 8

    block_table_cpu = torch.arange(16, dtype=torch.int32)
    block_table = block_table_cpu.npu()
    out = torch.empty((block, ), dtype=torch.int64, device="npu")

    block_table_ptrs = torch.tensor([block_table.data_ptr()], dtype=torch.int64).npu()

    _inttoptr_static_offset_kernel[(1, )](
        block_table_ptrs,
        out,
        BLOCK_SIZE=4,
        BLOCK=block,
    )
    torch.npu.synchronize()

    # offset 1 => skip block_table[0]
    expected = block_table_cpu[1:1 + block].to(torch.int64)
    torch.testing.assert_close(out.cpu(), expected.cpu())


def test_inttoptr_dyn_offset_reset():
    """Dynamic offset (runtime value) baked into base address."""
    block = 8

    block_table_cpu = torch.arange(16, dtype=torch.int32)
    block_table = block_table_cpu.npu()
    out = torch.empty((block, ), dtype=torch.int64, device="npu")

    block_table_ptrs = torch.tensor([block_table.data_ptr()], dtype=torch.int64).npu()
    dyn_offset = torch.tensor([2], dtype=torch.int32).npu()

    _inttoptr_dyn_offset_kernel[(1, )](
        block_table_ptrs,
        dyn_offset,
        out,
        BLOCK_SIZE=4,
        BLOCK=block,
    )
    torch.npu.synchronize()

    # dynamic offset 2 => skip block_table[0:2]
    expected = block_table_cpu[2:2 + block].to(torch.int64)
    torch.testing.assert_close(out.cpu(), expected.cpu())
