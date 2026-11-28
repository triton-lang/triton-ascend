import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension


@triton.jit
def triton_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, SLICE_OFFSET: tl.constexpr,
                  SLICE_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    x_sub = extension.extract_slice(x, [SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = extension.extract_slice(y, [SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    out = tl.load(output_ptr + offsets, mask=mask)
    out = extension.insert_slice(out, output_sub, [SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, out, mask=mask)


def triton_func(x: torch.Tensor, y: torch.Tensor, slice_offset: int, slice_size: int):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    triton_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024, SLICE_OFFSET=slice_offset, SLICE_SIZE=slice_size)
    return out


def test_insert_slice():
    size = 1024
    slice_offset = 0
    slice_size = 32
    x = torch.rand(size, device='npu')
    y = torch.rand(size, device='npu')
    ref = x + y
    out = triton_func(x, y, slice_offset, slice_size)
    torch.testing.assert_close(out[slice_offset:slice_offset + slice_size], ref[slice_offset:slice_offset + slice_size])


if __name__ == "__main__":
    test_insert_slice()
    print("test_insert_slice passed!")
