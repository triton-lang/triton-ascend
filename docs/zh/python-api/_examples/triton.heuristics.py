import triton
import triton.language as tl
import torch


@triton.heuristics({'BLOCK_SIZE': lambda args: triton.next_power_of_2(args['n_elements'] // 4)})
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def test_heuristics():
    N = 128
    x = torch.randn(N, dtype=torch.float32, device='npu')
    y = torch.randn(N, dtype=torch.float32, device='npu')
    out = torch.empty(N, dtype=torch.float32, device='npu')
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, out, N)
    torch.testing.assert_close(out.cpu(), (x + y).cpu())


if __name__ == "__main__":
    test_heuristics()
