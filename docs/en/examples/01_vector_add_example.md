# Vector Addition

In this section, you will use Triton to write a simple vector addition program.
In this process, you will learn:

- The basic programming model of Triton.
- The `triton.jit` decorator used to define Triton kernels.

Compute kernel:

```bash
import torch
import torch_npu

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,  # Pointer to the first input vector.
               y_ptr,  # Pointer to the second input vector.
               output_ptr,  # Pointer to the output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements that should be processed by each program.
               # Note: `constexpr` will mark the variable as a constant.
               ):
    # Different data is processed by different "processes", so you need to allocate:
    pid = tl.program_id(axis=0) # A 1D launch grid is used, so the axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For example, if there is a vector of length 256 and block size 64, the program will access the elements  [0:64, 64:128, 128:192, 192:256] respectively.
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent memory operations from out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, and mask out any extra elements if the input is not an integer multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

Create a helper function to:

- Generate the z tensor;
- Enqueue the above kernel with the appropriate grid/block sizes.

```Python
def add(x: torch.Tensor, y: torch.Tensor):
    # The output needs to be pre-allocated.
    output = torch.empty_like(x)
    n_elements = output.numel()
    # The launch grid indicates the number of kernel instances that run in parallel.
    # It can be Tuple[int] or Callable(metaparameters) -> Tuple[int].
    # In this case, a 1D grid is used, where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - The `triton.jit` function can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Pass meta-parameters as keywords.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # Returns the handle to z.
    return output
```

Use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

```Python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

Output:

```bash
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

"The maximum difference between torch and triton is 0.0" indicates that the output results of Triton and PyTorch are the same.
