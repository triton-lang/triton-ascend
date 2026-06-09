# Fused Softmax

In this section, you will use Triton to write a program of the fused softmax operation.
In this process, you will learn:

- The advantages of kernel fusion for bandwidth-bound operations.
- Reduction operations in Triton.

## Using Native PyTorch to Perform Softmax Operation on X Row by Row

```Python
import torch
import torch_npu

import triton
import triton.language as tl

def naive_softmax(x):
    """
    Subtract the maximum element to avoid overflow. Softmax is invariant to this offset.
    """
    # Read MN elements; write M elements.
    x_max = x.max(dim=1)[0]
    # Read MN + M elements; write MN elements.
    z = x - x_max[:, None]
    # Read MN elements; write MN elements.
    numerator = torch.exp(z)
    # Read MN elements; write M elements.
    denominator = numerator.sum(dim=1)
    # Read MN + M elements; write MN elements.
    ret = numerator / denominator[:, None]
    # Total: Read 5 × MN + 2 × M elements; write 3 × MN + 2 × M elements.
    return ret
```

Purpose of kernel fusion

When implemented naively in PyTorch, computing `y = naive_softmax(x)` requires reading 5 × *MN* + 2 × *M* elements from DRAM and writing back 3 *MN* + 2 *M* elements. Obviously, this is very inefficient. A more efficient solution is to use a custom "fused" kernel that reads `x` only once and completes all necessary computations on the chip.
Doing so requires reading and writing back only 2 × *MN* bytes. Therefore, the theoretical speedup ratio is about 4 times, that is, 8 × *MN* + 4 × *M*)/2 × *MN*.

`torch.jit.script` is designed to automatically perform this kind of "kernel fusion", but it is still far from ideal.

## Compute Kernel

The softmax kernel works as follows: Each compute unit (program) loads a group of data rows of the input matrix **X** stridden by number of programs, normalizes it, and writes back the result to the output matrix **Y**.
Note: A significant limitation of Triton is that each block must have a power-of-two number of elements. Therefore, to handle any possible input shapes, internally "pad" each row and ensure the correctness of memory operations.

```Python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # Program start row
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # The stride indicates the required increment of the pointer to advance one row.
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit
        # rows in a single block.
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM using a mask, because BLOCK_SIZE may be greater than n_cols.
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract the maximum value for numerical stability.
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate.
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write the output back to DRAM.
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

Create a helper function. This function can add the kernel function and its meta-parameters to the execution queue to process any given input tensor.

```Python
target = triton.runtime.driver.active.get_current_target()
kernels = {}

def softmax(x, stream):
    n_rows, n_cols = x.shape

    # The block size for each loop iteration is the smallest power of two greater than or equal to the number of columns in `x`.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Allocate output space.
    y = torch.empty_like(x)

    # Precompile the kernel to obtain the register usage and compute the thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        num_programs = 32
        kernel = softmax_kernel
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE
    )
    return y
```

## Unit Test

The processed kernel needs to be tested on a matrix with irregular numbers of rows and columns. This can verify that the padding mechanism works.

```Python
device = torch.npu.current_device()
stream = torch.npu.current_stream(device).npu_stream
torch.manual_seed(0)
x = torch.randn(1823, 781, device='npu')
y_triton = softmax(x, stream)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
print(y_triton)
print(y_torch)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(y_triton-y_torch))}')
```

Output:

```bash
tensor([[0.0002, 0.0017, 0.0009,  ..., 0.0009, 0.0013, 0.0073],
        [0.0001, 0.0004, 0.0006,  ..., 0.0006, 0.0004, 0.0003],
        [0.0007, 0.0002, 0.0006,  ..., 0.0011, 0.0004, 0.0039],
        ...,
        [0.0021, 0.0002, 0.0015,  ..., 0.0012, 0.0014, 0.0022],
        [0.0003, 0.0002, 0.0007,  ..., 0.0005, 0.0006, 0.0007],
        [0.0034, 0.0014, 0.0005,  ..., 0.0007, 0.0016, 0.0028]],
       device='npu:0')
tensor([[0.0002, 0.0017, 0.0009,  ..., 0.0009, 0.0013, 0.0073],
        [0.0001, 0.0004, 0.0006,  ..., 0.0006, 0.0004, 0.0003],
        [0.0007, 0.0002, 0.0006,  ..., 0.0011, 0.0004, 0.0039],
        ...,
        [0.0021, 0.0002, 0.0015,  ..., 0.0012, 0.0014, 0.0022],
        [0.0003, 0.0002, 0.0007,  ..., 0.0005, 0.0006, 0.0007],
        [0.0034, 0.0014, 0.0005,  ..., 0.0007, 0.0016, 0.0028]],
       device='npu:0')
The maximum difference between torch and triton is 1.4901161193847656e-08
```

"The maximum difference between torch and triton is 1.4901161193847656e-08" indicates that the output results of Triton and PyTorch are very close and cannot be visually distinguished.
