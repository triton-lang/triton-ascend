# 向量相加 （Vector Addition）

在本节中，我们将使用 Triton 编写一个简单的向量相加的程序。
在此过程中，你会学习到：

- Triton 的基本编程模式。
- 用于定义Triton内核的`triton.jit`装饰器（decorator）。

计算内核:

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
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # Note: `constexpr` marks the variable as a constant.
               ):
    # Different data is handled by different "processes", so we need to allocate:
    pid = tl.program_id(axis=0)  # Use a 1D launch grid, so axis is 0.
    # This program will handle inputs relative to the initial data offset.
    # For example, if there is a vector of length 256 with a block size of 64, the programs will each access elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent memory operations from accessing out of bounds.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements if the input is not a multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

创建一个辅助函数用于：

- 生成 z 张量；
- 用适当的 grid/block sizes 将上述内核加入队列。

```Python
def add(x: torch.Tensor, y: torch.Tensor):
    # The output needs to be pre-allocated.
    output = torch.empty_like(x)
    n_elements = output.numel()
    # The launch grid represents the number of kernel instances running in parallel.
    # It can be a Tuple[int], or a Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid whose size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted to a pointer to its first element.
    #  - `triton.jit` functions can be invoked through the launch grid index to obtain a callable NPU kernel.
    #  - Do not forget to pass meta-parameters as keywords.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # Return the handle of z.
    return output
```

使用上述函数计算两个 `torch.tensor` 对象的 element-wise sum，并测试其正确性：

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

Out:

```bash
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

"The maximum difference between torch and triton is 0.0" 表示Triton和PyTorch的输出结果一致。
