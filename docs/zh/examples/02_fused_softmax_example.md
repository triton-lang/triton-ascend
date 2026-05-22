# 融合 Softmax （Fused Softmax）

在本节中，我们将使用 Triton 编写一个融合的 softmax 操作的程序。
在此过程中，你会学习到：

- 内核融合对于带宽受限操作的优势。
- Triton 中缩减操作。

## 使用原生 PyTorch 对 X 逐行进行 Softmax 计算

```Python
import torch
import torch_npu

import triton
import triton.language as tl

def naive_softmax(x):
    """
    我们减去最大元素以避免溢出。Softmax 对于这种偏移是不变的。
    """
    # Read MN elements; write M elements
    x_max = x.max(dim=1)[0]
    # Read MN + M elements; write MN elements
    z = x - x_max[:, None]
    # Read MN elements; write MN elements
    numerator = torch.exp(z)
    # Read MN elements; write M elements
    denominator = numerator.sum(dim=1)
    # Read MN + M elements; write MN elements
    ret = numerator / denominator[:, None]
    # Total: read 5MN + 2M elements; write 3MN + 2M elements
    return ret
```

内核融合的目的

当在 PyTorch 中以原生方式实现时，计算`y=naive_softmax(x)`需要从 DRAM 中读取 5MN+2M 个元素，并写回 3MN+2M 个元素。显然这是非常低效的；我们更希望使用一个自定义的“融合”内核，它只需读取一次 x，并在芯片上完成所有必要的计算。
这样一来只需读取和写回 2MN 个字节，因此我们可以期望理论上的加速比大约为 4 倍（即 (8MN+4M)/2MN）。

`torch.jit.script`旨在自动执行这种“内核融合”，但它仍然远未达到理想状态。

## 计算内核

softmax 内核工作原理如下：每个计算单元（program）以程序数量为步长加载输入矩阵X的一组行数据，执行归一化处理后，将结果写入输出矩阵Y。
注意：Triton 的一个重要限制是每个块必须具有 2 的幂次数的元素，因此，如果我们要处理任意可能的输入形状，需要在内部「填充」每一行，并确保内存操作正确性。

```Python
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    # Program starting row
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # Stride represents how much we need to add to the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # Block size is the next power of 2 greater than n_cols, so we can fit
        # the row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM with a mask, because BLOCK_SIZE may be greater than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract max for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that the exponential operation in Triton is fast but approximate
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write output back to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
```

我们可以创建一个辅助函数，该函数能够将核函数及其元参数加入执行队列，以处理任意给定的输入张量。

```Python
kernels = {}

def softmax(x):
    n_rows, n_cols = x.shape

    # Block size per loop iteration is the smallest power of 2 greater than or equal to the number of columns of `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Allocate output space
    y = torch.empty_like(x)

    # Pre-compile kernel to get register usage and compute thread occupancy
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

## 单元测试

需要在一个具有不规则行和列数的矩阵上测试处理好的内核，此举可以验证Padding机制是否起作用

```Python
torch.manual_seed(0)
x = torch.randn(1823, 781, device='npu')
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
print(y_triton)
print(y_torch)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(y_triton-y_torch))}')
```

Out:

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

"The maximum difference between torch and triton is 1.4901161193847656e-08" 表示Triton和PyTorch的输出结果非常接近，肉眼不可区分。
