# 向量相加（Vector Addition）

本文档介绍使用Triton实现向量相加的完整流程，涵盖计算内核编写、辅助函数封装与结果验证，适用于需要了解Triton基本编程模式的开发者。
在此过程中，你会学习到：

- Triton的基本编程模式。
- 用于定义Triton内核的`triton.jit`装饰器（decorator）。

## 定义计算内核

```python
import torch
import torch_npu

import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr,  # 指向第一个输入向量的指针。
               y_ptr,  # 指向第二个输入向量的指针。
               output_ptr,  # 指向输出向量的指针。
               n_elements,  # 向量的大小。
               BLOCK_SIZE: tl.constexpr,  # 每个程序应处理的元素数量,取值需为2的整数次幂。
               # 注意：`constexpr` 将标记变量为常量。
               ):
    # 不同的数据由不同的程序来处理，因此需要分配：
    pid = tl.program_id(axis=0)  # 使用1D启动网格，因此轴为0。
    # 该程序将处理相对初始数据偏移的输入。
    # 例如，如果有一个长度为256、块大小为64的向量，程序将各自访问 [0:64, 64:128, 128:192, 192:256] 的元素。
    # 注意offsets是元素偏移量列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建掩码以防止内存操作超出边界访问。
    mask = offsets < n_elements
    # 从DRAM加载x和y，如果输入不是块大小的整数倍，则屏蔽掉任何多余的元素。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 将x + y写回DRAM。
    tl.store(output_ptr + offsets, output, mask=mask)
```

创建一个辅助函数用于：

- 生成z张量；
- 用适当的grid/block sizes将上述内核加入队列。

```python
def add(x: torch.Tensor, y: torch.Tensor):
    # 需要预分配输出。
    z = torch.empty_like(x)
    n_elements = z.numel()
    # 启动网格表示并行运行的内核实例的数量。
    # 可以是Tuple[int]，也可以是Callable(metaparameters) -> Tuple[int]。
    # 在本例中，使用1D网格，其中大小是块的数量：
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - 每个torch.tensor对象都会隐式转换为其第一个元素的指针。
    #  - `triton.jit` 函数可以通过启动网格索引来获得可调用的NPU内核。
    #  - 必须以关键字参数的形式传递元参数。
    add_kernel[grid](x, y, z, n_elements, BLOCK_SIZE=1024)
    # 返回z的句柄。
    return z
```

## 运行与验证

使用上述函数计算两个 `torch.tensor` 对象的element-wise sum，并测试其正确性：

```python
torch.manual_seed(0)
size = 98432  # 向量长度，特意取非块大小整数倍的值以验证掩码分支的处理
x = torch.rand(size, device='npu')
y = torch.rand(size, device='npu')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

## 输出结果

```output
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
tensor([0.8329, 1.0024, 1.3639,  ..., 1.0796, 1.0406, 1.5811], device='npu:0')
The maximum difference between torch and triton is 0.0
```

"The maximum difference between torch and triton is 0.0" 表示Triton和PyTorch的输出结果一致。
