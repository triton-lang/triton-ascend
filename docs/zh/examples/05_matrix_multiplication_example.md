# 矩阵乘法 （Matrix Multiplication）

在本节中，我们展示了使用 Triton 进行矩阵乘法的内核实现。

## 计算内核

以下 Triton 内核实现了一个带偏置项的批量矩阵乘法（Batched Matrix Multiplication with Bias）：
计算公式为：

$$ \mathrm{output}[b, i, j] = \sum_{k} x[b, i, k] \cdot y[k, j] + z[b, i, j] $$

其中：

- `x` 的形状为 `(A, B)`
- `y` 的形状为 `(B, C)`
- `z`（偏置）的形状为 `(A, C)`
- 输出 `output` 的形状为 `(A, C)`

该内核假设单个 block 负责整个输出矩阵的计算，适用于小规模矩阵（A、B、C 较小且能被当前程序块完全覆盖）。

```python
import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def triton_dot_2_Bias(
    output_ptr,   # Output tensor pointer, shape (A, C)
    x_ptr,        # Input tensor x pointer, shape (A, B)
    y_ptr,        # Input tensor y pointer, shape (B, C)
    z_ptr,        # Bias tensor z pointer, shape (A, C)
    A: tl.constexpr,  # Size of the first dimension (batch / number of rows)
    B: tl.constexpr,  # Shared dimension (number of columns in x, number of rows in y)
    C: tl.constexpr   # Size of the second dimension (number of columns)
):
    # Create index vectors
    bidx = tl.arange(0, A)  # [0, 1, ..., A-1], used for the row dimension
    cidx = tl.arange(0, B)  # [0, 1, ..., B-1], used for x's columns / y's rows
    didx = tl.arange(0, C)  # [0, 1, ..., C-1], used for the column dimension

    # Construct linear indices for x: (A, B) -> flattened to A*B
    Xidx = bidx[:, None] * B + cidx[None, :]  # Broadcast to form an (A, B) index grid

    # Construct linear indices for y: (B, C) -> flattened to B*C
    Yidx = cidx[:, None] * C + didx[None, :]  # (B, C) index grid

    # Construct linear indices for z and output: (A, C)
    Zidx = bidx[:, None] * C + didx[None, :]  # (A, C) index grid

    # Load data from global memory
    X = tl.load(x_ptr + Xidx)  # Load the (A, B) sub-block
    Y = tl.load(y_ptr + Yidx)  # Load the (B, C) sub-block
    Z = tl.load(z_ptr + Zidx)  # Load the bias (A, C)

    # Perform matrix multiplication and add the bias
    ret = tl.dot(X, Y) + Z  # tl.dot computes (A, B) × (B, C) → (A, C)

    # Write the result back to global memory
    oidx = bidx[:, None] * C + didx[None, :]  # Same as Zidx, can be reused
    tl.store(output_ptr + oidx, ret)
```

## 工具方法

以下辅助函数用于支持 Triton 内核的测试与验证，包括 PyTorch 参考实现、数据类型映射、随机张量生成及结果校验。

```Python
def torch_dot_Bias(x0, x1, bias):
    """PyTorch reference implementation: performs matrix multiplication and adds the bias term."""
    res = torch.matmul(x0, x1) + bias
    return res

def get_torch_typename(dtype):
    """Maps a string-form data type to the corresponding torch.dtype."""
    if dtype == 'float32':
        tyname = torch.float32
    elif dtype == 'int32':
        tyname = torch.int32
    elif dtype == 'int64':
        tyname = torch.int64
    elif dtype == 'float16':
        tyname = torch.float16
    elif dtype == 'int16':
        tyname = torch.int16
    elif dtype == 'int8':
        tyname = torch.int8
    elif dtype == 'bool':
        tyname = torch.bool
    elif dtype == 'bfloat16':
        tyname = torch.bfloat16
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    return tyname

def generate_tensor(shape, dtype):
    """Generates a random tensor with the given shape and dtype, adapted to the value range of different numeric types."""
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def validate_cmp(dtype, y_cal, y_ref):
    """Compares the Triton result with the PyTorch reference on the NPU, setting tolerances or strict equality by dtype."""
    y_cal=y_cal.npu()
    y_ref=y_ref.npu()
    if dtype == 'float16':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32),  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
```

## 参数化测试

使用 `pytest` 对 `triton_dot_2_Bias` 内核进行参数化功能验证，覆盖不同矩阵维度和数据类型组合。

```python
# Test case configuration: (A, B, C) means matrix x: (A,B), y: (B,C), bias/output: (A,C)
testlist = [
    (16, 16, 16),
]

# List of supported data types (currently only float16)
typelist = ['float16',]

@pytest.mark.parametrize('A, B, C', testlist)
@pytest.mark.parametrize('sigtype', typelist)
def test_dot_2_Bias(sigtype, A, B, C):
    """Performs an end-to-end functional test of the triton_dot_2_Bias kernel."""
    dtype = get_torch_typename(sigtype)

    # Generate input tensors and move them to the NPU
    x0 = generate_tensor(shape=(A, B), dtype=sigtype).npu()
    x1 = generate_tensor(shape=(B, C), dtype=sigtype).npu()

    # The bias is always generated as float32 (to avoid precision issues with integer bias)
    if 'int' in sigtype:
        bias = generate_tensor(shape=(A, C), dtype='int32').npu()
        # Integer inputs must be converted to float32 for computation, then converted back to the target type
        ans = torch_dot_Bias(x0.to(torch.float32), x1.to(torch.float32), bias.to(torch.float32)).to(dtype)
    else:
        bias = generate_tensor(shape=(A, C), dtype='float32').npu()
        ans = torch_dot_Bias(x0, x1, bias).to(eval(f"torch.{dtype}"))

    # Initialize the output tensor
    output = torch.zeros((A, C), dtype=dtype).npu()

    # Launch the Triton kernel (grid=(1,1,1), executed with a single block)
    triton_dot_2_Bias[1, 1, 1](output, x0, x1, bias, A, B, C, debug=True)

    # Verify the correctness of the result
    validate_cmp(sigtype, output, ans)
    print(f"Test matmul with dtype={sigtype}, shape=({A},{B},{C}) PASSED!")


if __name__ == "__main__":
    # Supports directly running a single test case (for debugging convenience)
    test_dot_2_Bias("float16", 16, 16, 16)
```

**输出示例：**

```python
Test matmul with dtype=float16, shape=(16,16,16) PASSED!
```

上面输出日志表明Triton和Pytorch上的输出结果完全一致。
