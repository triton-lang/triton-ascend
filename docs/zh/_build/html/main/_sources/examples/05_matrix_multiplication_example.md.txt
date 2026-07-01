# 矩阵乘法 （Matrix Multiplication）

在本节中，我们展示了使用 Triton 进行矩阵乘法的内核实现。

## 计算内核

以下 Triton 内核实现了一个带偏置项的批量矩阵乘法（Batched Matrix Multiplication with Bias）：  
计算公式为：  
$$ \text{output}[b, i, j] = \sum_k \text{x}[b, i, k] \cdot \text{y}[k, j] + \text{z}[b, i, j] $$  
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
    output_ptr,   # 输出张量指针，形状 (A, C)
    x_ptr,        # 输入张量 x 指针，形状 (A, B)
    y_ptr,        # 输入张量 y 指针，形状 (B, C)
    z_ptr,        # 偏置张量 z 指针，形状 (A, C)
    A: tl.constexpr,  # 第一维度大小（batch / 行数）
    B: tl.constexpr,  # 共享维度（x 的列数，y 的行数）
    C: tl.constexpr   # 第二维度大小（列数）
):
    # 创建索引向量
    bidx = tl.arange(0, A)  # [0, 1, ..., A-1]，用于行维度
    cidx = tl.arange(0, B)  # [0, 1, ..., B-1]，用于 x 的列 / y 的行
    didx = tl.arange(0, C)  # [0, 1, ..., C-1]，用于列维度

    # 构造 x 的线性索引：(A, B) -> 展平为 A*B
    Xidx = bidx[:, None] * B + cidx[None, :]  # 广播形成 (A, B) 索引网格

    # 构造 y 的线性索引：(B, C) -> 展平为 B*C
    Yidx = cidx[:, None] * C + didx[None, :]  # (B, C) 索引网格

    # 构造 z 和 output 的线性索引：(A, C)
    Zidx = bidx[:, None] * C + didx[None, :]  # (A, C) 索引网格

    # 从全局内存加载数据
    X = tl.load(x_ptr + Xidx)  # 加载 (A, B) 子块
    Y = tl.load(y_ptr + Yidx)  # 加载 (B, C) 子块
    Z = tl.load(z_ptr + Zidx)  # 加载偏置 (A, C)

    # 执行矩阵乘法并加上偏置
    ret = tl.dot(X, Y) + Z  # tl.dot 执行 (A, B) × (B, C) → (A, C)

    # 写回结果到全局内存
    oidx = bidx[:, None] * C + didx[None, :]  # 与 Zidx 相同，可复用
    tl.store(output_ptr + oidx, ret)
```

## 工具方法

以下辅助函数用于支持 Triton 内核的测试与验证，包括 PyTorch 参考实现、数据类型映射、随机张量生成及结果校验。

```Python
def torch_dot_Bias(x0, x1, bias):
    """PyTorch 参考实现：执行矩阵乘法并加上偏置项。"""
    res = torch.matmul(x0, x1) + bias
    return res

def get_torch_typename(dtype):
    """将字符串形式的数据类型映射为对应的 torch.dtype。"""
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
     """根据指定形状和数据类型生成随机张量，适配不同数值类型的取值范围。"""
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
    """在 NPU 上比较 Triton 计算结果与 PyTorch 参考结果，按数据类型设置容差或严格相等。"""
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
# 测试用例配置：(A, B, C) 表示矩阵 x: (A,B), y: (B,C), bias/output: (A,C)
testlist = [   
    (16, 16, 16), 
]

# 支持的数据类型列表（当前仅 float16）
typelist = ['float16',]

@pytest.mark.parametrize('A, B, C', testlist)
@pytest.mark.parametrize('sigtype', typelist)
def test_dot_2_Bias(sigtype, A, B, C):
    """对 triton_dot_2_Bias 内核进行端到端功能测试。"""
    dtype = get_torch_typename(sigtype)
    
    # 生成输入张量并移至 NPU
    x0 = generate_tensor(shape=(A, B), dtype=sigtype).npu()
    x1 = generate_tensor(shape=(B, C), dtype=sigtype).npu()
    
    # 偏置项统一用 float32 生成（避免整数偏置导致精度问题）
    if 'int' in sigtype:
        bias = generate_tensor(shape=(A, C), dtype='int32').npu()
        # 整数输入需转为 float32 计算后再转回目标类型
        ans = torch_dot_Bias(x0.to(torch.float32), x1.to(torch.float32), bias.to(torch.float32)).to(dtype)
    else:
        bias = generate_tensor(shape=(A, C), dtype='float32').npu()
        ans = torch_dot_Bias(x0, x1, bias).to(eval(f"torch.{dtype}"))
    
    # 初始化输出张量
    output = torch.zeros((A, C), dtype=dtype).npu()
    
    # 启动 Triton 内核（grid=(1,1,1)，单 block 执行）
    triton_dot_2_Bias[1, 1, 1](output, x0, x1, bias, A, B, C, debug=True)
    
    # 验证结果正确性
    validate_cmp(sigtype, output, ans)
    print(f"Test matmul with dtype={sigtype}, shape=({A},{B},{C}) PASSED!")


if __name__ == "__main__":
    # 支持直接运行单个测试用例（便于调试）
    test_dot_2_Bias("float16", 16, 16, 16)
```

**输出示例：**

```python
Test matmul with dtype=float16, shape=(16,16,16) PASSED!
```

上面输出日志表明Triton和Pytorch上的输出结果完全一致。
