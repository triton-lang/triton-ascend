# triton.language.static_print

## 1.函数概述

`static_print` 用于在编译时打印信息，类似于Python的`print()`函数，但它在内核编译期间执行而不是运行时执行。

```python
triton.language.static_print(*values, sep: str = ' ', end: str = '\n', file=None, flush=False, _semantic=None)
```

## 2.规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `values`| `tensor`/`scalar`| 必需 | 要打印的值，支持多个参数 |
| `sep` | `str` | `' '` | 值之间的分隔符 |
| `end` | `str` | `'\n'` | 打印结束时的后缀 |
|`file` | - | - | 写入的文件对象 |
|`flush` | `bool` | `False` | 是否刷新输出缓冲区 |
|`_semantic` | - | - | 保留参数，暂不支持外部调用 |

### 2.2.1 Data Type 支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ | ✓ | ✓ | × | ✓ | ✓ |

### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

### 2.4 使用方法

```python
import triton.language as tl

@triton.jit
def basic_static_print_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # 在编译时打印常量的值
    tl.static_print("BLOCK_SIZE =", BLOCK_SIZE)
    tl.static_print(BLOCK_SIZE)
    # 支持fstring打印方式
    tl.static_print(f"BLOCK_SIZE={BLOCK_SIZE}")
```

如果打印的**非常量**结果，会打印一个`数据类型[数据shape(标量为空)]`的值，如果下面的代码`x_ptr`指向的数据类型为`int32`就会打印`val:int32[constexpr[4]]`的结果

```python
import triton.language as tl

@triton.jit
def basic_static_print_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    idx = arange(0,4)
    val = tl.load(x_ptr + idx)
    tl.static_print("val:",val)
    #非常量不支fstring打印
    #tl.static_print(f"val:{val}")
```
