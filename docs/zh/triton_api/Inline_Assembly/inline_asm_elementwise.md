# triton.language.inline_asm_elementwise
## 1. 函数概述

`inline_asm_elementwise` 用于在Triton内核中执行内联汇编代码，实现对张量的逐元素操作。

```python
triton.language.inline_asm_elementwise(asm, constraints, args, dtype, is_pure, pack, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `asm` | `str` | 必需 | 要执行的汇编代码，必须匹配目标平台的汇编格式 |
| `constraints` | `str` | 必需 | LLVM格式的汇编约束条件 |
| `args` | `tensor` | 必需 | 输入张量，其值会传递给汇编块 |
| `dtype` | `dtype` / `Sequence[dtype]` | 必需 | 返回张量的元素类型（可以是单个类型或类型元组） |
| `is_pure` | `bool` | 必需 | 如果为True，编译器假设汇编块没有副作用 |
| `pack` | `int` | 必需 | 每次内联汇编调用处理的元素数量 |
| `_semantic` | - | - | 保留参数，暂不支持外部调用 |

### 2.2 类型支持

A3:

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | ✓ | × | × | ×| × | ✓ |×|   ✓  | × | × | ×  |

Ascend 对比 GPU 的输入张量类型 缺失uint8、uint16、uint32、uint64、fp16、 fp64 、bf16 、bool的支持能力。

### 2.3 使用方法

```python
import triton.language as tl
@triton.jit
def triton_asm_add(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.inline_asm_elementwise(
        asm="""
        ADD.s64 $0, $1, $2
        """,
        constraints=(
            "=l,l,l"
        ),
        args=[x, y],
        dtype=tl.int64,
        is_pure=True,
        pack=1,
    )
    tl.store(output_ptr + offsets, output, mask=mask)
```

## 3. 语义GAP

1.内联汇编的寄存器仅支持`int64(s64)` 和`float32(f32)`。
2.约束限制仅支持`l`。
3.目前仅支持输入一维张量，计算高维张量需展开。
