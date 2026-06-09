# triton.language.static_assert

## 1. 函数概述

`static_assert` 用于在编译时断言条件是否成立，如果条件不满足则编译失败。这是一个编译时检查工具，不需要设置调试环境变量。

```python
triton.language.static_assert(cond, msg='', _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数 | 类型 | 默认值 | 含义说明 |
|------|------|--------|----------|
| `cond` | `bool` | 必需 | 编译时需要断言的条件表达式 |
| `msg` | `str` | `''` | 断言失败时显示的错误消息 |
| `_semantic` | - | - | 保留参数，暂不支持外部调用 |

### 2.2 类型支持

A3：

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
|------|-------|-------|-------|-------|--------|--------|--------|-------|------|------|------|------|------|
| GPU | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |
| Ascend A2/A3 | × | × | × | × | × | × | × | × | × | × | × | × | ✓ |

**注意：** `cond` 语句中值的类型必须为 `constexpr`。

### 2.3 使用方法

```python
import triton.language as tl

@triton.jit
def basic_static_assert_example(x_ptr, BLOCK_SIZE: tl.constexpr):
    # 基本断言：检查BLOCK_SIZE是否为2的幂次
    tl.static_assert((BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0)

    # 带自定义错误消息的断言
    tl.static_assert(BLOCK_SIZE >= 64, "BLOCK_SIZE must be at least 64 for performance")

    # 在static_assert的条件中出现非常量会编译错误
    # val = tl.load(x_ptr)
    # tl.static_assert(val <= 64)
```
