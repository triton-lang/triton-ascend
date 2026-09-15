# bl.to_buffer 接口文档

## 1. 硬件背景

用于将 `tl.tensor` 张量对象转换为昇腾硬件专用的 `bl.buffer` 缓冲区对象，是张量与硬件内存缓冲区之间的核心转换接口。

## 2. 接口说明

```python
def to_buffer(
    tensor: tl.tensor,
    space: address_space = None,
    bind_buffer: buffer = None,
    _builder=None,
) -> buffer:
```

## 3. 参数说明

| 参数名 | 类型 | 是否必需 | 说明 |
|--------|------|----------|------|
| `tensor` | `tl.tensor` | 是 | 需要转换为缓冲区的输入张量 |
| `space` | `bl.address_space` | 否 | 指定目标缓冲区所在的昇腾硬件地址空间（UB / L1 / L0A / L0B / L0C） |
| `bind_buffer` | `bl.buffer` | 否 | 可选，将张量直接绑定到已有的目标缓冲区，复用其内存 |
| `_builder` | - | 内部参数 | 编译器自动传参，用户无需使用 |

## 4. 返回值

- 返回与输入张量对应的 `bl.buffer` 对象
- 若传入 `bind_buffer` 参数，则直接返回该绑定缓冲区本身（不会新分配内存）

## 5. 昇腾平台数据类型支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e4nv | fp8e5 | bool |
|------|-------|------|--------|-------|--------|-------|--------|-------|------|------|------|------|---------|-------|------|
| Ascend A2/A3 | √ | √ | × | √ | × | √ | √ | √ | × | √ | × | √ | × | × | √ |
| Ascend 950 | √ | √ | × | √ | × | √ | √ | √ | × | √ | × | √ | × | × | √ |

## 6. 约束说明

- 接口约束规则与 `bl.alloc` 保持一致（数据类型、shape 合法性等）
- 地址空间参数需严格匹配昇腾硬件支持的内存区域（UB / L1 / L0A / L0B / L0C）
- 使用 `bind_buffer` 时，`tensor` 和 `bind_buffer` 的 shape 与 element type 必须完全一致
- 同一个 tensor 不能绑定到多个 buffer

## 7. 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al


@triton.jit
def to_buffer_kernel():
    # 1. 基础转换：不指定地址空间（由编译器自动选择）
    a = tl.full((32, 2, 4), 0, dtype=tl.int64)
    a_buf = bl.to_buffer(a)

    # 2. 转换并指定 UB 地址空间
    b = tl.full((32, 2, 4), 0, dtype=tl.int64)
    b_buf = bl.to_buffer(b, al.ascend_address_space.UB)

    # 3. 转换并指定 L1 地址空间
    c = tl.full((32, 2, 4), 0, dtype=tl.int64)
    c_buf = bl.to_buffer(c, al.ascend_address_space.L1)

    # 4. 转换并指定 L0A 地址空间
    d = tl.full((32, 2, 4), 0, dtype=tl.int64)
    d_buf = bl.to_buffer(d, al.ascend_address_space.L0A)

    # 5. 转换并指定 L0B 地址空间
    e = tl.full((32, 2, 4), 0, dtype=tl.int64)
    e_buf = bl.to_buffer(e, al.ascend_address_space.L0B)

    # 6. 转换并指定 L0C 地址空间
    f = tl.full((32, 2, 4), 0, dtype=tl.int64)
    f_buf = bl.to_buffer(f, al.ascend_address_space.L0C)

    # 7. 绑定到预先分配的 buffer（不新分配内存）
    ub = bl.alloc(tl.int64, [32, 2, 4], al.ascend_address_space.UB)
    g = tl.full((32, 2, 4), 0, dtype=tl.int64)
    g_buf = bl.to_buffer(g, bind_buffer=ub)  # 复用 ub 的内存
```

## 8. 核心说明

- 该接口是 tensor ↔ 硬件 buffer 的核心转换入口
- 支持手动指定昇腾全系列硬件地址空间（UB / L1 / L0A / L0B / L0C）
- 支持绑定现有缓冲区，满足精细化内存管理需求（避免重复分配、手动控制 buffer 生命周期）
- 仅可在 `@triton.jit` 修饰的内核函数中使用

## 9. 相关接口

- [`bl.alloc`](alloc.md)：在指定地址空间分配新 buffer
- [`bl.to_tensor`](to_tensor.md)：将 buffer 转换回 tensor
- [`bl.bind_buffer`](bind_buffer.md)：bind_buffer 语义
- [`al.ascend_address_space`](../al/ascend_address_space.md)：昇腾平台地址空间枚举
