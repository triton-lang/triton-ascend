# bl.to_tensor 接口文档

## 1. 硬件背景

将 Ascend 上分配的 `bl.buffer` 对象转换为 `tl.tensor` 并返回，使其可参与 Triton 的张量计算（如 `tl.load`/`tl.store`/算术运算等）。与 `bl.to_buffer` 配对使用，构成 tensor ↔ buffer 的双向转换。

## 2. 接口说明

```python
def to_tensor(
    memref: bl.buffer,
    writable: bool = True,
    _builder=None,
) -> tl.tensor:
```

## 3. 参数说明

| 参数名 | 类型 | 是否必需 | 说明 |
|--------|------|----------|------|
| `memref` | `bl.buffer` | 是 | 输入的 `bl.buffer` 对象 |
| `writable` | `bool` | 否 | 若为 `True`，返回的 tensor 在 bufferization 阶段允许原地修改（write-through 到原 buffer）；默认为 `True` |
| `_builder` | - | 内部参数 | 编译器自动传参，用户无需使用 |

## 4. 返回值

返回一个 `tl.tensor`，与输入 buffer 共享底层内存，shape 和 element type 与 buffer 一致。

## 5. 昇腾平台数据类型支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e4nv | fp8e5 | bool |
|------|-------|------|--------|-------|--------|-------|--------|-------|------|------|------|------|---------|-------|------|
| Ascend A2/A3 | √ | √ | × | √ | × | √ | √ | √ | × | √ | × | √ | × | × | √ |
| Ascend 950 | √ | √ | × | √ | × | √ | √ | √ | × | √ | × | √ | × | × | √ |

## 6. 约束说明

- 接口约束规则与 `bl.alloc` 保持一致
- `writable=False` 时，编译器可以对该 tensor 做只读优化（如常量传播、跨 buffer 复用），但不能保证一定不写——用户必须确保内核不通过其他别名修改这块内存
- 仅可在 `@triton.jit` 修饰的内核函数中使用

## 7. 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al


@triton.jit
def to_tensor_kernel(XBLOCK: tl.constexpr):
    # 方式 1：通过 buffer 对象的成员方法调用
    buf1 = bl.alloc(tl.float32, [XBLOCK])
    t1 = buf1.to_tensor(writable=True)

    # 方式 2：通过 bl.to_tensor 函数形式调用
    buf2 = bl.alloc(tl.float32, [XBLOCK])
    t2 = bl.to_tensor(buf2, writable=True)

    # 3. writable=False 标记只读，便于编译器优化
    buf3 = bl.alloc(tl.float32, [XBLOCK], al.ascend_address_space.UB)
    t3 = bl.to_tensor(buf3, writable=False)
```

## 8. 编译输出结果

以 `XBLOCK=256` 为例，核心 IR 片段：

```mlir
// memref.alloc + effects annotation 与 bl.alloc 一致
%alloc = memref.alloc() : memref<256xf32>
annotation.mark %alloc {effects = ["write", "read"]} : memref<256xf32>

// to_tensor 在 bufferization dialect 中表示为 bufferization.to_tensor
//   restrict   —— 表明这块内存没有其他别名（独占）
//   writable   —— 对应 writable=True，允许写回
%0 = bufferization.to_tensor %alloc restrict writable : memref<256xf32>
```

### 输出要点说明

- `bl.to_tensor` 底层对应 `bufferization.to_tensor` op，本身不产生数据搬运，只是把 memref 类型"视图"成 tensor 类型
- `restrict` 属性表示该 buffer 在转换为 tensor 时没有其他活跃别名（由 TritonAscend buffer SSA 语义保证）
- `writable` 属性直接对应 `writable` 参数，影响 One-Shot Bufferize 阶段的就地化（in-place）决策

## 9. 相关接口

- [`bl.alloc`](alloc.md)：分配 buffer
- [`bl.to_buffer`](to_buffer.md)：tensor → buffer 的反向转换
