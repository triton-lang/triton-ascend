# bl.alloc 接口文档

## 1. 背景

为了支持 Ascend 级编程的需要，需要支持用户手动创建指定地址空间上的内存（buffer）。本接口是硬件无关的接口，底层对接 `memref.alloc`。

## 2. 接口说明

```python
def alloc(
    etype: tl.dtype,
    shape: List[tl.constexpr],
    _address_space: address_space = None,
    is_mem_unique: bool = False,
    _builder=None,
) -> buffer:
```

## 3. 返回值

返回一个 buffer language 下的 `buffer` 类型，与 triton language 下的 `tensor` 做语义上的隔离，不支持相互赋值，需要通过 `to_tensor` 和 `to_buffer` 来显式转换。返回值表示一段分配在指定地址空间的内存，携带**数据类型**、**形状**和**地址空间**三部分信息。

## 4. 入参

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| `etype` | `tl.dtype` | 是 | 数据类型（element type） |
| `shape` | `List[tl.constexpr]` | 是 | buffer 的形状 |
| `_address_space` | `bl.address_space` | 否 | buffer 所在的地址空间，默认为 `None`（不携带地址空间信息） |
| `is_mem_unique` | `bool` | 否 | 是否独占内存。生成的 `annotation.mark` 在 Plan Memory 阶段用于判断是否允许内存复用。默认为 `False` |

## 5. 昇腾平台数据类型支持

| 平台 | uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | fp8e4nv | fp8e5 | bool |
|------|-------|------|--------|-------|--------|-------|--------|-------|------|------|------|------|---------|-------|------|
| Ascend A2/A3 | √ | √ | × | √ | × | √ | √ | √ | × | √ | × | √ | × | × | √ |
| Ascend 950 | √ | √ | × | √ | × | √ | √ | √ | × | √ | × | √ | × | × | √ |

## 6. 约束说明

- `etype` 不支持 `tl.int1`（bool 类型）
- `shape` 每个元素必须是正整数
- 需自行保证分配大小符合指定地址空间的容量限制
- `_address_space` 参数默认为 `None`，表示不携带任何地址空间信息
- `is_mem_unique=True` 时，Plan Memory 不会将该 buffer 与其他生命周期不重叠的 buffer 进行内存复用

## 7. 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al


@triton.jit
def allocate_local_buffer(XBLOCK: tl.constexpr):
    # 默认地址空间
    bl.alloc(tl.float32, [XBLOCK])
    # 指定 UB 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB)
    # 指定 L1 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L1)
    # 指定 L0A / L0B / L0C 地址空间
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0A)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0B)
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0C)
    # 指定 is_mem_unique=True
    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB,
             is_mem_unique=True)
```

## 8. 编译输出结果

以 `XBLOCK=256` 为例，编译后生成的 TTIR 核心片段如下：

```mlir
// 默认地址空间：memref 类型不携带 address_space 属性
%alloc = memref.alloc() : memref<256xf32>
annotation.mark %alloc {effects = ["write", "read"]} : memref<256xf32>

// UB 地址空间
%alloc_0 = memref.alloc() : memref<256x256xf32, #hivm.address_space<ub>>
annotation.mark %alloc_0 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<ub>>

// L1 地址空间（IR 助记符为 cbuf）
%alloc_1 = memref.alloc() : memref<256x256xf32, #hivm.address_space<cbuf>>
annotation.mark %alloc_1 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<cbuf>>

// L0A 地址空间（IR 助记符为 ca）
%alloc_2 = memref.alloc() : memref<256x256xf32, #hivm.address_space<ca>>
annotation.mark %alloc_2 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<ca>>

// L0B 地址空间（IR 助记符为 cb）
%alloc_3 = memref.alloc() : memref<256x256xf32, #hivm.address_space<cb>>
annotation.mark %alloc_3 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<cb>>

// L0C 地址空间（IR 助记符为 cc）
%alloc_4 = memref.alloc() : memref<256x256xf32, #hivm.address_space<cc>>
annotation.mark %alloc_4 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<cc>>

// is_mem_unique=True：额外生成 {mem_unique} 标记
%alloc_5 = memref.alloc() : memref<256x256xf32, #hivm.address_space<ub>>
annotation.mark %alloc_5 {mem_unique}
    : memref<256x256xf32, #hivm.address_space<ub>>
annotation.mark %alloc_5 {effects = ["write", "read"]}
    : memref<256x256xf32, #hivm.address_space<ub>>
```

### 输出要点说明

- 每个 `bl.alloc` 对应一个 `memref.alloc` op，memref 类型携带 shape、element type 和 `#hivm.address_space<...>` 属性
- 每个 alloc 后都会生成一个 `annotation.mark` 携带 `{effects = ["write", "read"]}`，标记该 buffer 可读可写
- 当 `is_mem_unique=True` 时，额外生成 `{mem_unique}` 标记，告知 Plan Memory 该 buffer 不可与其他 buffer 复用地址空间
- 地址空间在 IR 中的助记符：UB → `ub`，L1 → `cbuf`，L0A → `ca`，L0B → `cb`，L0C → `cc`

## 9. 相关接口

- [`bl.to_buffer`](to_buffer.md)：将 tensor 绑定到已分配的 buffer
- [`bl.to_tensor`](to_tensor.md)：将 buffer 转换为 tensor
- [`bl.subview`](subview.md)：从 buffer 中创建子视图
- [`al.ascend_address_space`](../al/ascend_address_space.md)：昇腾平台地址空间枚举
