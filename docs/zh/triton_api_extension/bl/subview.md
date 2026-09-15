# bl.subview 接口文档

## 1. 硬件背景

昇腾硬件支持在已有 buffer 上定义新的视图（subview），仅通过偏移（offsets）、大小（sizes）和步幅（strides）实现，不复制底层数据。该能力在硬件侧通过 memref 的 strided layout 直接映射，零开销。

## 2. 接口说明

`bl.subview` 提供两种等价的调用形式：函数式调用和成员方法调用。

### 接口一：函数形式

```python
def subview(
    src: bl.buffer,
    offsets: List[tl.constexpr | tl.tensor],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    _builder=None,
) -> bl.buffer:
```

### 接口二：成员方法形式

```python
# 作为 buffer 对象的方法调用，src 隐式为 self
def subview(
    self,
    offsets: List[tl.constexpr | tl.tensor],
    sizes: List[tl.constexpr],
    strides: List[tl.constexpr],
    _builder=None,
) -> bl.buffer:
```

**返回值**：`bl.buffer` —— 源 buffer 的一个子视图，与源 buffer 共享底层内存。

## 3. 入参说明

| 参数名 | 类型 | 是否必需 | 说明 |
|--------|------|----------|------|
| `src` | `bl.buffer` | 是 | 源 buffer（函数形式显式传入，成员方法形式为 `self`） |
| `offsets` | `List[tl.constexpr \| tl.tensor]` | 是 | 每一维的起始偏移；支持编译期常量或运行期 tensor 值 |
| `sizes` | `List[tl.constexpr]` | 是 | 子视图每一维的大小；必须是编译期常量 |
| `strides` | `List[tl.constexpr]` | 是 | 子视图每一维的步幅；必须是编译期常量，且当前所有元素必须为 `1` |

> **注意**：`sizes` 和 `strides` 必须传入 `tl.constexpr`（不要误传 tensor，否则会报类型不匹配）。`offsets` 额外支持 `tl.tensor` 传入（运行期动态偏移）。

## 4. 约束说明

- `sizes`、`strides` 中每个元素必须大于 0；`offsets` 中每个元素必须大于等于 0，不允许负值。
- 子视图每一维的大小不能超过源 buffer 对应维度的大小。
- `stride` 访问不得超出源 buffer 的边界；当前 `strides` 所有元素必须为 `1`。
- 参数必须为每一维都指定值，维度数与输入 buffer 的 rank 保持一致。
- `offsets` 对应的起始字节偏移必须 32 字节对齐。
- 子视图中第二行首元素相对于子视图起点的偏移（即最后一维步长 × 倒数第二维 size）必须 32 字节对齐。

## 5. 用例示例

```python
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al


@triton.jit
def subview_const_offset(XBLOCK: tl.constexpr):
    """固定偏移：裁掉上下各 1 行。"""
    src = bl.alloc(tl.float32, [XBLOCK, XBLOCK])
    # 在行方向跳过 1 行，列方向从 0 开始
    # 结果 shape: [XBLOCK-2, XBLOCK]
    sub = bl.subview(
        src,
        offsets=[1, 0],
        sizes=[XBLOCK - 2, XBLOCK],
        strides=[1, 1],
    )


@triton.jit
def subview_dynamic_offset(XBLOCK: tl.constexpr, offset: tl.constexpr,
                           size: tl.constexpr):
    """参数化偏移/大小。"""
    src = bl.alloc(tl.float32, [XBLOCK, XBLOCK])
    sub = bl.subview(
        src,
        offsets=[offset, 0],
        sizes=[size, XBLOCK],
        strides=[1, 1],
    )


@triton.jit
def subview_method_form(XBLOCK: tl.constexpr):
    """通过 buffer 成员方法调用。"""
    src = bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB)
    sub = src.subview(offsets=[0, 0], sizes=[XBLOCK // 2, XBLOCK],
                      strides=[1, 1])
```

## 6. 编译输出结果

以 `XBLOCK=8, offsets=[1,0], sizes=[6,8], strides=[1,1]` 为例，核心 IR 片段：

```mlir
%alloc = memref.alloc() : memref<8x8xf32>
annotation.mark %alloc {effects = ["write", "read"]} : memref<8x8xf32>

%c1_i32 = arith.constant 1 : i32
%c0_i32 = arith.constant 0 : i32
%0 = arith.index_cast %c1_i32 : i32 to index
%1 = arith.index_cast %c0_i32 : i32 to index

// memref.subview 直接在 memref 层表示子视图，不涉及数据拷贝
// 注意源类型为 memref<8x8xf32>，结果为 strided layout 的 memref<6x8xf32, strided<[8, 1], offset: ?>>
%subview = memref.subview %alloc[%0, %1] [6, 8] [1, 1]
    : memref<8x8xf32> to memref<6x8xf32, strided<[8, 1], offset: ?>>
```

### 输出要点说明

- `bl.subview` 底层对应 `memref.subview` op，是零拷贝视图操作，仅在类型/布局元数据层面描述子区域
- i32 偏移常量会先通过 `arith.index_cast` 转为 `index` 类型，再作为 subview 的 offset
- 结果类型为带 strided layout 的 memref（`strided<[outer_stride, inner_stride], offset: ?>`），保留了在源 buffer 中的相对偏移信息
- 由于不复制数据，对 subview 的写入会直接反映到源 buffer（以及与源 buffer 共享内存的其他非重叠 subview）

## 7. 相关接口

- [`bl.alloc`](alloc.md)：分配源 buffer
- [`bl.to_tensor`](to_tensor.md)：将 subview 结果转为 tensor 参与计算
- [`bl.to_buffer`](to_buffer.md)：tensor → buffer
