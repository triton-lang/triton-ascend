# al.multibuffer

## 1. 函数概述

为张量设置多缓冲，允许编译器对同一张量创建多个副本。

```python
al.multibuffer(
    src,
    size,
    _semantic=None
) -> None
```

## 2. 规格

### 2.1 参数说明

| 参数名      | 类型                 | 说明                       |
| ----------- | -------------------- | -------------------------- |
| `src`       | `tensor`             | 需要进行多缓冲设置的源张量 |
| `size`      | `int` 或 `constexpr` | 要创建的缓冲区副本数量     |
| `_semantic` | -                    | 保留参数，暂不支持外部调用 |

此操作为一个编译提示，不会在运行时返回值，仅影响编译器的优化行为。

### 2.2 说明

| 限制参数 | 描述                           |
| -------- | ------------------------------ |
| `size`   | 当前实现仅支持 `size` 为 `2`。 |

`size` 的校验是一个 `assert isinstance(buffer_size, int) and buffer_size == 2`，不满足条件（传入 `1`、`4`、非整数等）会直接触发 `AssertionError: only support bufferize equals 2`。

### 2.3 实现说明与特殊限制

`al.multibuffer(src, size)` 底层实现就是 `compile_hint_impl(src, "hivm.multi_buffer", size, builder)`——本质上是 [`al.compile_hint`](./compile_hint.md) 的一个语义受限的封装（固定 `hint_name="hivm.multi_buffer"`，且强制 `size == 2`），两者共用同一套标注机制。

`al.compile_hint` 这个公开函数在调用真正的标注逻辑之前，会先判断 `if _semantic.builder.is_simt_mode(): return`，SIMT 模式下静默不生效；而 `al.multibuffer` 是直接调用底层的 `compile_hint_impl`，并不经过这层 SIMT 判断，所以它在 SIMT 模式下依然会正常生效。
## 3. 使用方法

以下示例展示了如何在 kernel 中为张量 `tmp0` 设置多缓冲，并结合其他编译提示使用：

```python
@triton.jit
def triton_compile_hint(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        al.compile_hint(tmp0, "hint_a")
        al.multibuffer(tmp0, 2)
        tmp2 = tmp0
        al.compile_hint(tmp2, "hint_b", 42)
        al.compile_hint(tmp2, "hint_c", True)
        al.compile_hint(tmp2, "hint_d", [XBLOCK, XBLOCK_SUB])
        tl.store(out_ptr0 + (xindex), tmp2, xmask)
```
