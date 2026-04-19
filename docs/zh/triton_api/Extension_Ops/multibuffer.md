# triton.language.multibuffer

## 1. OP 概述

简介：为张量设置多缓冲，允许编译器对同一张量创建多个副本。
原型：

```python
triton.language.multibuffer(
 src,
 size,
 _builder=None
)→ None
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `src`        | `tensor`          | 需要进行多缓冲设置的源张量                                                     |
| `size`       | `int` 或 `constexpr`    | 要创建的缓冲区副本数量                                                        |
| `_builder` |- | 保留参数，暂不支持外部调用                                            |

返回值：
`None`：此操作为一个编译提示，不会在运行时返回值，仅影响编译器的优化行为。

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 |  bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √       | √         |  √       | √    | √    |  √    | √    |

#### 2.2.2 Shape 支持

支持任意形状的张量。

### 2.3 特殊限制说明

| 限制参数                   | 描述                                                                           |
| --------------------- | ---------------------------------------------------------------------------- |
|`size` | 当前实现仅支持 `size` 为 `2`。 |

### 2.4 使用方法

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
        # 为 tmp0 设置双缓冲
        tl.multibuffer(tmp0, 2)
        tmp2 = tmp0
        tl.compile_hint(tmp2, "hint_b", 42)
        tl.compile_hint(tmp2, "hint_c", True)
        tl.compile_hint(tmp2, "hint_d", [XBLOCK, XBLOCK_SUB])
        tl.store(out_ptr0 + (xindex), tmp2, xmask)
```
