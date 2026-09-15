# al.compile_hint

## 1. 函数概述

`compile_hint` 是一个编译器提示（hint）机制，允许用户为张量附加元数据信息，这些信息会被传递到编译器后端，用于指导优化和代码生成。

```python
al.compile_hint(ptr, hint_name, hint_val=None, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数        | 类型                                          | 默认值 | 含义说明                         |
| ----------- | --------------------------------------------- | ------ | -------------------------------- |
| `ptr`       | `tensor`                                      | 必需   | 需要附加提示的张量对象           |
| `hint_name` | `str` `constexpr`                             | 必需   | 提示的名称标识符（必须为字符串） |
| `hint_val`  | `None` `bool` `int` `constexpr` `list` `tuple` | `None` | 提示的值，支持多种类型           |
| `_semantic` | -                                              | `None` | 保留参数，暂不支持外部调用       |

### 2.2 说明

1. **hint_name 必须为字符串类型**：内部会先对 `hint_name` 做 `constexpr` 解包，再 `assert isinstance(hint_name, str)`——传入其他类型会直接触发 `AssertionError: hint name: ... is not string`。
2. **list/tuple 参数仅支持整数数组**：元素须为整数（`int` 或 `constexpr` 整数），当前仅序列化为 i64 array attribute，不支持浮点数或混合类型的列表。
3. **非侵入式设计**：`compile_hint` 不改变计算语义，仅添加元数据。
4. **同一张量可多次标注**：同一个张量可以附加多个不同名称的提示。

### 2.3 特殊限制说明

- **SIMT 模式下是静默空操作**：`compile_hint` 内部第一步就是 `if _semantic.builder.is_simt_mode(): return`。也就是说在 SIMT 模式的 kernel 里调用 `al.compile_hint(...)` 不会报错，也不会附加任何标注，效果等同于没调用过。如果 hint 没生效，先确认当前是不是 SIMT 模式，而不是去查 hint 本身写得对不对。
- **`hint_val` 的 falsy 值会被吞掉，退化成"无值"标注**：值类型的判定顺序是先判断 `bool`，再判断 `not hint_val`，最后才轮到 `int`/`constexpr`/`list`。这意味着 `hint_val=0`、`hint_val=""`、`hint_val=[]` 这些 falsy 但"看起来是有效值"的输入，会在 `int`/`list` 分支之前就被 `not hint_val` 拦截，退化成和不传 `hint_val`（即 `None`）完全一样的 unit attribute，而不是 `int32_attr(0)` 或空数组。换句话说：
  ```python
  al.compile_hint(x, "foo", 0)   # 和下面这行生成的 IR 完全一样
  al.compile_hint(x, "foo")      # hint_val 默认 None
  ```
  如果确实需要传递整数 `0`，目前没有绕过这个判定顺序的办法，需要注意这个语义上的陷阱。

## 3. 使用方法

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
