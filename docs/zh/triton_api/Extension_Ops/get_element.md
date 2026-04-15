# triton.language.get_element
## 1. OP 概述

简介：根据给定的索引，从输入张量中读取单个元素。
原型：

```python
triton.language.get_element(
	src,
	indice,
	_builder=None,
	_generator=None
)→ scalar
```

可以作为tensor的成员函数调用，如`x.get_element(...)`，与`get_element(x, ...)`等效。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `src`        | `tensor`          | 要被访问的源张量                                                     |
| `indice`       | `tuple of ints` 或 `tuple of tensors`    | 用于指定元素位置的索引                                                    |
| `_builder` |- | 保留参数，暂不支持外部调用                                            |
| `_generator`   | -               | 保留参数，暂不支持外部调用                                                |

返回值：
`scalar`：与 `src` 张量元素类型相同的标量值

### 2.2 支持规格

#### 2.2.1 DataType 支持


|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- |
| Ascend A2/A3 | √    | √     | √     | √     | √     | √       | √         |  √       | √    | √    |  √    | ×    |



#### 2.2.2 Shape 支持

支持任意形状的张量，但需满足：
`indice` 的长度必须与 `src` 张量的维度数相同。

### 2.3 特殊限制说明

无特殊限制

### 2.4 使用方法

以下示例实现了get_element的调用：

```python@triton.jit
def index_select_manual_kernel(in_ptr, indices_ptr, out_ptr, dim,
                                g_stride: tl.constexpr, indice_length: tl.constexpr,
                                g_block: tl.constexpr, g_block_sub: tl.constexpr,
                                other_block: tl.constexpr):
    """
    Manual implementation using tl.get_element and tl.insert_slice.
    """
    g_begin = tl.program_id(0) * g_block
    for goffs in range(0, g_block, g_block_sub):
        g_idx = tl.arange(0, g_block_sub) + g_begin + goffs
        g_mask = g_idx < indice_length
        indices = tl.load(indices_ptr + g_idx, g_mask, other=0)

        for other_offset in range(0, g_stride, other_block):
            tmp_buf = tl.zeros((g_block_sub, other_block), in_ptr.dtype.element_ty)
            other_idx = tl.arange(0, other_block) + other_offset
            other_mask = other_idx < g_stride

            # Manual gather: iterate over each index
            for i in range(0, g_block_sub):
                gather_offset = tl.get_element(indices, (i,)) * g_stride
                val = tl.load(in_ptr + gather_offset + other_idx, other_mask)
                tmp_buf = tl.insert_slice(tmp_buf, val[None, :],
                                          offsets=(i, 0), sizes=(1, other_block), strides=(1, 1))

            tl.store(out_ptr + g_idx[:, None] * g_stride + other_idx[None, :],
                     tmp_buf, g_mask[:, None] & other_mask[None, :])
```

## 3. 语义GAP

无语义差异
