# triton.language.associative_scan
## 1. OP 概述

简介：`triton.language.associative_scan` 对输入tensor沿指定轴应用关联扫描操作，使用combine_fn函数组合元素并更新进位值。

```
triton.language.associative_scan(input, axis, combine_fn, reverse=False, _semantic=None, _generator=None)
```

## 2. OP 规格

### 2.1 参数说明

| 参数 | 类型 | 含义说明 |
|--------|------|------|
| `input` | `Tensor` 或 `tuple of Tensor` | 输入tensor，可以是单个tensor或tensor元组 |
| `axis` | `int` | 沿着哪个维度进行关联扫描操作 |
| `combine_fn` | `Callable` | 用于组合两个标量tensor组的函数（必须用@triton.jit标记） |
| `reverse` | `bool` | 是否沿轴的反方向应用关联扫描|
| `_semantic` | `Optional[str]` | 保留参数，暂不支持外部调用 |
| `_generator` | `Optional[Generator]` | 保留参数，暂不支持外部调用 |


返回值：
`tensor`：对输入tensor沿指定轴应用关联扫描操作，使用combine_fn函数组合元素并更新进位值之后的tensor。

### 2.2 支持规格

#### 2.2.1 DataType 支持


|| uint8 | int8 | uint16 | int16 | uint32 | int32 | uint64 | int64 | fp16 | fp32 | bf16 | bool/int1 |
|---| ------- | ------ | -------- | ------- | -------- | ------- | -------- | ------- | ------ | ------ | ------ | ----------- |
| GPU支持 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Ascend A2/A3 | ✓ | ✓ | × | ✓ | × | ✓ | × | ✓ | ✓ | ✓ | ✓ | ✓ |





#### 2.2.2 Shape 支持

结论：在 Shape 方面，GPU 与 Ascend 平台无差异。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现
> reverse=True是否沿轴的反方向应用关联扫描，该功能需要tl.load加载数据时对齐，即不使用mask过滤掉多余数据索引，即如下面示例代码：

```python
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)
```

### 2.4 使用方法

以下示例实现了对2Dshape的tensor进行associative_scan运算：

```python

@triton.jit
def bitwise_and_fn(a, b):
    return a & b


@triton.jit
def bitwise_or_fn(a, b):
    return a | b


@triton.jit
def bitwise_xor_fn(a, b):
    return a ^ b


@triton.jit
def minimum_fn(a, b):
    return tl.minimum(a, b)


@triton.jit
def maximum_fn(a, b):
    return tl.maximum(a, b)

@triton.jit
def triton_kernel_2d_scan(
        out_ptr0,
        in_ptr0,
        dim: tl.constexpr,
        reverse: tl.constexpr,
        numel_x: tl.constexpr,
        numel_r: tl.constexpr,
        XBLOCK: tl.constexpr,
        RBLOCK: tl.constexpr,
        combine_fn_name: tl.constexpr,
):
    tl.static_assert(
        numel_x == XBLOCK, "numel_x must be equal to XBLOCK in this kernel"
    )
    tl.static_assert(
        numel_r == RBLOCK, "numel_r must be equal to RBLOCK in this kernel"
    )
    idx_x = tl.arange(0, XBLOCK)
    idx_r = tl.arange(0, RBLOCK)
    idx = idx_x[:, None] * numel_r + idx_r[None, :]
    x = tl.load(in_ptr0 + idx)

    if combine_fn_name == "maximum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=maximum_fn)
    elif combine_fn_name == "minimum_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=minimum_fn)
    elif combine_fn_name == "bitwise_or_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_or_fn)
    elif combine_fn_name == "bitwise_xor_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_xor_fn)
    elif combine_fn_name == "bitwise_and_fn":
        ret = tl.associative_scan(x, axis=dim, reverse=reverse, combine_fn=bitwise_and_fn)
    tl.store(out_ptr0 + idx, ret)

```
