# triton.language.clamp
## 1. 函数概述

简介：限制张量x的范围为[min，max]之间。

```python
triton.language.clamp(x, min, max, propagate_nan: ~triton.language.core.constexpr = <PROPAGATE_NAN.NONE: 0>, _semantic=None)
```

## 2. 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `x`        | `tensor`          | 张量数据                                                      |
| `min`       | `constexpr`    | 最小值                                                        |
| `max`     | `constexpr`    | 最大值 |
| `propagate_nan` | `triton.language.core.constexpr` | 是否对min或max做Nan的传播                                              |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用

返回值：
`x`：输出张量的shape与输入x的shape相同

### 2.2 OP 规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | √    | √    | ×    |
| Ascend A2/A3 | ×    | ×     | ×     | ×     | ×     | ×      | ×      | ×     | √    | √    | ×    | √    | ×    |



#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

比 GPU 少了fp64的支持。



### 2.4 使用方法

以下示例实现了对输入张量 `x` 做截断计算：

```python
@triton.jit
def tt_clamp_2d(in_ptr, out_ptr, min_ptr, max_ptr,
                   xnumel: tl.constexpr, ynumel: tl.constexpr, znumel: tl.constexpr,
                   XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
       xoffs = tl.program_id(0) * XB
       yoffs = tl.program_id(1) * YB
       xidx = tl.arange(0, XB) + xoffs
       yidx = tl.arange(0, YB) + yoffs
       idx = xidx[:, None] * ynumel + yidx[None, :]

       x = tl.load(in_ptr + idx)
       min_ = tl.load(min_ptr + idx)
       max_ = tl.load(max_ptr + idx)
       ret = tl.clamp(x, min_, max_)

       tl.store(out_ptr + idx, ret)
```
