# al.flip

## 1. 函数概述

将 tensor 沿某一维度进行翻转。

```python
al.flip(x, dim=-1, _semantic=None, _generator=None)
```

## 2. 规格

### 2.1 参数说明

| 参数名       | 类型     | 说明                       |
| ------------ | -------- | -------------------------- |
| `x`          | `tensor` | 张量数据                   |
| `dim`        | `int`    | 翻转维度，默认为 -1（最后一维） |
| `_semantic`  | -        | 保留参数，暂不支持外部调用 |
| `_generator` | -        | 保留参数，暂不支持外部调用 |

返回值：

`out`：输出张量的 shape 与输入 x 的 shape 相同

### 2.2 特殊限制说明

- 要求 `x` 的 rank ≥ 1，且 `dim` 归一化后必须落在 `[0, rank)` 范围内，否则 `raise ValueError`。
- **SIMT 模式和默认（SIMD）模式的实现完全不同，约束也不对称**（源码 `vec_ops.py::flip`）：
  - 非 SIMT 模式（默认走的 `flip_simd` 路径）直接调用底层 `builder.create_flip`，没有额外的维度大小限制。
  - **SIMT 模式下**，实现是先把翻转维度 reshape 成若干个 `2` 组成的维度，再做多轮 XOR-swap，因此有一条硬性前提：`core.static_assert(standard._is_power_of_two(ptr.shape[dim]))`——**被翻转维度上的元素个数必须是 2 的幂**，否则编译期直接断言失败。也就是说同一份 kernel 代码，在 SIMT 模式下可能因为翻转维度大小不是 2 的幂而编译失败，在默认模式下却完全没问题，写跨模式代码时需要特别注意这一点。

#### 2.2.1 DataType 支持

| | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| --- | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
|Ascend A2/A3| √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | ×    | √    | ×    |
|Ascend 950| √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | ×    | √    | ×    |

> `fp64` 均编译报错；其余类型（含 `uint16`/`uint32`/`uint64`）均编译、运行、结果正确，A2/A3 与 Ascend 950 结论一致；`bool` 不支持：kernel 内直接构造的 `tl.int1`（如比较运算结果）传给 `flip`，会在编译到硬件算子 `hivm.hir.vflip` 时报错（不接受 1-bit 类型）。若 bool 数据是从内存 `tl.load` 来的，会先被 Ascend 前端转成 int8 再参与运算，此时能跑通只是这层转换的副作用，不代表 `flip` 支持 bool，不建议依赖这种用法。

## 3. 使用方法

以下示例将输入张量 `X` 沿指定维度进行翻转：

```python
@triton.jit
def flip_kernel_2d(X, Z, M: tl.constexpr, N: tl.constexpr, dim: tl.constexpr):
    offx = tl.arange(0, M)
    offy = tl.arange(0, N) * M
    off2d = offx[None, :] + offy[:, None]
    x = tl.load(X + off2d)
    x = extension.flip(x, dim)
    tl.store(Z + off2d, x)
```
