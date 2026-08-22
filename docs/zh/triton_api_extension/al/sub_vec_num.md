# al.sub_vec_num 接口文档

## 1. 硬件背景

Ascend NPU 使用 AI Cube Core（AIC）完成矩阵计算，使用 AI Vector Core（AIV）完成向量计算。
当一份 AIC 计算结果需要交给多个 AIV 分片处理时，编译器需要知道每个 AIC 对应多少个 AIV。

`al.sub_vec_num` 在 CPU/JIT 编译阶段查询当前活动 NPU 的 AIV 和 AIC 数量，并返回两者的
整数比值。

## 2. 接口说明

### 2.1 接口定义

源码中的函数签名如下，函数体在此省略：

```python
@builtin
def sub_vec_num(_semantic=None) -> tl.constexpr:
    ...
```

`_semantic` 由 Triton 编译器内部传入，用户调用时不需要提供：

```python
sub_num = al.sub_vec_num()
```

当前实现的返回值按以下方式计算：

```text
AIV Core 数量 // AIC Core 数量
```

### 2.2 参数

无用户参数。

### 2.3 返回值

返回一个 `tl.constexpr` 编译期常量，表示每个 AIC 对应的 AIV 数量。

## 3. 约束说明

- `al.sub_vec_num()` 只能在 `@triton.jit` 函数中调用。
- 返回值取决于当前活动 NPU 的 AIV/AIC 数量比例，不能假定所有设备都固定返回 2。
- 使用返回值切分数据时，调用者需要保证数据能被正确划分到各个分片。

## 4. 用例示例

不是完整可运行用例。示例假设 `BLOCK_ROWS`
能够被 `sub_num` 整除，并且 `rows_per_sub` 是 2 的幂，以满足 `tl.arange` 对区间长度的要求：

```python
sub_num: tl.constexpr = al.sub_vec_num()
sub_id = al.sub_vec_id()

rows_per_sub: tl.constexpr = BLOCK_ROWS // sub_num
row_offsets = sub_id * rows_per_sub + tl.arange(0, rows_per_sub)
```

`sub_num` 在编译期确定每个分片处理的行数，`sub_id` 在设备运行时区分当前 Vector 分片。
`tl.arange(0, rows_per_sub)` 生成一组连续的分片内行偏移，加上当前分片的起始位置后，
`row_offsets` 就是当前 Vector 分片需要处理的行索引。

## 5. 编译输出结果

`sub_vec_num` 在 JIT 编译期间返回常量，相关表达式会直接使用该常量进行编译，不会生成
独立的 `get_sub_block_num` IR 或设备端查询指令。

上例中的 `sub_vec_id()` 会生成运行时 ID，对应的关键 IR 为：

```mlir
%sub_id = hivm.hir.get_sub_block_idx -> i64
```
