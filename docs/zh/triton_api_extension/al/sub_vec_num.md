# sub_vec_num

## 1. 硬件背景

Ascend NPU 使用 AI Cube Core（AIC）完成矩阵计算，使用 AI Vector Core（AIV）完成向量计算。当一份 AIC 计算结果需要交给多个 AIV 分片处理时，开发者需要知道每个 AIC 对应多少个 AIV。

sub_vec_num 编程接口在 CPU/JIT 编译阶段读取编译器可见的 AIV 和 AIC 数量，并返回 AIV 数量除以 AIC 数量所得的整数商。该结果可以与 sub_vec_id 配合使用，确定每个 Vector 核处理的数据分片。

## 2. 接口说明

<table>
  <tr>
    <td>Python<br>def sub_vec_num() -&gt; tl.constexpr</td>
  </tr>
</table>

- 返回值：返回 tl.constexpr 编译期常量，值为 AIV Core 数量除以 AIC Core 数量所得的整数商，通常用于表示每个 AIC 对应的 AIV 数量

- 入参：无

## 3. 约束说明

- 仅能在 @triton.jit 函数中调用

- 返回值取决于 JIT 编译阶段可见的 AIV/AIC 数量比例，不能假定所有设备均固定返回 2

## 4. 用例示例

示例假设 BLOCK_ROWS 能够被 sub_num 整除，并且 rows_per_sub 是 2 的幂，以满足 tl.arange 对区间长度的要求。使用该片段切分数据时，调用者还需要保证各分片能够覆盖全部待处理数据：

```python
sub_num: tl.constexpr = al.sub_vec_num()
sub_id = al.sub_vec_id()

rows_per_sub: tl.constexpr = BLOCK_ROWS // sub_num
row_offsets = sub_id * rows_per_sub + tl.arange(0, rows_per_sub)
```

sub_num 在编译期确定每个分片处理的行数，sub_id 在设备运行时区分当前 Vector 分片。tl.arange(0, rows_per_sub) 生成一组连续的分片内行偏移，加上当前分片的起始位置后，row_offsets 就是当前 Vector 分片需要处理的行索引。

输出：

sub_vec_num 的结果在 JIT 编译期间参与常量计算。
