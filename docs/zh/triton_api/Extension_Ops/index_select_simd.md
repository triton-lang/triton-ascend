# triton.language.extra.ascend.libdevice.index_select_simd
## 1 功能作用说明

在非尾轴维度上并行 gather 多个索引，并以 tile 为单位将数据零拷贝地从全局内存（GM）直接搬运到统一缓冲区（UB）的正确位置。该操作等效于 `torch.index_select` 的高性能实现，适用于嵌入层查找、稀疏索引访问等场景。

**语法：**

- `triton.language.extra.ascend.libdevice.index_select_simd(src, dim, index, src_shape, src_offset, read_shape)`

**功能：**

- 在源张量的指定维度上，根据索引数组批量读取数据
- 支持指定读取区域的偏移和大小，实现灵活的切片
- 零拷贝高效实现，直接从 GM 搬运到 UB
- 保持元素类型和编码方式不变

**典型应用场景：**

- 嵌入层（Embedding）查找：从大词汇表中根据 token ID 批量读取词向量
- 稀疏张量操作：根据稀疏索引访问密集张量的特定行
- 动态路由和注意力机制：根据动态计算的索引选择特定特征

## 2 参数规格

### 2.1 参数说明

| 参数名 | 类型 | 必需 | 说明 |
|--------|------|------|------|
| src | tensor/pointer | 是 | 源张量指针，位于全局内存（GM）上的数据 |
| dim | int | 是 | 在哪个维度执行 index_select 操作，取值范围 [0, len(src_shape)-2]，**不支持尾轴**（最后一个维度） |
| index | tensor | 是 | 1D 索引数组，位于 UB 上，指定要读取的索引位置 |
| src_shape | Tuple[int] | 是 | 源张量的完整形状 |
| src_offset | Tuple[int] | 是 | 从哪个位置开始读取，在 dim 维度可设为 -1（该维度由 index 决定） |
| read_shape | Tuple[int] | 是 | 读取数据的大小，在 dim 维度必须设为 -1（该维度由 index 长度决定） |

**返回值：**

- **类型：** tensor (位于 UB 上)
- **形状：** 与 read_shape 一致，其中 dim 维度的大小等于 index 的长度
- **数据类型：** 与源张量相同
- **内存位置：** 统一缓冲区（UB）

**约束条件：**

- `read_shape[dim]` 必须为 -1
- `src_offset[dim]` 可以设为 -1（会被忽略，因为该维度由 index 决定）
- `len(src_shape) == len(src_offset) == len(read_shape)`
- `index` 必须是 1D 张量
- `dim` 不能是尾轴（最后一个维度），即 `dim < len(src_shape) - 1`
- 对于非 dim 维度：`0 <= src_offset[i] < src_shape[i]`
- 对于非 dim 维度：`src_offset[i] + read_shape[i] <= src_shape[i]`（超出边界自动截断）
- index 中的索引值必须在 `[0, src_shape[dim])` 范围内

### 2.2 DataType支持表

| 支持情况 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | float16 | float32 | bfloat16 | float8e4 | float8e5 | float64 | bool |
|----------|:----:|:-----:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|:------:|:------:|:-------:|:----:|:----:|:------:|:---:|
| Ascend A2/A3 | ✓ | ✓ | ✓ | ✓ | ✓ | × | × | × | ✓ | ✓ | ✓ | × | × | × | ✓ |
| GPU支持 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

**说明：**

- index 的数据类型必须为 int32 或 int64
- GPU 平台不支持此操作（Ascend 专用 intrinsic）

### 2.3 Shape支持表

支持任意维度数（1D 到高维张量），但需满足以下条件：

- index 必须是 1D 张量
- 源张量的各维度大小需根据实际硬件内存限制
- read_shape 在非 dim 维度上的大小需考虑 UB 空间限制

**常见形状组合：**

- 2D 张量：适用于嵌入层查找、稀疏矩阵行选择
- 3D 张量：适用于批量嵌入查找、序列特征提取
- 高维张量：适用于复杂的多维索引操作

### 2.4 特殊限制说明

1. **dim 限制：** 不支持在尾轴（最后一个维度）上执行 index_select 操作，dim 必须满足 `dim < len(src_shape) - 1`
2. **数据类型限制：** 暂不支持 uint16/uint32/uint64/float8/float64 数据类型
3. **索引越界：** 不检查 index 中的索引是否越界，用户需自行保证索引合法性

### 2.5 使用方法

**基本用法（2D 嵌入查找）：**

```python
import triton
import triton.language as tl
import triton.language.extra.ascend.libdevice as libdevice

@triton.jit
def embedding_kernel(
    embed_ptr,      # [vocab_size, embed_dim]
    indices_ptr,    # [batch_size]
    output_ptr,     # [batch_size, embed_dim]
    vocab_size: tl.constexpr,
    embed_dim: tl.constexpr,
):
    pid = tl.program_id(0)

    # 加载索引
    indices = tl.load(indices_ptr + pid * 16 + tl.arange(0, 16))

    # 使用 index_select 批量读取嵌入向量
    embeddings = libdevice.index_select_simd(
        src=embed_ptr,
        dim=0,
        index=indices,
        src_shape=(vocab_size, embed_dim),
        src_offset=(-1, 0),
        read_shape=(-1, embed_dim)
    )

    # 存储结果
    offsets = tl.arange(0, 16)[:, None] * embed_dim + tl.arange(0, embed_dim)[None, :]
    tl.store(output_ptr + pid * 16 * embed_dim + offsets, embeddings)
```

**与 torch.index_select 的关系：**

- `index_select_simd` 等价于 `torch.index_select(src, dim, index)` 加上切片操作
- 但 index_select_simd 在硬件层面实现，性能优于 PyTorch 实现（约 0.6~1.5x AscendC 性能）

**与常规 load 的差异：**

```python
## 常规 load 方式（低效）
for i in range(len(indices)):
    idx = tl.load(indices_ptr + i)
    offsets = idx * stride + tl.arange(0, size)
    data = tl.load(src_ptr + offsets)
    # ... 处理 data

## index_select 方式（高效）
indices = tl.load(indices_ptr + tl.arange(0, len(indices)))
data = libdevice.index_select_simd(
    src=src_ptr,
    dim=0,
    index=indices,
    src_shape=(...),
    src_offset=(-1, 0),
    read_shape=(-1, size)
)
## 一次性获取所有数据
```

## 3 与GPU差异

新增OP，无差异

## 4 测试用例说明

**测试文件：**

- `ascend/examples/pytest_ut/test_index_select.py` - 2D 张量 index_select 测试（多种形状组合）
