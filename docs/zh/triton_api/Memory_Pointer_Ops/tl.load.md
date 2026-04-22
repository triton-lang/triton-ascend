# triton.language.load

## 1. OP 概述

原型：

```python
triton.language.load(
 pointer,
 mask=None,
 other=None,
 boundary_check=(),
 padding_option='',
 cache_modifier='',
 eviction_policy='',
 volatile=False,
 _semantic=None
)
```

简介：返回一个Tensor/Scalar，其值从GlobalMemory中`pointer`参数指向的位置加载。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `pointer`        | `triton.PointerType` <br> 或 `tensor<triton.PointerType>` <br> 或`triton.PointerType<tensor>`（来源于`tl.make_block_ptr`）         | 指向GM上待读取数据的指针                                                    |
| `mask`       | `int1`或`tensor<int1>`    | 可选参数，当且仅当`pointer` 不来源于`tl.make_block_ptr`时可传入<br>若`mask[i]==False` ，则不会读取`pointer[i]`指向的数据,是`True`则正常读取 <br>若`pointer`来源于`tl.make_block_ptr`，则`mask`必须是`None`                                        |
| `other`     | `tensor` 或`scalar`   | 可选参数，当且仅当`mask!=None`时可传入<br> 若`mask[i]==False` ，将返回值的第`i`个位置设置为`other[i]`或`other`（若`other`是`scalar`类型）, 需要支持tensor，因为tritonGPU社区上是tensor和scalar都支持的，other[]] = mask[i]|
| `boundary_check` | `tuple(int)` | 可选参数，当且仅当`pointer`来源于`tl.make_block_ptr`时可传入<br>整数元组，指示需要做边界检查的维度                                         |
| `padding_option`   | `""`或`"zero"`或`"nan"`               | 可选参数，当且仅当`boundary_check`不为空时可传入<br>表示访问越界时填充的值 |
| `cache_modifier`   | `""` 或 `"ca"`或`"cg"`                | 可选参数，控制NVIDIA PTX上的cache选项，对Ascend硬件无效                                                |
| `eviction_policy`   | `str`                | 控制NVIDIA PTX的eviction策略， 对Ascend硬件无效                                                |
| `volatile`   | `str`                 | 控制NVIDIA PTX的volatile选项， 对Ascend硬件无效                                        |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用                                                |

当前910代际均还不支持cache_modifier，eviction_policy， volatile等参数

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | ×    | √    |  √    |

结论：Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 支持scalar和1~5维tensor |
| Ascend A2/A3 | 支持scalar和1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

#### 2.2.3 社区约束

1. 若`pointer`是一个单指针：
   - 此时`tl.load`返回一个标量
   - `mask`和`other`必须是标量
   - `other`会隐式类型转换成`pointer.dtype.element_ty`的数据类型
   - 此时不允许传入`boundary_check`和`padding_option`
2. 若`pointer`是一个N-Dimensional tensor：
   - 此时`tl.load`返回一个与`pointer`shape相同的N-Dimensional tensor
   - `mask`和`other`会隐式广播到和`pointer`相同的shape
   - 此时不允许传入`boundary_check`和`padding_option`
3. 若`pointer`来自于`tl.make_block_ptr`:
   - 此时`mask` 和 `other` 必须是None
   - 此时可以通过`boundary_check`和`padding_option`设置边界检查和越界补充值

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

| 差异点                                 | 描述                                                         | 解决途径                       |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------ |
| 不支持`padding_option`入参             | 当前使用的社区分支新增`padding_option`入参，用于越界元素填充策略。 | 可软件开发支持                 |
| 与分支、循环语句搭配使用时的泛化性问题 | 当前tl.load的`pointer`和`mask`的计算过程，如果涉及较复杂的循环和分支语句，可能会出现编译问题 | 大量泛化测试暴露问题，迭代解决 |

### 2.4 使用方法

以下示例中通过`triton_ldst_indirect_07_kernel`和`triton_ldst_indirect_07_func`的配合调用，实现了`torch_ldst_indirect_07_func`的功能：

```python
@triton.jit
def triton_ldst_indirect_07_kernel(
    out_ptr0, in_ptr0, in_ptr1, in_ptr2, stride_in_r,
    XS: tl.constexpr, RS: tl.constexpr
):
    pid = tl.program_id(0)
    in_idx0 = pid * XS + tl.arange(0, XS)
    in_idx1 = tl.arange(0, RS)
    tmp0 = tl.load(in_ptr0 + in_idx0)
    tmp1 = tl.load(in_ptr1 + in_idx1)
    in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
    tmp2 = tl.load(in_ptr2 + in_idx2)
    out0_idx = in_idx0[:, None] * RS + in_idx1[None, :]
    tl.store(out_ptr0 + out0_idx, tmp2)

def triton_ldst_indirect_07_func(xr, xc, x2, xs, rs):
    nr = x2.size()[0]
    nc = xc.numel()
    stride_in_r = x2.stride()[0]
    assert nr == xs, "test only single core"
    y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
    triton_ldst_indirect_07_kernel[nr // xs, 1, 1](
        y0, xr, xc, x2, stride_in_r, XS = xs, RS = rs)
    return y0

def torch_ldst_indirect_07_func(xr, xc, x2):
    flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
    extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
    return extracted

DEV = "npu"
DTYPE = torch.float32
offset = 8
N0, N1 = 16, 32
blocksize = 4
lowdimsize = N0
assert N1 >= N0+offset, "N1 must be >= N0+offset"
assert N0 == lowdimsize, "N0 must be == lowdimsize"
xc = offset + torch.arange(0, N0, device=DEV)
xr = torch.arange(0, blocksize, device=DEV)
x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
torch_ref = torch_ldst_indirect_07_func(xr, xc, x2)
triton_cal = triton_ldst_indirect_07_func(xr, xc, x2, blocksize, lowdimsize)
torch.testing.assert_close(triton_cal, torch_ref)
```
