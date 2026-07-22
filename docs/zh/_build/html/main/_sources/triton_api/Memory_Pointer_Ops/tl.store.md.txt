# triton.language.store

## 1. OP 概述

原型：

```python
triton.language.store(
 pointer,
 value,
 mask=None,
 boundary_check=(),
 cache_modifier='',
 eviction_policy='',
 _semantic=None
)
```

简介：将一个Tensor/Scalar从UnifiedBuffer按照`pointer`所指的地址存回GlobalMemory

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `pointer`        | `triton.PointerType` <br> 或 `tensor<triton.PointerType>` <br> 或`triton.PointerType<tensor>`（来源于`tl.make_block_ptr`）         | 指向GM上待存储地址的指针                                                    |
| `value`       | `tensor` 或 `scalar`  | 要存储的值，支持隐式广播和隐式类型转换  |
| `mask`       | `int1`或`tensor<int1>`    | 可选参数，当且仅当`pointer` 不来源于`tl.make_block_ptr`时可传入<br>若`mask[i]==False` ，则不会将`value[i]`存储到`pointer[i]`指向的地址,是`True`则正常存储 <br>若`pointer`来源于`tl.make_block_ptr`，则`mask`必须是`None`                                        |
| `boundary_check` | `tuple(int)` | 可选参数，当且仅当`pointer`来源于`tl.make_block_ptr`时可传入<br>整数元组，指示需要做边界检查的维度                                         |
| `cache_modifier`   | `""` 或 `"ca"`或`"cg"`                | 可选参数，控制NVIDIA PTX上的cache选项，对Ascend硬件无效                                                |
| `eviction_policy`   | `str`                | 控制NVIDIA PTX的eviction策略， 对Ascend硬件无效                                                |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用                                                |

返回值：无返回值

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    | √    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | ×    | √    |  √    |

结论：Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。
专家意见：eviction_policy和cache_modifier参见load

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 支持scalar和1~5维 tensor |
| Ascend | 支持scalar和1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

#### 2.2.3 社区约束

1. 若`pointer`是一个单指针：
   - 此时`value`和`mask`必须是一个标量
   - `other`会隐式类型转换成`pointer.dtype.element_ty`的数据类型
   - 此时不允许传入`boundary_check`
2. 若`pointer`是一个N-Dimensional tensor：
   - `mask`和`value`会隐式广播到和`pointer`相同的shape
   - 此时不允许传入`boundary_check`
3. 若`pointer`来自于`tl.make_block_ptr`:
   - 此时`mask` 必须是None
   - 此时可以通过`boundary_check`设置边界检查

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）, eviction_policy和cache_modifier在NPU上功能还不完善。

| 差异点                                 | 描述                                                         | 解决途径                       |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------------ |
| 离散mask的泛化性问题                   | 当前对于store中离散mask的处理，是将store拆解为atomic {load,select,store}，在corner case中存在一定泛化性问题 | 大量泛化测试暴露问题，迭代解决 |
| 与分支、循环语句搭配使用时的泛化性问题 | 当前tl.load的`pointer`和`mask`的计算过程，如果涉及较复杂的循环和分支语句，可能会出现编译问题 | 大量泛化测试暴露问题，迭代解决 |

### 2.4 使用方法

以下示例中通过`triton_ldst_indirect_08_kernel`和`triton_ldst_indirect_08_func`的配合调用，实现了`torch_ldst_indirect_08_func`的功能：

```python
@triton.jit
def triton_ldst_indirect_08_kernel(
    out_ptr0, in_ptr1, in_ptr2, in_ptr3, stride_in_r,
    XS: tl.constexpr, RS: tl.constexpr
):
    pid = tl.program_id(0)
    in_idx0 = pid * XS + tl.arange(0, XS)
    in_idx1 = tl.arange(0, RS)
    tmp0 = tl.arange(0, XS)
    tmp1 = tl.load(in_ptr1 + in_idx1)
    in_idx2 = tmp0[:, None] * stride_in_r + tmp1[None, :]
    tmp2 = tl.load(in_ptr2 + in_idx2)
    tmp2 = tl_math.exp(tmp2)
    tmp3 = tl.load(in_ptr3 + in_idx1)
    tmp3 = tmp3 + 1 - 8
    out0_idx = in_idx0[:, None] * RS + tmp3[None, :]
    tl.store(out_ptr0 + out0_idx, tmp2)
    
def triton_ldst_indirect_08_func(xc, x2, xs, rs): # [8-24] ori 8 16
    nr = x2.size()[0]
    nc = xc.numel()
    stride_in_r = x2.stride()[0]
    assert nr == xs, "test only single core"
    y0 = torch.empty((nr, nc), dtype=x2.dtype, device=x2.device)
    xc1 = xc - 1
    triton_ldst_indirect_08_kernel[nr // xs, 1, 1](
        y0, xc, x2, xc1, stride_in_r, XS = xs, RS = rs)
    return y0
    
def torch_ldst_indirect_08_func(xr, xc, x2):
    flatten_idx = (xr[:, None] * x2.stride()[0] + xc[None, :]).flatten()
    extracted = x2.flatten()[flatten_idx].reshape([xr.numel(), xc.numel()])
    print(extracted)
    return torch.exp(extracted)
    
DEV = "npu"
DTYPE = torch.float32
offset = 8
N0, N1 = 16, 32
blocksize = 8
lowdimsize = N0
assert N1 >= N0+offset, "N1 must be >= N0+offset"
assert N0 == lowdimsize, "N0 must be == lowdimsize"
xc = offset + torch.arange(0, N0, device=DEV)
xr = torch.arange(0, blocksize, device=DEV)
x2 = torch.randn((blocksize, N1), dtype=DTYPE, device=DEV)
torch_ref = torch_ldst_indirect_08_func(xr, xc, x2)
triton_cal = triton_ldst_indirect_08_func(xc, x2, blocksize, lowdimsize)
torch.testing.assert_close(triton_cal, torch_ref)
```
