# triton.language.make_block_ptr

## 1. OP 概述

简介：创建指向GM上张量的指针
原型：

```python
triton.language.make_block_ptr(
 base: triton.PointerType,
 shape: List[tensor],
 strides: tuple(int | constexpr),
 offsets: tuple(int | constexpr),
 block_shape:tuple(int | constexpr),
 order:tuple(constexpr),
 _semantic=None
)
```

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `base`        | `triton.PointerType`          | 张量的基指针                                                      |
| `shape`       | `tuple(int \| constexpr)`    | 张量在GM上的形状                                                        |
| `strides`     | `tuple(int \| constexpr)`    | 张量各维度的步长列表 |
| `offsets`     | `tuple(int \| constexpr)`    | 张量各维度的基址偏移量列表 |
| `block_shape` | `tuple(constexpr)` | 单次从全局内存加载 / 存储的块的形状                                              |
| `order` | `tuple(constexpr)` | 单次从全局内存加载 / 存储的块的形状                                              |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用                                                |

返回值：`pointer_type<blocked<shape, element_type>>`： 指向tensor的指针

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU    | √    | √     | √     | √     | √      | √      | √      | √     | √    | √    | √    | √    | ×    |
| Ascend A2/A3 | √    | √     | √     | ×     | ×      | ×      | ×      | √     | √    | √    | ×    | √    | ×    |

结论：Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

#### 2.2.2 Shape 支持

|        | 支持维度范围          |
| ------ | --------------- |
| GPU    | 仅支持 1~5维 tensor |
| Ascend A2/A3 | 仅支持 1~5维 tensor |

结论：在 Shape 方面，GPU 与 Ascend 平台无差异，均支持 1 至 5 维张量。

#### 2.2.3 社区约束

`tl.make_block_ptr`的结果不允许进行算数运算，需要改变偏移量时，可以通过：

1. 重新调用make_block_ptr，修改`offset`参数:

   ```python
   for block_idx in range(pid, NUM_BLOCKS, 20):
       task_hz_idx = block_idx // NUM_BLOCKS_M
       task_m_idx = block_idx % NUM_BLOCKS_M
       off_z = task_hz_idx // H
       off_h = task_hz_idx % H
       qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
       # Create block pointers for Q, K, V, Output
       Q_block_ptr = tl.make_block_ptr(
           base=Q + qvk_offset,
           shape=(N_CTX, HEAD_DIM),
           strides=(stride_qm, stride_qk),
           offsets=(task_m_idx * BLOCK_M, 0),
           block_shape=(BLOCK_M, HEAD_DIM),
           order=(1, 0),
       )
   ```

2. 调用`tl.advance`调整偏移量:

   ```python
   block_ptr_in=tl.make_block_ptr(
       base = x_ptr,
       shape = (XB,YB,ZB),
       strides = (YB*ZB,ZB,1),
       offsets = (9,6,5),
       block_shape = (XB,YB,ZB),
       order = (2,1,0),
   )
   bbptr = tl.advance(block_ptr_in,(-9,-6,-5))
   ```

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

- Ascend 对比 GPU 缺失uint8、uint16、uint32、uint64、fp64的支持能力（硬件限制）。

- Ascend只允许通过调整`order`参数的顺序来表达转置语义，不能通过调整`stride`参数的顺序实现转置语义。

- | 差异点                                 | 描述                                                         | 解决途径                       |
  | -------------------------------------- | ------------------------------------------------------------ | ------------------------------ |
  | 与分支、循环语句搭配使用时的泛化性问题 | 当前`tl.make_tensor_ptr`，如果与较复杂的循环和分支语句搭配使用，可能会出现编译问题 | 大量泛化测试暴露问题，迭代解决 |

### 2.4 使用方法

以下示例实现了通过tl.make_block_ptr读取一个tensor并将其转置的功能：

```python
import torch
import torch_npu
import triton
import triton.language as tl
import pytest
import test_common

@pytest.mark.parametrize('shape', [(1, 4, 2)])
@pytest.mark.parametrize('permute_order', [(2, 0, 1)])
def test_makeblockptr_order(shape, permute_order):

    @triton.jit
    def triton_kernel(in0_ptr: tl.tensor, # of tl.pointer_type
                out0_ptr: tl.tensor, # of tl.pointer_type
                in0_stride0: int, in0_stride1: int, in0_stride2: int, # strides for in0
                in0_stride_order0: tl.constexpr, in0_stride_order1: tl.constexpr, in0_stride_order2: tl.constexpr, # stride order for in0
                out0_stride0: int, out0_stride1: int, out0_stride2: int, # strides for out0
                out0_stride_order0: tl.constexpr, out0_stride_order1: tl.constexpr, out0_stride_order2: tl.constexpr, # stride order for out0
                s0: int, s1: int, s2: int,
                tile_size0: tl.constexpr, tile_size1: tl.constexpr, tile_size2: tl.constexpr,
                ):
        tile_id0 = tl.program_id(axis=0)
        tile_id1 = tl.program_id(axis=1)
        tile_id2 = tl.program_id(axis=2)
        offset0 = (tile_id0 * tile_size0).to(tl.int32)
        offset1 = (tile_id1 * tile_size1).to(tl.int32)
        offset2 = (tile_id2 * tile_size2).to(tl.int32)
        in0_bptr = tl.make_block_ptr(in0_ptr,
                                    (s0, s1, s2),
                                    (in0_stride0, in0_stride1, in0_stride2),
                                    (offset0, offset1, offset2),
                                    (tile_size0, tile_size1, tile_size2),
                                    order=(in0_stride_order0, in0_stride_order1, in0_stride_order2))
        in0 = tl.load(in0_bptr, boundary_check=(in0_stride_order0, in0_stride_order1, in0_stride_order2)).to(in0_ptr.type.element_ty)

        out0 = in0

        out0_bptr = tl.make_block_ptr(out0_ptr, (s0, s1, s2), (out0_stride0, out0_stride1, out0_stride2), (offset0, offset1, offset2), (tile_size0, tile_size1, tile_size2),
                                    order=(out0_stride_order0, out0_stride_order1, out0_stride_order2))
        tl.store(out0_bptr, out0.to(out0_bptr.type.element_ty), boundary_check=(out0_stride_order0, out0_stride_order1, out0_stride_order2))

    def triton_func(in0: torch.Tensor, permute_order):
        # in fact, it adjusts the layout metadata instead of doing a real permutation.
        in0_permuted_tmp = in0.permute(permute_order)
        in0_permuted_shape = in0_permuted_tmp.size()
        in0_permuted_strides = in0_permuted_tmp.stride()
        in0_stride_order = [len(permute_order)-1-i for i in permute_order]
        shape = (in0_permuted_shape[0], in0_permuted_shape[1], in0_permuted_shape[2])
        tile_sizes = (shape[0], shape[1], shape[2])
        out0 = torch.empty(shape, dtype=in0.dtype, device=in0.device)
        out0_strides = out0.stride()
        out0_stride_order = [len(permute_order)-1-i for i in range(len(permute_order))]
        grid = (shape[0]//tile_sizes[0], shape[1]//tile_sizes[1], shape[2]//tile_sizes[2])
        triton_kernel[grid](
                in0, out0,
                in0_permuted_strides[0], in0_permuted_strides[1], in0_permuted_strides[2], # stride for in0
                in0_stride_order[0], in0_stride_order[1], in0_stride_order[2], # stride order for in0
                out0_strides[0], out0_strides[1], out0_strides[2], # stride for out0
                out0_stride_order[0], out0_stride_order[1], out0_stride_order[2], # stride orderfor out0
                shape[0], shape[1], shape[2], # task indexing space
                tile_size0=tile_sizes[0],
                tile_size1=tile_sizes[1],
                tile_size2=tile_sizes[2],
            )
        return out0

    x0 = torch.randint(0, 9, shape, dtype=torch.int32).npu()
    torch_ref = torch.permute(x0, permute_order)
    triton_cal = triton_func(x0, permute_order)
    test_common.validate_cmp("int32", triton_cal, torch_ref)

```
