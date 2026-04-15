# triton.language.num_programs
## 1. OP 概述

简介：返回沿给定 axis 启动的程序实例数量
函数原型：

```
triton.language.num_programs(axis)
```


## 2. OP 规格

### 2.1 参数说明

| 参数名 | 类型 | 说明 |
| :---: | :---: | :---: |
| `axis` | `int` | 3D 启动网格的轴。必须是 0、1 或 2。 |

返回值：
由启动的程序实例数量值组成的tl.tensor

### 2.2 支持规格

#### 2.2.1 DataType 支持

|       | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 |fp16 | fp32 | fp64 | bf16 | bool |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| GPU          | × | × | √ | × | × | × | × | × | × | × | × | × | × |
| Ascend A2/A3 | × | × | √ | × | × | × | × | × | × | × | × | × | × |



#### 2.2.2 Shape 支持

无相关设置


### 2.3 特殊限制说明

无


### 2.4 使用方法

例子可以参考[test_3Dgrid.py](../../../../ascend/examples/pytest_ut/test_3Dgrid.py)

```
@triton.jit
def triton_(in_ptr0, out_ptr0, x0_numel, r1_numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr,
            block_id_threshold: tl.constexpr, XBLOCK1: tl.constexpr, num_core: tl.constexpr):
    RBLOCK: tl.constexpr = 64

    block_idx=tl.program_id(0)*tl.num_programs(1)*tl.num_programs(2)+tl.program_id(1)*tl.num_programs(2)+tl.program_id(2)
    if (block_idx < block_id_threshold):
        offset = block_idx * XBLOCK
        loops1 = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB # 32+23 / 24 = 2
        upper = offset + XBLOCK
    else:
        offset = block_id_threshold * XBLOCK + (block_idx - block_id_threshold) * XBLOCK1 #pid=34 offset = 9*32 + (34-9)*24 = 888
        loops1 = (XBLOCK1 + XBLOCK_SUB - 1) // XBLOCK_SUB #1
        if (block_idx ==num_core -1):
            upper = x0_numel
        else:
            upper = offset + XBLOCK1 # 912

    base1 = tl.arange(0, XBLOCK_SUB)
    base2 = tl.arange(0, RBLOCK)
    loops2: tl.constexpr = (r1_numel + RBLOCK - 1) // RBLOCK
    for loop1 in range(loops1):
        x = offset + (loop1 * XBLOCK_SUB) + base1
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1[None, :]
        x0 = offset + (loop1 * XBLOCK_SUB) + base1[:, None]
        xmask = x0 < upper
        r1_prime = base2[:, None]
        rindex = base2
        r1 = base2[None, :]
        rmask = r1 < r1_numel
        tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask,other=0.0)

        tmp1 = tl.reshape(tmp0, [XBLOCK_SUB, RBLOCK])
        tmp2_tmp = tl.sum(tmp1,1)
        tmp2 = tmp2_tmp.reshape(XBLOCK_SUB,1)

        tl.store(out_ptr0 + (x0), tmp2, xmask)

```
