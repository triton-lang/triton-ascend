# triton.language.atomic_and

## 1. OP 概述

简介：原子性逻辑与操作，在指定的内存位置执行逻辑与
原型：

```python
triton.language.atomic_and(
 pointer, 
 val, 
 mask=None, 
 sem=None, 
 scope=None, 
 _semantic=None
)→ pointer
```

可以作为tensor的成员函数调用，如`x.atomic_and(...)`，与`atomic_and(x, ...)`等效。

## 2. OP 规格

### 2.1 参数说明

| 参数名           | 类型                | 说明                                                             |
| ------------- | ----------------- | -------------------------------------------------------------- |
| `pointer`        | `triton.PointerDType`          | 要操作的内存位置，执行 *pointer & val 计算后的结果写回到该内存                                                     |
| `val`       | `pointer.dtype.element_ty`    | 执行原子与操作的值（右操作数）                                                        |
| `mask`     | `int1`或`tensor<int1>`，可选    | 指定数据范围，防止访问越界 |
| `sem` | `str`，可选 | 指定操作的内存语义<br>社区官方配置可接受的值为“acquire”、“release”、“acq_rel”（默认，代表“ACQUIRE_RELEASE”）和“relaxed”<br>我们只支持“acq_rel”：<br>- acquire：获取锁后，能够看到之前的释放操作（相当于一个“读取”操作，并且这个读取操作会阻塞，直到能够读取到“最新”的数据，也就是其他线程释放后的数据）<br>- release：在释放锁之前的所有操作，对后续获取锁的线程可见（相当于一个“写入”操作，并且这个写入操作会“同步”所有之前的写操作）                                             |
| `scope` | `str`，可选 | 观察原子操作同步效果的线程范围<br>可接受的值为“gpu”（默认）、“cta”（协作线程数组、线程块）或“sys”（代表“SYSTEM”） <br>我们只支持“gpu”                                            |
| `_semantic`   | -                 | 保留参数，暂不支持外部调用                                              |

返回值：
`pointer`：tensor，执行操作之前的旧值

### 2.2 支持规格

#### 2.2.1 DataType 支持

|        | int8 | int16 | int32 | uint8 | uint16 | uint32 | uint64 | int64 | fp16 | fp32 | fp64 | bf16 | bool |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ----- | ---- | ---- | ---- | ---- | ---- |
| GPU   | ×     | ×      |√     | ×     | ×      | ×      | ×      | √     | ×    | ×    | ×    | ×    | ×    |
| Ascend A2/A3| √    | √     | √     | √     | √      | √      | √      | √     | ×    | ×    | ×    | ×    | ×    |

#### 2.2.2 Shape 支持

无特殊要求

### 2.3 特殊限制说明

> 相对社区能力缺失且无法实现

| 差异点                   | 描述                                                                           | 解决途径                          |
| --------------------- | ---------------------------------------------------------------------------- | ----------------------------- |
|sem| 社区官方配置可接受的值为“acquire”、“release”、“acq_rel”（默认，代表“ACQUIRE_RELEASE”）和“relaxed”<br>我们只支持“acq_rel” | 待开发 |
|scope               | 可接受的值为“gpu”、“cta”、或“sys”、 <br>我们只支持“gpu” | 待开发 |

### 2.4 使用方法

以下示例实现了原子与计算：

```python
@triton.jit
def atomic_and(in_ptr0, out_ptr0, out_ptr1, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)[:]
    yindex = tl.arange(0, BLOCK_SIZE)[:]
    xmask = xindex < n_elements
    x0 = xindex
    x1 = yindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl.atomic_and(out_ptr0 + (x1), tmp0, xmask)
    tl.store(out_ptr1 + (x1), tmp1, xmask

dtype, shape, ncore = ['int32', (32, 32), 2]
block_size = shape[0] * shape[1] // ncore
split_size = shape[0] // ncore
val = torch.randint(low=0, high=10, size=shape, dtype=eval(f'torch.{dtype}')).npu()
pointer = torch.randint(low=0, high=10, size=(split_size, shape[1]), dtype=eval(f'torch.{dtype}')).npu()
pointer_old = torch.full_like(pointer, -10).npu()
n_elements = shape[0] * shape[1]
atomic_and[ncore, 1, 1](val, pointer, pointer_old, n_elements, BLOCK_SIZE=split_size * shape[1])
```
